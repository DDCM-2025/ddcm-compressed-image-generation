from typing import Tuple, Optional, Dict, List
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import random
import pyiqa

from ddcm.util.img_utils import clear_color
from .posterior_mean_variance import get_mean_processor, get_var_processor


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

__SAMPLER__ = {}

def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!") 
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing=""):
    
    sampler = get_sampler(name=sampler)
    
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
         
    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised, 
                   rescale_timesteps=rescale_timesteps)


class GaussianDiffusion:
    def __init__(self,
                 betas,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <=1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)    
    
        self.var_processor = get_var_processor(model_var_type,
                                               betas=betas)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        
        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)

        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(self.posterior_log_variance_clipped, t, x_t)

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    torch.no_grad()
    def p_sample_loop_compression(self,
                                  model,
                                  img_to_compress: Optional[torch.Tensor],
                                  image_size: int,
                                  device: torch.device,
                                  num_noises: int,
                                  num_pursuit_noises: Optional[int] = 1,
                                  num_pursuit_coef_bits: Optional[int] = 3,
                                  decompress_indices: Optional[Dict[str, List[List[int]]]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The function used for sampling from noise.
        """

        decompress = img_to_compress is None
        if decompress and decompress_indices is None:
            raise ValueError("Either img_to_compress or loaded_indices must be provided.")

        set_seed(100000)
        im_shape = (3, image_size, image_size)
        img = torch.randn(1, *im_shape, device=device)

        best_indices_list = []
        best_coeffs_list = []

        # TODO add decompression
        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            set_seed(idx)
            time = torch.tensor([idx] * img.shape[0], device=device)
            noise = torch.randn(num_noises, *img.shape[1:], device=device)
            out = self.p_sample(x=img,
                                t=time,
                                model=model,
                                noise=noise,
                                ref=img_to_compress,
                                loss_type='dot_prod',
                                eta=1,
                                num_pursuit_noises=num_pursuit_noises,
                                num_pursuit_coef_bits=num_pursuit_coef_bits)
            best_indices = out['best_indices']
            best_coeffs = out['best_coeffs']
            best_indices_list.append(best_indices)
            best_coeffs_list.append(best_coeffs)

            img = out['sample']
            del noise

        return img, torch.tensor(best_indices_list, dtype=torch.int).cpu().squeeze(), torch.tensor(best_coeffs_list, dtype=torch.int).cpu().squeeze()

    @torch.no_grad()
    def p_sample_loop_blind_restoration(self,
                                        model,
                                        x_start,
                                        mmse_img,
                                        num_opt_noises,
                                        iqa_metric,
                                        iqa_coef,
                                        eta,
                                        loaded_indices):

        assert iqa_metric == 'niqe' or iqa_metric == 'clipiqa+' or iqa_metric == 'topiq_nr-face'
        iqa = pyiqa.create_metric(iqa_metric, device=x_start.device)
        device = x_start.device

        set_seed(100000)
        img = torch.randn(2, *x_start.shape[1:], device=device)

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        next_idx = np.array([0, 1])
        if loaded_indices is not None:
            indices = loaded_indices
            assert loaded_indices.shape[1] == 1
            loaded_indices = torch.cat((loaded_indices[:, 0], torch.tensor([0], device=device, dtype=loaded_indices.dtype)), dim=0)
        else:
            indices = []
        for i, idx in enumerate(pbar):
            set_seed(idx)

            noise = torch.randn(num_opt_noises, *img.shape[1:], device=device)
            if loaded_indices is None:
                time = torch.tensor([idx] * img.shape[0], device=device)
                out = self.p_sample(x=img,
                                    t=time,
                                    model=model,
                                    noise=noise,
                                    ref=mmse_img,
                                    loss_type='dot_prod',
                                    optimize_iqa=True,
                                    eta=eta,
                                    iqa=iqa,
                                    iqa_coef=iqa_coef)
                img = out['sample']
                best_perceptual_idx_cur = out['best_perceptual_idx']
                indices.append(next_idx[best_perceptual_idx_cur])
                next_idx = out['best_idx']
            else:
                time = torch.tensor([idx], device=device)
                if i == 0:
                    img = img[loaded_indices[0]].unsqueeze(0)
                out = self.p_sample(x=img,
                                    t=time,
                                    model=model,
                                    noise=noise[loaded_indices[i+1]].unsqueeze(0),
                                    ref=img,
                                    loss_type='dot_prod',
                                    optimize_iqa=False,
                                    eta=eta,
                                    iqa='niqe',
                                    iqa_coef=0.0)
                img = out['sample']

        if type(indices) is list:
            indices = torch.tensor(indices).flatten()
        return img[0].unsqueeze(0), indices

    @torch.no_grad()
    def p_sample_loop_linear_restoration(self,
                                         model,
                                         x_start,
                                         ref_img,
                                         linear_operator,
                                         y_n,
                                         num_opt_noises,
                                         eta,
                                         loaded_indices):
        set_seed(100000)
        device = x_start.device
        img = torch.randn(1, *x_start.shape[1:], device=device)

        pbar = tqdm(list(range(self.num_timesteps))[::-1])

        if loaded_indices is not None:
            indices = loaded_indices
            assert loaded_indices.shape[1] == 1
        else:
            indices = []



        for i, idx in enumerate(pbar):
            set_seed(idx)
            time = torch.tensor([idx] * img.shape[0], device=device)

            noise = torch.randn(num_opt_noises, *img.shape[1:], device=device)
            if loaded_indices is None:
                out = self.p_sample(x=img,
                                    t=time,
                                    model=model,
                                    noise=noise,
                                    ref=ref_img,
                                    loss_type='mse',
                                    eta=eta,
                                    y_n=y_n,
                                    linear_operator=linear_operator,
                                    optimize_iqa=False,
                                    iqa=None,
                                    iqa_coef=None)
                img = out['sample']
                indices.append(out['best_indices'])
            else:
                class Dummy:
                    def forward(self, x):
                        return x
                out = self.p_sample(x=img,
                                    t=time,
                                    model=model,
                                    noise=noise[loaded_indices[i].squeeze()].unsqueeze(0),
                                    ref=torch.zeros_like(img), # Dummy
                                    loss_type='mse',
                                    eta=eta,
                                    y_n=torch.zeros_like(img), # Dummy,
                                    linear_operator=Dummy(), # Dummy
                                    optimize_iqa=False,
                                    iqa=None,
                                    iqa_coef=None)
                img = out['sample']
            del noise
        if type(indices) is list:
            indices = torch.tensor(indices).flatten()
        return img, indices

    def p_sample(self, model, x, t, noise, ref, loss_type, eta=None):
        raise NotImplementedError

    def p_mean_variance(self, model, x, t):
        model_output = model(x, self._scale_timesteps(t))
        
        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            # The name of variable is wrong. 
            # This will just provide shape information, and 
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}
  
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddpm')
class DDPM(SpacedDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def p_sample(self, model, x, t, noise, ref, eta=None):
        out = self.p_mean_variance(model, x, t)
        pred_xstart = out['pred_xstart']

        loss = torch.matmul(noise.view(noise.shape[0], -1), (ref - pred_xstart).view(pred_xstart.shape[0], -1).transpose(0, 1))
        best_idx = torch.argmax(loss)
        samples = out['mean'] + torch.exp(0.5 * out['log_variance']) * noise[best_idx].unsqueeze(0)

        return {'sample': samples if t[0] > 0 else pred_xstart,
                'pred_xstart': pred_xstart,
                'mse': loss[best_idx].item(),
                'best_idx': best_idx}


@register_sampler(name='ddim')
class DDIM(SpacedDiffusion):
    @torch.no_grad()
    def p_sample(self, model, x, t, noise, ref, loss_type='mse', eta=0.0, iqa=None, iqa_coef=1.0,
                 optimize_iqa=False, linear_operator=None, y_n=None, random_opt_mse_noises=0,
                 num_pursuit_noises=1, num_pursuit_coef_bits=1,
                 cond_fn=None,
                 cls=None
                 ):

        out = self.p_mean_variance(model, x, t)
        pred_xstart = out['pred_xstart']
        best_perceptual_idx = None
        if optimize_iqa:
            assert not random_opt_mse_noises
            coef_sign = 1 if iqa.lower_better else -1
            if iqa.metric_name == 'topiq_nr-face':
                assert not iqa.lower_better
                # topiq_nr-face doesn't support a batch size larger than 1.
                scores = []
                for elem in pred_xstart:
                    try:
                        scores.append(iqa((elem.unsqueeze(0) * 0.5 + 0.5).clip(0, 1)).squeeze().view(1))
                    except AssertionError:
                        # no face detected...
                        scores.append(torch.zeros(1, device=x.device))
                scores = torch.stack(scores, dim=0).squeeze()
                loss = (((ref - pred_xstart) ** 2).view(pred_xstart.shape[0], -1).mean(1) + coef_sign * iqa_coef * scores)
            else:
                loss = (((ref - pred_xstart) ** 2).view(pred_xstart.shape[0], -1).mean(1) + coef_sign * iqa_coef * iqa((pred_xstart * 0.5 + 0.5).clip(0, 1)).squeeze())
            best_perceptual_idx = torch.argmin(loss)
            out['pred_xstart'] = out['pred_xstart'][best_perceptual_idx].unsqueeze(0)
            pred_xstart = pred_xstart[best_perceptual_idx].unsqueeze(0)
            t = t[best_perceptual_idx]
            x = x[best_perceptual_idx].unsqueeze(0)
        elif random_opt_mse_noises > 0:
            loss = (((ref - pred_xstart) ** 2).view(pred_xstart.shape[0], -1).mean(1))
            best_mse_idx = torch.argmin(loss)
            out['pred_xstart'] = out['pred_xstart'][best_mse_idx].unsqueeze(0)
            pred_xstart = pred_xstart[best_mse_idx].unsqueeze(0)
            t = t[best_mse_idx]
            x = x[best_mse_idx].unsqueeze(0)

        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        mean_pred = (
                out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        sample = mean_pred

        if y_n is not None:
            assert linear_operator is not None
        y_n = ref if y_n is None else y_n

        if not optimize_iqa and random_opt_mse_noises <= 0 and cond_fn is None:
            if loss_type == 'dot_prod':
                if linear_operator is None:
                    compute_loss = lambda noise_cur: torch.matmul(noise_cur.view(noise_cur.shape[0], -1), (ref - pred_xstart).view(pred_xstart.shape[0], -1).transpose(0, 1))
                else:
                    compute_loss = lambda noise_cur: torch.matmul(linear_operator.forward(noise_cur).reshape(noise_cur.shape[0], -1), (y_n -  linear_operator.forward(pred_xstart)).reshape(pred_xstart.shape[0], -1).transpose(0, 1))
            elif loss_type == 'mse':
                if linear_operator is None:
                    compute_loss = lambda noise_cur: - (((sigma / torch.sqrt(alpha_bar_prev)) * noise_cur + pred_xstart - y_n) ** 2).mean((1, 2, 3))
                else:
                    compute_loss = lambda noise_cur: - (((sigma / torch.sqrt(alpha_bar_prev))[:, :, :y_n.shape[2], :y_n.shape[3]] * linear_operator.forward(noise_cur) + linear_operator.forward(pred_xstart) - y_n) ** 2).mean((1, 2, 3))
            else:
                raise NotImplementedError()
            loss = compute_loss(noise)
            best_idx = torch.argmax(loss)
            best_noises = [best_idx]
            best_coeffs = []
            best_noise = noise[best_idx]
            best_loss = loss[best_idx]

            if num_pursuit_noises > 1:
                # avoid zero, but leave bits for it
                pursuit_coefs = np.linspace(0, 1, 2 ** num_pursuit_coef_bits)[1:]
                for _ in range(num_pursuit_noises - 1):
                    next_best_noise = best_noise
                    next_noise_idx = 0
                    next_coeff_idx = 0
                    for coeff_idx, pursuit_coef in enumerate(pursuit_coefs, 1):
                        new_noise = best_noise.unsqueeze(0) * np.sqrt(pursuit_coef) + noise * np.sqrt(1 - pursuit_coef)
                        new_noise /= new_noise.view(noise.shape[0], -1).std(1).view(noise.shape[0], 1, 1, 1)
                        cur_loss = compute_loss(new_noise)
                        cur_best_idx = torch.argmax(cur_loss)
                        cur_best_loss = cur_loss[cur_best_idx]

                        if cur_best_loss > best_loss:
                            next_best_noise = new_noise[cur_best_idx]
                            best_loss = cur_best_loss
                            next_noise_idx = cur_best_idx
                            next_coeff_idx = coeff_idx
                    best_noises.append(next_noise_idx)
                    best_coeffs.append(next_coeff_idx)
                    best_noise = next_best_noise
            if t != 0:
                sample += sigma * best_noise.unsqueeze(0)

            return {'sample': sample if t[0] > 0 else pred_xstart,
                    'pred_xstart': pred_xstart,
                    'mse': loss[best_idx].item(),
                    'best_indices': torch.tensor(best_noises, dtype=torch.int).cpu(),
                    'best_coeffs': torch.tensor(best_coeffs, dtype=torch.int).cpu()}
        else:
            if random_opt_mse_noises > 0 and not optimize_iqa:
                num_rand_indices = random_opt_mse_noises
            elif optimize_iqa and random_opt_mse_noises <= 0:
                num_rand_indices = 1
            elif cond_fn is not None:
                num_rand_indices = 2
            else:
                raise NotImplementedError()
            loss = torch.matmul(noise.view(noise.shape[0], -1),
                                (ref - pred_xstart).view(pred_xstart.shape[0], -1).transpose(0, 1)).squeeze()
            best_idx = torch.argmax(loss).reshape(1)
            rand_idx = torch.randint(0, noise.shape[0], size=(num_rand_indices, ), device=best_idx.device).reshape(num_rand_indices)
            best_and_rand_idx = torch.cat((best_idx, rand_idx), dim=0).flatten()
            if t != 0:
                sample = sample + sigma * noise[best_and_rand_idx]
            return {'sample': sample,
                    'pred_xstart': pred_xstart,
                    'best_idx': best_and_rand_idx,
                    'best_perceptual_idx': best_perceptual_idx}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2

# =================
# Helper functions
# =================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
   
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)