import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from typing import Optional, Tuple, Union


class PipelineWrapper(torch.nn.Module):
    def __init__(self, model_id: str,
                 timesteps: int,
                 device: torch.device,
                 float16: bool = False,
                 compile: bool = False,
                 token: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self.num_timesteps = timesteps
        self.device = device
        self.float16 = float16
        self.token = token
        self.compile = compile
        self.model = None

    @property
    def timesteps(self) -> torch.Tensor:
        return self.model.scheduler.timesteps

    @property
    def dtype(self) -> torch.dtype:
        if self.model is None:
            raise AttributeError("Model is not initialized.")
        return self.model.unet.dtype

    def get_x_0_hat(self, xt: torch.Tensor, epst: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.model.scheduler.get_x_0_hat(xt, epst, timestep)

    def finish_step(self, xt: torch.Tensor, pred_x0: torch.Tensor, epst: torch.Tensor,
                    timestep: torch.Tensor, variance_noise: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        return self.model.scheduler.finish_step(xt, pred_x0, epst, timestep, variance_noise, **kwargs)

    def get_variance(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.model.scheduler.get_variance(timestep)

    def set_timesteps(self, timesteps: int, device: torch.device) -> None:
        self.model.scheduler.set_timesteps(timesteps, device=device)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def decode_image(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def encode_prompt(self, prompt: torch.Tensor, negative_prompt=None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_epst(self, xt: torch.Tensor, t: torch.Tensor, prompt_embeds: torch.Tensor,
                 guidance_scale: Optional[float] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def get_image_size(self) -> Tuple[int, int]:
        return self.model.unet.config.sample_size * self.model.vae_scale_factor

    def get_noise_shape(self, imsize: Union[int, Tuple[int]], batch_size: int) -> Tuple[int, ...]:
        if isinstance(imsize, int):
            imsize = (imsize, imsize)
        variance_noise_shape = (batch_size,
                                self.model.unet.config.in_channels,
                                imsize[-2],
                                imsize[-1])
        return variance_noise_shape

    def get_latent_shape(self, orig_image_shape: Union[int, Tuple[int, int]]) -> Tuple[int, ...]:
        if isinstance(orig_image_shape, int):
            orig_image_shape = (orig_image_shape, orig_image_shape)
        return (self.model.unet.config.in_channels,
                orig_image_shape[0] // self.model.vae_scale_factor,
                orig_image_shape[1] // self.model.vae_scale_factor)

    def get_pre_kwargs(self, **kwargs) -> dict:
        return {}


class StableDiffWrapper(PipelineWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            self.model = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.float16 else torch.float32,
                token=self.token).to(self.device)
        except OSError:
            self.model = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.float16 else torch.float32,
                token=self.token, force_download=True
                ).to(self.device)

        self.model.scheduler = DDIMWrapper(model_id=self.model_id, device=self.device,
                                           eta=1.0, float16=self.float16, token=self.token)

        self.model.scheduler.set_timesteps(self.num_timesteps, device=self.device)
        if self.compile:
            try:
                self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)
            except Exception as e:
                print(f"Error compiling model: {e}")

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return (self.model.vae.encode(x).latent_dist.mode() * self.model.vae.config.scaling_factor)  # .float()

    def decode_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            orig_device = self.model.vae.device
            self.model.vae.to(x.device)
            ret = self.model.vae.decode(x / self.model.vae.config.scaling_factor).sample.clamp(-1, 1)
            self.model.vae.to(orig_device)
            return ret
        return self.model.vae.decode(x / self.model.vae.config.scaling_factor).sample.clamp(-1, 1)

    def encode_prompt(self, prompt: torch.Tensor, negative_prompt=None) -> Tuple[torch.Tensor, torch.Tensor]:
        do_cfg = (negative_prompt is not None) or prompt != ""

        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt, self.device, 1,
            do_cfg,
            negative_prompt,
        )

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def get_epst(self, xt: torch.Tensor, t: torch.Tensor, prompt_embeds: torch.Tensor,
                 guidance_scale: Optional[float] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        do_cfg = prompt_embeds.shape[0] > 1
        xt = torch.cat([xt] * 2) if do_cfg else xt

        # predict the noise residual
        noise_pred = self.model.unet(xt, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

        # perform guidance
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            return noise_pred, noise_pred_uncond, noise_pred_text
        return noise_pred, noise_pred, None


class SchedulerWrapper(torch.nn.Module):
    def __init__(self, model_id: str, device: torch.device,
                 float16: bool = False, token: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self.device = device
        self.float16 = float16
        self.token = token
        self.scheduler = None

    @property
    def timesteps(self) -> torch.Tensor:
        return self.scheduler.timesteps

    def set_timesteps(self, timesteps: int, device: torch.device) -> None:
        self.scheduler.set_timesteps(timesteps, device=device)
        if self.scheduler.timesteps[0] == 1000:
            self.scheduler.timesteps -= 1

    def get_x_0_hat(self, xt: torch.Tensor, epst: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        pass

    def finish_step(self, xt: torch.Tensor, pred_x0: torch.Tensor, epst: torch.Tensor,
                    timestep: torch.Tensor, variance_noise: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        pass

    def get_variance(self, timestep: torch.Tensor) -> torch.Tensor:
        pass


class DDIMWrapper(SchedulerWrapper):
    def __init__(self, eta, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler",
            torch_dtype=torch.float16 if self.float16 else torch.float32,
            token=self.token,
            device=self.device, timestep_spacing='linspace')
        self.eta = eta

    def get_x_0_hat(self, xt: torch.Tensor, epst: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        # compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.scheduler.config.prediction_type == 'epsilon':
            pred_original_sample = (xt - beta_prod_t ** (0.5) * epst) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == 'v_prediction':
            pred_original_sample = (alpha_prod_t ** 0.5) * xt - (beta_prod_t ** 0.5) * epst

        return pred_original_sample

    def finish_step(self, xt: torch.Tensor, pred_x0: torch.Tensor, epst: torch.Tensor,
                    timestep: torch.Tensor, variance_noise: torch.Tensor,
                    eta=None) -> torch.Tensor:
        if eta is None:
            eta = self.eta
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // \
            self.scheduler.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self._get_alpha_prod_t_prev(prev_timestep)
        beta_prod_t = 1 - alpha_prod_t

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self.get_variance(timestep)
        std_dev_t = eta * variance ** (0.5)

        # std_dev_t = eta * variance ** (0.5)
        # Take care of asymetric reverse process (asyrp)
        if self.scheduler.config.prediction_type == 'epsilon':
            model_output_direction = epst
        elif self.scheduler.config.prediction_type == 'v_prediction':
            model_output_direction = (alpha_prod_t**0.5) * epst + (beta_prod_t**0.5) * xt

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_x0 + pred_sample_direction

        # 8. Add noice if eta > 0
        if eta > 0:
            sigma_z = std_dev_t * variance_noise
            prev_sample = prev_sample + sigma_z

        return prev_sample

    def get_variance(self, timestep: torch.Tensor) -> torch.Tensor:
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // \
            self.scheduler.num_inference_steps
        variance = self.scheduler._get_variance(timestep, prev_timestep)
        return variance

    def _get_alpha_prod_t_prev(self, prev_timestep: torch.Tensor) -> torch.Tensor:
        return self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 \
            else self.scheduler.final_alpha_cumprod


def load_model(model_id: str, timesteps: int,
               device: torch.device,
               float16: bool = False, token: Optional[str] = None,
               compile: bool = False) -> PipelineWrapper:
    pipeline = StableDiffWrapper(model_id=model_id, timesteps=timesteps, device=device,
                                 float16=float16, token=token, compile=compile)

    pipeline = pipeline.to(device)
    image_size = pipeline.get_image_size()
    return pipeline, image_size
