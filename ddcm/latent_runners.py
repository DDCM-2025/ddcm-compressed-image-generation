import os
import time
from glob import glob
from typing import Callable, Optional, Tuple, Union, Dict, List
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from tqdm import tqdm
from .util.img_utils import clear_color

from .latent_models import PipelineWrapper


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class MinusOneToOne(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 2 - 1


class ResizePIL(torch.nn.Module):
    def __init__(self, image_size: Optional[Union[int, Tuple[int, int]]] = None):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

    def forward(self, pil_image: Image.Image) -> Image.Image:
        if self.image_size is not None and pil_image.size != self.image_size:
            pil_image = pil_image.resize(self.image_size)
        return pil_image


def get_loader(datadir: str, batch_size: int = 1,
               resize_to: Optional[Union[int, Tuple[int, int]]] = None) -> DataLoader:
    transform = transforms.Compose([
        ResizePIL(resize_to),
        transforms.ToTensor(),
        MinusOneToOne(),
    ])
    loader = DataLoader(FoldersDataset(datadir, transform),
                        batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=False)
    return loader


class FoldersDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None) -> None:
        super().__init__(root, transforms)
        self.root = root

        if os.path.isdir(root):
            self.fpaths = glob(os.path.join(root, '**', '*.png'), recursive=True)
            self.fpaths += glob(os.path.join(root, '**', '*.JPEG'), recursive=True)
            self.fpaths += glob(os.path.join(root, '**', '*.jpg'), recursive=True)
            self.fpaths = sorted(self.fpaths)
            assert len(self.fpaths) > 0, "File list is empty. Check the root."
        elif os.path.exists(root):
            self.fpaths = [root]
        else:
            raise FileNotFoundError(f"File not found: {root}")

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, fpath[len(self.root) + 1:]


def compress(model: PipelineWrapper,
             img_to_compress: Optional[torch.Tensor],
             num_noises: int,
             device: torch.device,
             num_pursuit_noises: Optional[int] = 1,
             num_pursuit_coef_bits: Optional[int] = 3,
             t_range: Tuple[int, int] = (999, 0),
             decompress_indices: Optional[Dict[str, List[List[int]]]] = None,
             # ---- editing -----
             edit_src_prompt: str = "",
             edit_dst_prompt: str = "",
             guidance_scale: float = 0.0,
             t_edit: int = 0,
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    decompress = img_to_compress is None
    if decompress and decompress_indices is None:
        raise ValueError("Either img_to_compress or loaded_indices must be provided.")
    edit = (edit_dst_prompt != edit_src_prompt) and decompress

    model.set_timesteps(model.num_timesteps, device=device)
    dtype = model.dtype

    src_prompt_embeds = model.encode_prompt(edit_src_prompt, None)
    dst_prompt_embeds = src_prompt_embeds if not edit else model.encode_prompt(edit_dst_prompt, None)

    set_seed(88888888)
    if decompress:
        im_shape = model.get_image_size()
        im_shape = (im_shape, im_shape)
        gen_shape = model.get_latent_shape(im_shape)
    else:
        enc_im = model.encode_image(img_to_compress.to(dtype))
        im_shape = img_to_compress.shape
        gen_shape = enc_im.shape[1:]

    src_kwargs = model.get_pre_kwargs(height=im_shape[-2], width=im_shape[-1],
                                      prompt_embeds=src_prompt_embeds)
    dst_kwargs = src_kwargs if not edit else model.get_pre_kwargs(height=im_shape[-2], width=im_shape[-1],
                                                                  prompt_embeds=dst_prompt_embeds)

    set_seed(100000)
    xt = torch.randn(1, *gen_shape, device=device, dtype=dtype)

    noise_indices = []
    coeffs_indices = []
    curr_decomp_optimized_step_indx = 0
    if decompress:
        noise_indices = decompress_indices['noise_indices']
        coeffs_indices = decompress_indices['coeff_indices']
    pbar = tqdm(model.timesteps)
    for idx, t in enumerate(pbar):
        set_seed(idx)

        # t_range: optimize only in range
        optimize_t = (t_range[0] >= t >= t_range[1])
        
        noise = torch.randn(num_noises if optimize_t else 1, *xt.shape[1:], device=device, dtype=dtype)

        curr_prompt_embeds = src_prompt_embeds if idx < t_edit else dst_prompt_embeds
        curr_kwargs = src_kwargs if idx < t_edit else dst_kwargs

        epst, _, _ = model.get_epst(xt, t, curr_prompt_embeds, guidance_scale, **curr_kwargs)
        x_0_hat = model.get_x_0_hat(xt, epst, t)
        if not decompress:
            if t >= 1 and optimize_t:
                dot_prod = torch.matmul(noise.view(noise.shape[0], -1),
                                        (enc_im - x_0_hat).view(enc_im.shape[0], -1).transpose(0, 1))
                best_idx = torch.argmax(dot_prod)

                t_noises = [best_idx]
                t_coeffs = []
                best_noise = noise[best_idx]
                best_dot_prod = dot_prod[best_idx]
                if num_pursuit_noises > 1:
                    pursuit_coefs = torch.linspace(0, 1, 2 ** num_pursuit_coef_bits)[1:]  # avoid zero, but leave bits for it
                    for _ in range(num_pursuit_noises - 1):
                        next_best_noise = best_noise
                        next_noise_idx = 0
                        next_coeff_idx = 0
                        for coeff_idx, pursuit_coef in enumerate(pursuit_coefs, 1):
                            new_noise = best_noise.unsqueeze(0) * torch.sqrt(pursuit_coef) + noise * torch.sqrt(1 - pursuit_coef)
                            new_noise /= new_noise.view(noise.shape[0], -1).std(1).view(noise.shape[0], 1, 1, 1)
                            cur_dot_prod = torch.matmul(new_noise.view(new_noise.shape[0], -1),
                                                        (enc_im - x_0_hat).view(enc_im.shape[0], -1).transpose(0, 1))
                            cur_best_idx = torch.argmax(cur_dot_prod)
                            cur_best_dot_prod = cur_dot_prod[cur_best_idx]

                            if cur_best_dot_prod > best_dot_prod:
                                next_best_noise = new_noise[cur_best_idx]
                                best_dot_prod = cur_best_dot_prod
                                next_noise_idx = cur_best_idx
                                next_coeff_idx = coeff_idx
                        best_noise = next_best_noise
                        t_noises.append(next_noise_idx)
                        t_coeffs.append(next_coeff_idx)
                noise_indices.append(t_noises)
                coeffs_indices.append(t_coeffs)
            else:
                best_noise = noise[0]
        else:
            if t >= 1 and optimize_t:
                t_noise_indices = noise_indices[curr_decomp_optimized_step_indx]
                best_noise = noise[t_noise_indices[0]]
                pursuit_coefs = torch.linspace(0, 1, 2 ** num_pursuit_coef_bits)  # have a zero here
                if num_pursuit_noises > 1:
                    t_coeffs_indices = coeffs_indices[curr_decomp_optimized_step_indx]
                    for pursuit_idx in range(num_pursuit_noises - 1):
                        if t_coeffs_indices[pursuit_idx] == 0:
                            break  # no more noises added
                        pursuit_coef = pursuit_coefs[t_coeffs_indices[pursuit_idx]]
                        pursuit_noise = noise[t_noise_indices[pursuit_idx + 1]]  # +1 because the first noise was already added
                        best_noise = best_noise * torch.sqrt(pursuit_coef) + pursuit_noise * torch.sqrt(1 - pursuit_coef)
                        best_noise /= best_noise.std()
                curr_decomp_optimized_step_indx +=1
            else:
                best_noise = noise[0]

        xt = model.finish_step(xt, x_0_hat, epst, t, best_noise.unsqueeze(0), eta=None)

    try:
        img = model.decode_image(xt)
    except torch.OutOfMemoryError:
        img = model.decode_image(xt.to('cpu'))

    return img, torch.tensor(noise_indices, dtype=torch.int).cpu().squeeze(), torch.tensor(coeffs_indices, dtype=torch.int).cpu().squeeze()


def generate(model: PipelineWrapper,
             num_noises: int,
             device: torch.device,
             prompt: str = "",
             negative_prompt: Optional[str] = None,
             guidance_scale: float = 7.0,
             ) -> Tuple[torch.Tensor, torch.Tensor]:

    model.set_timesteps(model.num_timesteps, device=device)
    dtype = model.dtype

    random_seeds_for_choices = torch.randint(0, 2**16, (model.num_timesteps+1,), dtype=int)

    set_seed(88888888)
    prompt_embeds = model.encode_prompt(prompt, negative_prompt)
    kwargs = model.get_pre_kwargs(height=model.get_image_size(),
                                  width=model.get_image_size(),
                                  prompt_embeds=prompt_embeds)

    set_seed(100000)
    xt_shape = model.get_latent_shape(model.get_image_size())
    noise = torch.randn(num_noises, *xt_shape, device=device, dtype=dtype)  # Codebook
    set_seed(random_seeds_for_choices[-1].item())
    best_idx = torch.randint(0, num_noises, (1,), device=device)
    xt = noise[best_idx]
    
    pbar = tqdm(model.timesteps)
    for idx, t in enumerate(pbar):
        set_seed(idx)
        noise = torch.randn(num_noises, *xt.shape[1:], device=device, dtype=dtype)  # Codebook

        epst, _, _ = model.get_epst(xt, t, prompt_embeds, guidance_scale, **kwargs)
        x_0_hat = model.get_x_0_hat(xt, epst, t)

        set_seed(random_seeds_for_choices[idx].item())
        best_idx = torch.randint(0, num_noises, (1,), device=device)
        xt = model.finish_step(xt, x_0_hat, epst, t, noise[best_idx])

    set_seed(idx)
    try:
        img = model.decode_image(xt)
    except torch.OutOfMemoryError:
        img = model.decode_image(xt.to('cpu'))

    return img


def generate_ccfg(model: PipelineWrapper,
                  num_noises: int,
                  num_noises_to_optimize: int,
                  prompt: str = "",
                  negative_prompt: Optional[str] = None,
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    device = model.device
    dtype = model.dtype

    model.set_timesteps(model.num_timesteps, device=device)

    set_seed(88888888)
    prompt_embeds = model.encode_prompt(prompt, negative_prompt)

    kwargs = model.get_pre_kwargs(height=model.get_image_size(),
                                  width=model.get_image_size(),
                                  prompt_embeds=prompt_embeds)

    set_seed(100000)
    xt = torch.randn(1, *model.get_latent_shape(model.get_image_size()), device=device, dtype=dtype)

    result_noise_indices = []
    pbar = tqdm(model.timesteps)
    for idx, t in enumerate(pbar):
        set_seed(idx)
        noise = torch.randn(num_noises, *xt.shape[1:], device=device, dtype=dtype)  # Codebook

        _, epst_uncond, epst_cond = model.get_epst(xt, t, prompt_embeds, 1.0, return_everything=True, **kwargs)

        x_0_hat = model.get_x_0_hat(xt, epst_uncond, t)
        if t >= 1:
            prev_classif_score = epst_uncond - epst_cond
            set_seed(int(time.time_ns() & 0xFFFFFFFF))
            noise_indices = torch.randint(0, num_noises, size=(num_noises_to_optimize,), device=device)
            loss = torch.matmul(noise[noise_indices].view(num_noises_to_optimize, -1),
                                prev_classif_score.view(prev_classif_score.shape[0], -1).transpose(0, 1))
            best_idx = noise_indices[torch.argmax(loss)]
            best_noise = noise[best_idx]
            result_noise_indices.append(best_idx)
        else:
            best_noise = torch.zeros_like(noise[0])
        xt = model.finish_step(xt, x_0_hat, epst_uncond, t, best_noise)

    try:
        img = model.decode_image(xt)
    except torch.OutOfMemoryError:
        img = model.decode_image(xt.to('cpu'))
    return img, torch.stack(result_noise_indices).squeeze().cpu()

