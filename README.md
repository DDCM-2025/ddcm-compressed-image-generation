[![Python 3.8.10](https://img.shields.io/badge/python-3.8.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![NumPy](https://img.shields.io/badge/numpy-1.26.4-green?logo=numpy&logoColor=white)](https://pypi.org/project/numpy/1.23.5/)
[![torch](https://img.shields.io/badge/torch-2.5.0-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.20.1+-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![diffusers](https://img.shields.io/badge/diffusers-0.31.0-green)](https://github.com/huggingface/diffusers/)
[![transformers](https://img.shields.io/badge/transformers-4.37.2-green)](https://github.com/huggingface/transformers/)

<!-- omit in toc -->
# Compressed Image Generation with Denoising Diffusion Codebook Models [ICML 2025]

<!-- omit in toc -->
### [Project page](https://ddcm-2025.github.io) | [Arxiv](https://arxiv.org/abs/2502.01189) | [Demo](https://huggingface.co/spaces/DDCM/DDCM-Compressed-Image-Generation)

![DDCM results overview](assets/ddcm.png)

<!-- omit in toc -->
## Table of Contents

- [Requirements](#requirements)
- [Change Log](#change-log)
- [Usage Examples](#usage-examples)
  - [Compression](#compression)
  - [Compressed Posterior Sampling](#compressed-posterior-sampling)
  - [Compressed Blind Face Image Restoration](#compressed-blind-face-image-restoration)
  - [Additional Applications](#additional-applications)
    - [Compressed Classifier Free Guidance](#compressed-classifier-free-guidance)
    - [Compressed Text-Based Image Editing](#compressed-text-based-image-editing)
  - [Extras](#extras)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Requirements

Install dependencies using:

```bash
python -m pip install -r requirements.txt
```

## Change Log

- **10.08.25**: Code release for compressed posterior sampling, compressed blind face image restoration & compressed text-based image editing, along with the extra experiments shown in the paper.
- **27.07.25**: Initial release with code for latent compression.

## Usage Examples

### Compression

Our code supports compressing images of size $256^{2}$, $512^{2}$ and $768^{2}$. For $512^{2}$ and $768^{2}$ we use latent diffusion models; $256^{2}$ uses pixel-space diffusion (coming soon).

Run compression / decompression / roundtrip:

```bash
python latent_compression.py compress|decompress|roundtrip [OPTIONS]
```

You should specify the following arguments:

- `--model_id`: HuggingFace model ID. Choose between `stabilityai/stable-diffusion-2-1` (for images of size $768^2$) and `stabilityai/stable-diffusion-2-1-base` or `CompVis/stable-diffusion-v1-4` (for images of size $512^2$).
- `--timesteps`: Number of denoising steps ($T$ in our paper).
- `--num_noises`: Size of each codebook ($K$ in our paper).
- `--input_dir`: Input directory (images to compress or binary files to decompress).

See the `--help` flag for more options and details.
The generated binary file name includes the compression metadata (e.g., $T$, $K$), which are automatically parsed at decompression.

Compression example:

```bash
python latent_compression.py compress \
--float16 \
--input_dir ./test_imgs \
--output_dir ./outputs \
--model_id "stabilityai/stable-diffusion-2-1-base" \
--num_noises 256 \
--timesteps 1000
```

Decompression example:

```bash
python latent_compression.py decompress \
--float16 \
--input_dir ./compressed_binary_files/ \
--output_dir ./output_imgs_decompressed
```

### Compressed Posterior Sampling

This module can compress & perform posterior sampling for linear inverse problems (simultaneously), such as image super-resolution, colorization, and Gaussian blur. Internally, we use a pre-trained ImageNet diffusion model (see [guided-diffusion](https://github.com/openai/guided-diffusion)).

Run restoration & compression / decompression:

```bash
python compressed_posterior_sampling.py restore|decompress [OPTIONS]
```

You should specify the following arguments:

- `--input_path`: Path to image or binary `.bin` file.
- `--output_dir`: Directory to save the output images or binary files.
- `--task_config`: Configuration file for the task (e.g., `colorization.yaml`, `gaussian_blur.yaml`, `super_resolution.yaml`).
- `--timesteps`: Number of denoising steps ($T$ in our paper).
- `--num_noises`: Size of each codebook ($K$ in our paper).
- `--eta`: Denoising parameter ($\eta$ in our paper).

See the `--help` flag for more options and details.
The generated binary file name includes the compression metadata (e.g., $T$, $K$), which are automatically parsed at decompression.

Restoration & compression example for super-resolution:

```bash
python compressed_posterior_sampling.py restore \
--task_config super_resolution.yaml \
--input_path ./test_imgs \
--output_dir ./outputs \
--timesteps 1000 \
--num_noises 256 \
--eta 1.0
```

Decompression example:

```bash
python compressed_posterior_sampling.py decompress \
--eta 1.0 \
--input_path "./compressed_binary_files/image_T=1000_K=256.bin" \
--output_dir ./output_imgs_decompressed
```

### Compressed Blind Face Image Restoration

This module can restore & compress (simultaneously) real-world degraded face images. It supports both aligned (recommended) and unaligned face images of size $512^2$. Internally, we use a FFHQ-trained diffusion model (see [DifFace](https://github.com/zsyOAOA/DifFace)) and a MSE SwinIR restoration model (see [SwinIR](https://github.com/JingyunLiang/SwinIR)).

Run restoration & compression / decompression:
Command:

```bash
python compressed_blind_face_restoration.py restore|decompress [OPTIONS]
```

You should specify the following arguments:

- `--input_path`: Path to image or binary `.bin` file.
- `--output_path`: Path prefix for saving the output.
- `--num_noises`: Size of each codebook ($K$ in our paper).
- `--timesteps`: Number of denoising steps ($T$ in our paper).
- `--iqa_metric`: IQA metric to optimize (`niqe`, `clipiqa+`, `topiq_nr-face`)
- `--aligned`: Flag if input face is aligned

See the `--help` flag for more options and details.
The generated binary file name includes the compression metadata (e.g., $T$, $K$), which are automatically parsed at decompression.

Restoration & compression example:

```bash
python compressed_blind_face_restoration.py restore \
--input_path ./aligned_degraded_face_img.jpg \
--output_path ./aligned_degraded_face_img_restored \
--aligned \
--num_noises 4096 \
--timesteps 1000 \
--iqa_metric "niqe"
```

Decompression example:

```bash
python compressed_blind_face_restoration.py restore_and_compress \
--input_path ./aligned_degraded_face_img_restored.bin \
--output_path ./aligned_degraded_face_img_restored
```

### Additional Applications

#### Compressed Classifier Free Guidance

Coming soon.

#### Compressed Text-Based Image Editing

This module enables text-guided image editing, where the resulting image is automatically compressed.

Run editing & compression / re-editing from a previously compressed image:

```bash
python compressed_textbased_editing.py edit|reedit [OPTIONS]
```

You should specify the following arguments:

- `--input_dir`: Input directory containing images to edit or binary files to re-edit.
- `--output_dir`: Output directory for saving edited images or binary files.
- `--model_id`: HuggingFace model ID for the diffusion model (e.g., `stabilityai/stable-diffusion-2-1-base`).
- `--num_noises`: Size of each codebook ($K$ in our paper).
- `--timesteps`: Number of denoising steps ($T$ in our paper).
- `--num_pursuit_noises`: Number of matching-pursuit noises ($M$ in our paper).
- `--num_pursuit_coef_bits`: Number of bits for matching-pursuit coefficients ($C$ in our paper).
- `--guidance_scale`: Guidance scale for classifier-free guidance.
- `--src_prompt`: Source prompt describing the original input image.
- `--dst_prompts`: Target prompts describing the wanted edited images.
- `--tskips`: How many timesteps-selection steps to skip (e.g., `200 500 700`). Smaller number yields stronger edits. Note that this number is dependent on how many timesteps are used.

See the `--help` flag for more options and details.
The generated binary file name includes the compression metadata (e.g., $T$, $K$), which are automatically parsed at decompression.

Editing & compressing example:

```bash
python compressed_textbased_editing.py edit \
--float16 \
--input_dir ./images_to_edit \
--output_dir ./outputs \
--model_id "stabilityai/stable-diffusion-2-1-base" \
--num_noises 8192 \
--num_pursuit_noises 6 \
--num_pursuit_coef_bits 1 \
--timesteps 1000 \
--guidance_scale 6.0 \
--src_prompt "a photo of a cat" \
--dst_prompts "a photo of a dog" "a photo of a lion" \
--tskips 500 650 800
```

Re-editing & compressing example:

```bash
python compressed_textbased_editing.py reedit \
--float16 \
--input_dir ./compressed_binary_files \
--output_dir ./outputs \
--src_prompt "a photo of a cat" \
--dst_prompts "a photo of a tiger" \
--tskips 500 650 800
```

### Extras

We provide the code for additional experiements in the paper in the `extras` folder.

## Citation

If you use this code for your research, please cite our paper:

```
@inproceedings{
    ohayon2025compressed,
    title     = {Compressed Image Generation with Denoising Diffusion Codebook Models},
    author    = {Guy Ohayon and Hila Manor and Tomer Michaeli and Michael Elad},
    booktitle = {Forty-second International Conference on Machine Learning},
    year      = {2025},
    url       = {https://openreview.net/forum?id=cQHwUckohW}
}
```

## Acknowledgements

This project is released under the [MIT license](https://github.com/DDCM-2025/ddcm-compressed-image-generation/blob/main/LICENSE).

We borrowed codes from [huggingface](https://github.com/huggingface), [guided diffusion](https://github.com/openai/guided-diffusion), [DPS](https://github.com/DPS2022/diffusion-posterior-sampling), [SwinIR](https://github.com/JingyunLiang/SwinIR), [BasicSR](https://github.com/XPixelGroup/BasicSR), and [DifFace](https://github.com/zsyOAOA/DifFace). We thank the authors of these repositories for their useful implementations.
