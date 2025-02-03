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
- [Usage Example](#usage-example)
  - [Compression](#compression)
  - [Compressed Posterior Sampling](#compressed-posterior-sampling)
  - [Compressed Blind Face Image Restoration](#compressed-blind-face-image-restoration)
  - [Additional Applications](#additional-applications)
    - [Compressed Classifier Free Guidance](#compressed-classifier-free-guidance)
    - [Editing](#editing)
  - [Extras](#extras)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Requirements

```bash
python -m pip install -r requirements.txt
```

## Change Log

- **27.07.25**: Initial release with code for latent compression.

## Usage Example

### Compression

Our code supports compressing images of size $256^{2}$, $512^{2}$ and $768^{2}$.

To compress images of size $512^{2}$ or $768^{2}$, we are using latent diffusion models. In this case, you should use

```bash
python latent_compression.py compress|decompress|roundtrip
```

You should specify the following arguments:

- `model_id`: The pre-trained latent diffusion model to use (pulled from hugging face). You can choose between `stabilityai/stable-diffusion-2-1` (for images of size $768^2$), `stabilityai/stable-diffusion-2-1-base` (for images of size $512^2$), and  `CompVis/stable-diffusion-v1-4` (for images of size $512^2$).
- `timesteps`: The number of sampling steps, e.g. `1000` (this is $T$ in our paper).
- `num_noises`: The size of the codebook (this is $K$ in our paper).
- `input_dir`: path to a directory that contains the images you wish to compress, or the binary files you wish to decompress.

See the `--help` flag for more options and details.

Here is a full example you could use to compress an image:

```bash
python latent_compression.py compress \
--gpu 0 \
--float16 \
--input_dir ./assets/ \
--output_dir ./compressed_imgs/ \
--model_id "stabilityai/stable-diffusion-2-1-base" \
--num_noises 256 \
--timesteps 1000
```

To decompress a saved binary file:

```bash
python latent_compression.py decompress \
--gpu 0 \
--float16 \
--input_dir ./compressed_binary_files/ \
--output_dir ./compressed_imgs/
```

Compressing images of size $256^2$ uses a pixel-space model, coming soon.

### Compressed Posterior Sampling

Coming soon.

### Compressed Blind Face Image Restoration

Coming soon.

### Additional Applications

#### Compressed Classifier Free Guidance

Coming soon.

#### Editing

Coming soon.

### Extras

We provide the code for additional experiements in the paper in the `extras` folder (coming soon).

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

We borrowed codes from [guided diffusion](https://github.com/openai/guided-diffusion) and [DPS](https://github.com/DPS2022/diffusion-posterior-sampling). We thank the authors of these repositories for their useful implementations.
