import argparse
import os
from functools import partial
import re

import torch
import numpy as np
import yaml
from PIL import Image
from torchvision.transforms.functional import resize

from ddcm.latent_runners import get_loader
from ddcm.guided_diffusion.gaussian_diffusion import create_sampler
from ddcm.guided_diffusion.measurements import get_noise, get_operator
from ddcm.guided_diffusion.unet import create_model
from ddcm.util.file import save_as_binary_bitwise, load_from_binary_bitwise, get_args_from_filename
from ddcm.util.img_utils import clear_color


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


diffusion_model_ckpt = "./ddcm/ckpt/imagenet_256.pth"
model_config_path = "./ddcm/configs/imagenet_model_config.yaml"
diffusion_config_path = "./ddcm/configs/diffusion_config.yaml"

if not os.path.isfile(diffusion_model_ckpt):
    os.makedirs(os.path.dirname(diffusion_model_ckpt), exist_ok=True)
    os.system(
        f"wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt -O {diffusion_model_ckpt}"
    )


def inference(args, device):
    if args.mode == 'decompress':
        args.T = int(re.search(r'T=(\d+)', args.input_path).group(1))
        args.K = int(re.search(r'K=(\d+)', args.input_path).group(1))

    # Load configs
    model_config = load_yaml(model_config_path)
    diffusion_config = load_yaml(diffusion_config_path)
    model_config["model_path"] = diffusion_model_ckpt

    # Load model and sampler
    model = create_model(**model_config).to(device).eval()
    diffusion_config['timestep_respacing'] = args.T
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop_linear_restoration,
                        model=model,
                        num_opt_noises=args.K,
                        eta=args.eta)

    if args.input_path.endswith(".bin"):
        loaded_indices = load_from_binary_bitwise(args.input_path, args.K, 1, 1, args.T)["noise_indices"].to(device)
        x_start = torch.randn(1, 3, 256, 256).to(device)

        with torch.no_grad():
            sample, indices = sample_fn(x_start=x_start, ref_img=None, y_n=None,
                                        linear_operator=None, loaded_indices=loaded_indices)
            os.makedirs(args.output_dir, exist_ok=True)
            Image.fromarray((clear_color(sample.squeeze(0)) * 255).round().astype(np.uint8)).save(
                args.output_dir + '/' + os.path.basename(args.input_path).split('.')[0] + '.png')

    else:
        task_config = load_yaml('./ddcm/configs/linear_inverse_problems/' + args.task_config)

        measure_config = task_config['measurement']
        operator = get_operator(device=device, **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])

        loader = get_loader(args.input_path, resize_to=256)

        for i, (ref_img, orig_path) in enumerate(loader):
            ref_img = ref_img.to(device)
            ref_img = resize(ref_img, (model_config['image_size'], model_config['image_size']))
            y = operator.forward(ref_img)
            y_n = noiser(y)
            fname = os.path.basename(orig_path[0]).split(os.extsep)[0]

            os.makedirs(args.output_dir, exist_ok=True)
            if args.save_degraded_img:
                Image.fromarray((clear_color(y_n.squeeze(0)) * 255).round().astype(np.uint8)).save(
                    os.path.join(args.output_dir, f'{fname}_degraded.png'))

            x_start = torch.randn(1, 3, 256, 256).to(device)

            with torch.no_grad():
                sample, indices = sample_fn(x_start=x_start, ref_img=ref_img, y_n=y_n,
                                            linear_operator=operator, loaded_indices=None)

            save_as_binary_bitwise(indices.numpy(), None, args.K, 1, 1,
                                   os.path.join(args.output_dir, f'{fname}_T={args.T}_K={args.K}.bin'))
            Image.fromarray((clear_color(sample.squeeze(0)) * 255).round().astype(np.uint8)).save(
                os.path.join(args.output_dir, f'{fname}_restored.png'))


def main():
    parser = argparse.ArgumentParser(description="Compressed Posterior Sampling CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    common_parser.add_argument('--eta', type=float, default=1.0)
    common_parser.add_argument('--input_path', type=str, required=True)
    common_parser.add_argument('--output_dir', type=str, required=True)

    restoration_parser = subparsers.add_parser("restore", parents=[common_parser])
    restoration_parser.add_argument("-K", "--num_noises", dest="K", type=int, default=4096, help="Codebook size.")
    restoration_parser.add_argument("-T", "--timesteps", dest="T", type=int, default=1000, help="Restore using T diffusion steps.")
    restoration_parser.add_argument('--task_config',
                                    choices=['colorization.yaml',
                                             'super_resolution.yaml'],
                                    type=str,
                                    required=True)
    restoration_parser.add_argument('--save_degraded_img', action='store_true')

    decompress_parser = subparsers.add_parser("decompress", help='Decompress a file', parents=[common_parser])

    args = parser.parse_args()
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    inference(args, device)

if __name__ == '__main__':
    main()
