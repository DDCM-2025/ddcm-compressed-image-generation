import os
import argparse
from pathlib import Path
from glob import glob

import torch
import matplotlib.pyplot as plt
import numpy as np

from ddcm.latent_runners import get_loader, compress
from ddcm.latent_models import load_model
from ddcm.util.img_utils import clear_color
from ddcm.util.file import save_as_binary_bitwise, load_from_binary_bitwise, get_args_from_filename


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    common_parser.add_argument('--float16', action='store_true', help='Use float16 precision for model inference')
    common_parser.add_argument('--input_dir', type=str, required=True, help="Directory containing images or bin files to edit.")
    common_parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the results')

    compression_args_parser = argparse.ArgumentParser(add_help=False)
    compression_args_parser.add_argument('--model_id', type=str, required=True, help='Pre-trained diffusion model to use', 
                                         choices=['CompVis/stable-diffusion-v1-4',
                                                  'stabilityai/stable-diffusion-2-1-base',
                                                  'stabilityai/stable-diffusion-2-1'
                                                  ])
    compression_args_parser.add_argument('--timesteps', dest='T', type=int, default=1000, help='Compress using T steps')
    compression_args_parser.add_argument('--num_noises', dest='K', type=int, default=8192, help="Codebook size")
    compression_args_parser.add_argument('--num_pursuit_noises', dest='M', type=int, default=6, help="Atoms in the MP version. MP starts when M > 1.")
    compression_args_parser.add_argument('--num_pursuit_coef_bits', dest='C', type=int, default=1, help="Amount of discrete coefficients for MP.")
    compression_args_parser.add_argument('--t_range', nargs=2, type=int, default=[999, 0], help="Optimize only a subset of the timesteps range.")
    compression_args_parser.add_argument('--guidance_scale', type=float, default=6.0, help="Scale for the CFG of the diffusion model")

    edit_args_parser = argparse.ArgumentParser(add_help=False)
    edit_args_parser.add_argument('--src_prompt', type=str, required=True, help="Source prompt describing the original image.")
    edit_args_parser.add_argument('--dst_prompts', type=str, required=True, nargs='+', help="A target prompt describing the required edit.")
    edit_args_parser.add_argument('--tskips', type=int, default=[0], nargs='+', help="The timestep from which to change the prompt and apply the edit.")

    compress_parser = subparsers.add_parser('edit', help='Edit a file', parents=[common_parser, compression_args_parser, edit_args_parser])
    edit_parser = subparsers.add_parser('reedit', help='Edit a previously-compressed bin file', parents=[common_parser, edit_args_parser])

    args = parser.parse_args()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f'Using {device}')

    if args.mode == 'reedit':
        # get the first indices file in the input directory
        indices_file = glob(os.path.join(args.input_dir, '**', '*.bin'), recursive=True)[0]
        args.T, args.K, args.M, args.C, args.t_range, args.model_id, args.guidance_scale = get_args_from_filename(indices_file, cfg=True)

    # Load model
    model, _ = load_model(args.model_id, args.T, device, args.float16)
    imsize = model.get_image_size()

    optimized_Ts = len([t for t in model.timesteps if (args.t_range[0] >= t >= args.t_range[1])])
    out_prefix = f"T={args.T}_in{args.t_range[0]}-{args.t_range[1]}_K={args.K}_M={args.M}_C={args.C}_cfg={args.guidance_scale}_model={args.model_id.split('/')[1]}"
    bpp = (optimized_Ts - 1) * (args.M * np.log2(args.K) + (args.M - 1) * args.C) / (imsize ** 2)
    print(f'The BPP will be: {bpp:.4f}')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'edit':
        iterator = get_loader(args.input_dir, resize_to=imsize)
    elif args.mode == 'reedit':
        iterator = glob(os.path.join(args.input_dir, '**', '*.bin'), recursive=True)

    for inp in iterator:
        if args.mode == 'edit':
            orig_img, orig_path = inp
            respath = os.path.join(args.output_dir, out_prefix, os.path.dirname(orig_path[0]))
            os.makedirs(respath, exist_ok=True)
            fname = os.path.basename(orig_path[0]).split(os.extsep)[0]

            orig_img = orig_img.to(device)
            with torch.no_grad():
                compressed_im, noise_indices, coeff_indices = compress(
                    model, orig_img, args.K, device, args.M, args.C, args.t_range,
                    edit_src_prompt=args.src_prompt, guidance_scale=args.guidance_scale)

            plt.imsave(os.path.join(respath, f'{fname}_comp.png'), clear_color(compressed_im, normalize=False))
            indices_file = os.path.join(respath, f'{fname}_noise_indices.bin')
            save_as_binary_bitwise(noise_indices.numpy(), coeff_indices.numpy(),
                                   args.K, args.M, args.C, indices_file)
            with open(os.path.join(respath, f'{fname}_srcprompt.txt'), 'w') as f:
                f.writelines([args.src_prompt])

        elif args.mode == 'reedit':
            indices_file = inp
            respath = os.path.join(args.output_dir, Path(os.path.dirname(indices_file)).relative_to(args.input_dir))
            os.makedirs(respath, exist_ok=True)
            fname = os.path.basename(indices_file).split(os.extsep)[0].split('_noise_indices')[0]

        # Now to edit, we load from file the indices.
        indices = load_from_binary_bitwise(indices_file, args.K, args.M, args.C, optimized_Ts)
        for dst_prompt in args.dst_prompts:
            for tskip in args.tskips:
                with torch.no_grad():
                    edited_im, _, _ = compress(model, None, args.K, device, args.M, args.C, args.t_range, indices,
                                               edit_src_prompt=args.src_prompt, edit_dst_prompt=dst_prompt,
                                               guidance_scale=args.guidance_scale, t_edit=tskip)
                    plt.imsave(os.path.join(respath, f'{fname}_edit_{dst_prompt.replace(" ", "_")}_skip{tskip}.png'), clear_color(edited_im, normalize=False))


if __name__ == '__main__':
    main()
