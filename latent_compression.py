import os
import argparse
from pathlib import Path 

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import numpy as np

from ddcm.latent_runners import get_loader, compress
from ddcm.latent_models import load_model
from ddcm.util.img_utils import clear_color
from ddcm.util.file import get_args_from_filename, save_as_binary_bitwise, load_from_binary_bitwise


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    common_parser.add_argument('--float16', action='store_true', help='Use float16 precision for model inference')
    common_parser.add_argument('--input_dir', required=True, help="Directory containing images to compress or bin files to decmopress.")
    common_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the results')

    compression_args_parser = argparse.ArgumentParser(add_help=False)
    compression_args_parser.add_argument('--model_id', type=str, required=True, help='Pre-trained diffusion model to use',
                                         choices=['CompVis/stable-diffusion-v1-4',
                                                  'stabilityai/stable-diffusion-2-1',
                                                  'stabilityai/stable-diffusion-2-1-base'
                                                  ])
    compression_args_parser.add_argument('--timesteps', dest='T', type=int, default=1000, help='Compress using T steps')
    compression_args_parser.add_argument('--num_noises', dest='K', type=int, default=1024, help="Codebook size")
    compression_args_parser.add_argument('--num_pursuit_noises', dest='M', type=int, default=1, help="Atoms in the MP version. MP starts when M > 1.")
    compression_args_parser.add_argument('--num_pursuit_coef_bits', dest='C', type=int, default=1, help="Amount of discrete coefficients for MP.")
    compression_args_parser.add_argument('--t_range', nargs=2, type=int, default=[999, 0], help="Optimize only a subset of the timesteps range.")

    compress_parser = subparsers.add_parser('compress', help='Compress a file', parents=[common_parser, compression_args_parser])
    decompress_parser = subparsers.add_parser('decompress', help='Decompress a file', parents=[common_parser])
    roundtrip_parser = subparsers.add_parser('roundtrip', help='Compress and then decompress', parents=[common_parser, compression_args_parser])

    args = parser.parse_args()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    if args.mode == 'decompress':
        # get the first indices file in the input directory
        indices_file = glob(os.path.join(args.input_dir, '**', '*.bin'), recursive=True)[0]
        args.T, args.K, args.M, args.C, args.t_range, args.model_id = get_args_from_filename(indices_file)

    # Load model
    model, _ = load_model(args.model_id, args.T, device, args.float16, compile=False)
    imsize = model.get_image_size()

    optimized_Ts = len([t for t in model.timesteps if (args.t_range[0] >= t >= args.t_range[1])])
    if args.mode == 'compress' or args.mode == 'roundtrip':
        out_prefix = f"T={args.T}_in{args.t_range[0]}-{args.t_range[1]}_K={args.K}_M={args.M}_C={args.C}_model={args.model_id.split('/')[1]}"
        bpp = (optimized_Ts - 1) * (args.M * np.log2(args.K) + (args.M - 1) * args.C) / (imsize ** 2)
        print(f'The BPP will be: {bpp:.4f}')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'compress' or args.mode == 'roundtrip':
        loader = get_loader(args.input_dir, resize_to=imsize)

        # Do Inference
        for orig_img, orig_path in loader:
            respath = os.path.join(args.output_dir, out_prefix, os.path.dirname(orig_path[0]))
            os.makedirs(respath, exist_ok=True)  # keep dir structure
            fname = os.path.basename(orig_path[0]).split(os.extsep)[0]

            orig_img = orig_img.to(device)
            with torch.no_grad():
                compressed_im, noise_indices, coeff_indices = compress(model, orig_img, args.K, device, args.M, args.C, args.t_range)
            plt.imsave(os.path.join(respath, f'{fname}_comp.png'), clear_color(compressed_im, normalize=False))
            save_as_binary_bitwise(noise_indices.numpy(), coeff_indices.numpy(),
                                   args.K, args.M, args.C,
                                   os.path.join(respath, f'{fname}_noise_indices.bin'))

    if args.mode == 'roundtrip':
        args.input_dir = os.path.join(args.output_dir, out_prefix)
        args.output_dir = args.input_dir

    if args.mode == 'decompress' or args.mode == 'roundtrip':
        indices_files = glob(os.path.join(args.input_dir, '**', '*.bin'), recursive=True)
        for indices_file in tqdm(indices_files, desc="Decompressing files"):
            indices = load_from_binary_bitwise(indices_file, args.K, args.M, args.C, optimized_Ts)
            respath = os.path.join(args.output_dir, Path(os.path.dirname(indices_file)).relative_to(args.input_dir))
            os.makedirs(respath, exist_ok=True)
            fname = os.path.basename(indices_file).split(os.extsep)[0].split('_noise_indices')[0]

            # Decompress
            with torch.no_grad():
                decompressed_im, _, _ = compress(model, None, args.K, device, args.M, args.C, args.t_range, indices)
            plt.imsave(os.path.join(respath, f'{fname}_decomp.png'), clear_color(decompressed_im, normalize=False))


if __name__ == '__main__':
    main()
