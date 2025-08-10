import os
import argparse
import torch
import matplotlib.pyplot as plt

from ddcm.latent_runners import get_loader, set_seed
from ddcm.latent_models import load_model
from ddcm.util.img_utils import clear_color


"""
Run from the main directory as:
python -m extras.latent_vae_bound --model_id ...
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--float16', action='store_true', help='Use float16 precision for model inference')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    parser.add_argument('--model_id', type=str, required=True, help='Pre-trained diffusion model to use',
                        choices=['stabilityai/stable-diffusion-2-1',
                                 'stabilityai/stable-diffusion-2-1-base',
                                 ])
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results/vae_bound')

    args = parser.parse_args()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    # Load model
    model, _ = load_model(args.model_id, 1000, device, float16=args.float16)
    imsize = model.get_image_size()

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataloader
    loader = get_loader(args.input_dir, resize_to=imsize)
    for orig_img, orig_path in loader:
        respath = os.path.join(args.output_dir, os.path.dirname(orig_path[0]))
        os.makedirs(respath, exist_ok=True)  # keep dir structure
        fname = os.path.basename(orig_path[0]).split(os.extsep)[0]

        orig_img = orig_img.to(device)
        with torch.no_grad():
            set_seed(88888888)
            enc_im = model.encode_image(orig_img)
            set_seed(0)
            dec_img = model.decode_image(enc_im)

        plt.imsave(os.path.join(respath, f'{fname}.png'), clear_color(dec_img, normalize=False))


if __name__ == '__main__':
    main()
