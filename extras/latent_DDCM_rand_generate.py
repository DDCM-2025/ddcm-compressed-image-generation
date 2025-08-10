import os
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from ddcm.latent_runners import generate, set_seed
from ddcm.latent_models import load_model
from ddcm.util.img_utils import clear_color


"""
Run from the main directory as:
python -m extras.latent_DDCM_rand_generate --model_id ...
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--float16', action='store_true', help='Use float16 precision for model inference')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    parser.add_argument('--output_dir', type=str, default='./randgen_results')

    parser.add_argument('--model_id', type=str, required=True, help='Pre-trained diffusion model to use',
                        choices=['CompVis/stable-diffusion-v1-4',
                                 'stabilityai/stable-diffusion-2-1',
                                 'stabilityai/stable-diffusion-2-1-base',
                                 ])
    parser.add_argument('-T', '--timesteps', dest='T', type=int, default=1000, help='Compress using T diffusion steps.')
    parser.add_argument('-K', '--num_noises', dest='K', type=int, default=2, help="Codebook size")
    parser.add_argument('--prompt', type=str, required=True, help="Generation prompt.")
    parser.add_argument('--guidance_scale', type=float, default=6.0, help="Scale for the CFG of the diffusion model")

    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--negative_prompt', type=str, default=None)

    args = parser.parse_args()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    # Load model
    model, _ = load_model(args.model_id, args.T, device, float16=args.float16)

    out_path = os.path.join(args.output_dir)
    os.makedirs(out_path, exist_ok=True)

    random_seeds_for_ims = torch.randint(0, 2**16, (args.n_samples,), dtype=int)
    # Do Inference
    for i in tqdm(range(args.n_samples), desc="DB"):
        set_seed(random_seeds_for_ims[i].item())
        with torch.no_grad():
            gen_im = generate(model,
                              args.K,
                              device,
                              args.prompt,
                              negative_prompt=args.negative_prompt,
                              guidance_scale=args.guidance_scale,
                              )

        plt.imsave(os.path.join(out_path, f'{i}.png'), clear_color(gen_im, normalize=False))


if __name__ == '__main__':
    main()
