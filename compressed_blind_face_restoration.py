import argparse
import os
from functools import partial

import yaml
import cv2
import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from ddcm.util.basicsr_img_util import img2tensor, tensor2img
from ddcm.util.swinir import create_swinir_model
from ddcm.util.file import save_as_binary_bitwise, load_from_binary_bitwise, get_args_from_filename
from ddcm.guided_diffusion.gaussian_diffusion import create_sampler
from ddcm.guided_diffusion.unet import create_model

# Re-use your existing helper functions: create_swinir_model, load_yaml

# Your existing model setup...

ffhq_diffusion_model = "./ddcm/ckpt/iddpm_ffhq512_ema500000.pth"
mmse_model_ckpt = "./ddcm/ckpt/swinir_restoration512_L1.pth"
model_config_path = "./ddcm/configs/ffhq512_model_config.yaml"
diffusion_config_path = "./ddcm/configs/diffusion_config.yaml"

if not os.path.isfile(ffhq_diffusion_model):
    os.makedirs(os.path.dirname(ffhq_diffusion_model), exist_ok=True)
    os.system(
        f"wget https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq512_ema500000.pth -O {ffhq_diffusion_model}"
    )
if not os.path.isfile(mmse_model_ckpt):
    os.makedirs(os.path.dirname(mmse_model_ckpt), exist_ok=True)
    os.system(
        f"wget https://github.com/zsyOAOA/DifFace/releases/download/V1.0/swinir_restoration512_L1.pth -O {mmse_model_ckpt}"
    )


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


@torch.no_grad()
def generate_reconstruction(degraded_face_img, K, T, iqa_metric, iqa_coef, loaded_indices):
    assert iqa_metric in ["niqe", "clipiqa+", "topiq_nr-face"]

    model_config = load_yaml(model_config_path)
    diffusion_config = load_yaml(diffusion_config_path)

    model_config["model_path"] = ffhq_diffusion_model
    models = {"main_model": create_model(**model_config), "mmse_model": create_swinir_model(mmse_model_ckpt)}

    diffusion_config["timestep_respacing"] = T
    sampler = create_sampler(**diffusion_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models["main_model"].to(device)
    mmse_model = models["mmse_model"].to(device)

    sample_fn = partial(
        sampler.p_sample_loop_blind_restoration,
        model=model,
        num_opt_noises=K,
        eta=1.0,
        iqa_metric=iqa_metric,
        iqa_coef=iqa_coef,
    )

    if degraded_face_img is not None:
        mmse_img = mmse_model(degraded_face_img).clip(0, 1) * 2 - 1
        x_start = torch.randn(mmse_img.shape, device=device)
    else:
        mmse_img = None
        x_start = torch.randn(1, 3, 512, 512, device=device)
    restored_face, indices = sample_fn(x_start=x_start, mmse_img=mmse_img, loaded_indices=loaded_indices)

    return restored_face, indices


def resize(img, size):
    # From https://github.com/sczhou/CodeFormer/blob/master/facelib/utils/face_restoration_helper.py
    h, w = img.shape[0:2]
    scale = size / min(h, w)
    h, w = int(h * scale), int(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


@torch.no_grad()
def enhance_faces(img, face_helper, has_aligned, K, T, iqa_metric, iqa_coef, loaded_indices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_helper.clean_all()
    if has_aligned:  # The inputs are already aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        face_helper.input_img = resize(face_helper.input_img, 640)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        face_helper.align_warp_face()
    if len(face_helper.cropped_faces) == 0:
        raise ValueError("Could not identify any face in the image.")
    if has_aligned and len(face_helper.cropped_faces) > 1:
        raise ValueError("You marked that the input image is aligned, but multiple faces were detected.")
    restored_faces = []
    generated_indices = []
    for i, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        cur_loaded_indices = loaded_indices[i] if loaded_indices is not None else None

        output, indices = generate_reconstruction(cropped_face_t, K, T, iqa_metric, iqa_coef, cur_loaded_indices)

        restored_face = tensor2img(output.to(torch.float32).squeeze(0), rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype("uint8")
        restored_faces.append(restored_face),
        generated_indices.append(indices)
    return restored_faces, generated_indices


@torch.no_grad()
def decompress_face(K, T, iqa_metric, iqa_coef, loaded_indices):
    assert loaded_indices is not None

    output, indices = generate_reconstruction(None, K, T, iqa_metric, iqa_coef, loaded_indices)

    restored_face = tensor2img(output.to(torch.float32).squeeze(0), rgb2bgr=True, min_max=(-1, 1)).astype("uint8")

    return restored_face, loaded_indices


@torch.no_grad()
def inference(
    input_path,
    output_path,
    T,
    K,
    iqa_metric,
    iqa_coef,
    aligned,
    device
):
    if input_path.endswith(".bin"):
        indices = load_from_binary_bitwise(input_path, K, 1, 1, T + 1)["noise_indices"].to(device)
    else:
        indices = None

    if indices is None:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        h, w = img.shape[0:2]
        if h > 4500 or w > 4500:
            raise ValueError("Image size too large.")

        face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
            model_rootpath=None,
        )

        x, indices = enhance_faces(
            img,
            face_helper,
            aligned,
            K=K,
            T=T,
            iqa_metric=iqa_metric,
            iqa_coef=iqa_coef,
            loaded_indices=indices,
        )
        for i, restored_img in enumerate(x):
            cv2.imwrite(output_path + f"_{i}.png", restored_img)

        for i, index in enumerate(indices):
            save_as_binary_bitwise(index.numpy(), None, K, 1, 1,
                                   output_path + f"_{i}_T={T}_K={K}_iqam={iqa_metric[0]}_iqac={iqa_coef}_aligned={int(aligned)}.bin")
    else:
        x, indices = decompress_face(
            K=K,
            T=T,
            iqa_metric=iqa_metric,
            iqa_coef=iqa_coef,
            loaded_indices=indices,
        )
        cv2.imwrite(f"{output_path}.png", x)

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    common_parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input image/bitstream (including the file format)."
    )
    common_parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path of the output image/bitstream to save (excluding the file format).",
    )

    restoration_parser = subparsers.add_parser('restore', help='Restore an image', parents=[common_parser])
    restoration_parser.add_argument("-K", "--num_noises", dest="K", type=int, default=4096, help="Codebook size.")
    restoration_parser.add_argument("-T", "--timesteps", dest="T", type=int, default=1000, help="Restore using T diffusion steps.")
    restoration_parser.add_argument("--aligned", action="store_true", help="Whether the input face image is aligned.")
    restoration_parser.add_argument("--iqa_metric", type=str, default="niqe", choices=["niqe", "clipiqa+", "topiq_nr-face"], help="Non-reference quality metric for the restoration loss.")
    restoration_parser.add_argument("--iqa_coef", type=float, default=1.0, help="Coefficient for the non-reference quality metric in the restoration loss.")

    decompress_parser = subparsers.add_parser('decompress', help='Decompress a file', parents=[common_parser])

    args = parser.parse_args()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    if args.mode == 'decompress':
        args.T, args.K, args.iqa_metric, args.iqa_coef, args.aligned = get_args_from_filename(args.input_path, iqa=True)

    inference(args.input_path, args.output_path, args.T, args.K, args.iqa_metric, args.iqa_coef, args.aligned, device)
