import re
import numpy as np
import torch


def get_args_from_filename(filename, cfg=False, iqa=False):
    T = int(re.search(r'T=(\d+)', filename).group(1))
    K = int(re.search(r'K=(\d+)', filename).group(1))

    if iqa:
        iqa_metric = re.search(r'iqam=(\w)', filename).group(1)
        iqa_coef = float(re.search(r'iqac=(\d+\.\d+)', filename).group(1))
        aligned = bool(re.search(r'aligned=(\d)', filename).group(1))
        if iqa_metric == 'n':
            iqa_metric = 'niqe'
        elif iqa_metric == 'c':
            iqa_metric = 'clipiqa+'
        elif iqa_metric == 't':
            iqa_metric = 'topiq_nr-face'
        else:
            raise ValueError(f"Unknown iqa_metric: {iqa_metric}")
        return T, K, iqa_metric, iqa_coef, aligned

    t_range = re.search(r'_in(\d+)-(\d+)', filename)
    t_range = (int(t_range.group(1)), int(t_range.group(2)))
    M = int(re.search(r'M=(\d+)', filename).group(1))
    C = int(re.search(r'C=(\d+)', filename).group(1))
    model_id = re.search(rf'model=(.+?)[\\/]', filename).group(1)
    if model_id == 'stable-diffusion-v1-4':
        model_id = 'CompVis/stable-diffusion-v1-4'
    elif model_id == 'stable-diffusion-v1-5':
        model_id = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    elif model_id.startswith('stable-diffusion-'):
        model_id = f'stabilityai/{model_id}'
    elif model_id == 'imagenet':
        pass
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    if cfg:
        cfg_scale = float(re.search(r'cfg=(\d+\.\d+)', filename).group(1))
        return T, K, M, C, t_range, model_id, cfg_scale

    return T, K, M, C, t_range, model_id


def save_as_binary_bitwise(noise_indices, coeff_indices, K, M, C, filename):

    bits_per_index = int(np.ceil(np.log2(K)))
    # bits_per_coeff = C

    if M == 1:
        # noise_indices = [[noise], [noise], [noise], ...]
        bitstring = ''.join(format(val, f'0{bits_per_index}b') for val in noise_indices)
    else:
        # noise_indices: [[noise1, noise2, ..., noiseM], [coeff1, coeff2, ..., coeffM-1], 
        #                   [noise1, noise2, ..., noiseM], [coeff1, coeff2, ..., coeffM-1], ...]
        # encode like this: noise1, coeff1, noise2, ..., coeffM-1, noiseM, noise1, coeff1, ...
        bitstring = ''
        for noise_indices_t, coeff_indices_t in zip(noise_indices, coeff_indices):
            bitstring += format(noise_indices_t[0], f'0{bits_per_index}b')
            for coeff, noise in zip(coeff_indices_t, noise_indices_t[1:]):
                bitstring += format(coeff, f'0{C}b')
                bitstring += format(noise, f'0{bits_per_index}b')

    # Convert bitstring to bytes **including padding**
    byte_array = int(bitstring, 2).to_bytes((len(bitstring) + 7) // 8, byteorder='big')

    # Write to binary file
    with open(filename, 'wb') as f:
        f.write(byte_array)


def load_from_binary_bitwise(filename, K, M, C, optimized_Ts):
    bits_per_index = int(np.ceil(np.log2(K)))
    # bits_per_coeff = C

    with open(filename, 'rb') as f:
        byte_data = f.read()
        
    bitstring = bin(int.from_bytes(byte_data, byteorder='big'))[2:]  # Remove '0b' prefix

    # Pad with leading zeros if needed (since we saved an int number)
    bits_amount = (optimized_Ts - 1) * (M * bits_per_index + (M - 1) * C)
    bitstring = bitstring.zfill(bits_amount)

    if M == 1:
        # noise_indices = [[noise], [noise], [noise], ...]
        noise_indices = [[int(bitstring[i:i + bits_per_index], 2)] for i in range(0, len(bitstring), bits_per_index)]
        coeff_indices = [[]] * len(noise_indices)
    else:
        # noise_indices: [[noise1, noise2, ..., noiseM], [coeff1, coeff2, ..., coeffM-1], 
        #                   [noise1, noise2, ..., noiseM], [coeff1, coeff2, ..., coeffM-1], ...]
        # encoded like this: noise1, coeff1, noise2, ..., coeffM-1, noiseM, noise1, coeff1, ...
        noise_indices = []
        coeff_indices = []
        for i in range(0, len(bitstring), (M * bits_per_index + (M - 1) * C)):
            time_segment = bitstring[i:i + (M * bits_per_index + (M - 1) * C)]
            t_noise_indices = []
            t_coeff_indices = []
            for j in range(0, len(time_segment), bits_per_index + C):
                noise = int(time_segment[j:j + bits_per_index], 2)
                t_noise_indices.append(noise)
                left_seg = time_segment[j + bits_per_index:j + bits_per_index + C]
                if len(left_seg):
                    coeff = int(left_seg, 2)
                    t_coeff_indices.append(coeff)
            noise_indices.append(t_noise_indices)
            coeff_indices.append(t_coeff_indices)

    return {'noise_indices': torch.from_numpy(np.array(noise_indices, dtype=np.int32)),
            'coeff_indices': torch.from_numpy(np.array(coeff_indices, dtype=np.int32))}
