"""Create a noisy test cube (clean `gt` + noisy `input`) for predict.py.

    python get_patches_wdc_test.py --config-file gauss_50_config.yaml
    python get_patches_wdc_test.py --noise mixture --out ../datasets/WDC/test/mixture.mat

The clean cube is read from --src_file (default: the 200 x 200 x 191 WDC test area
used in the paper), noise of the requested type is added with a fixed seed
(noise.py, Cases 1-5 of the paper) and the result is written as a single .mat file
with keys 'gt' and 'input' ([H, W, B], float32). predict.py pads and slides the
K-band window itself, so no per-band window files are needed.

With --config-file, the noise type (`test_noisy_type`, falling back to `noisy_type`),
`noise_level` and the output path (`test_file`) are taken from the config.
"""
import argparse
import os

import numpy as np
from scipy.io import loadmat, savemat

from noise import build_noise
from utils import load_config

parser = argparse.ArgumentParser(description='Generate a noisy test cube')
parser.add_argument('--config-file', help='take noise type, noise_level and output path from a config')
parser.add_argument('--src_file', default='../datasets/WDC/test/GT_crop.mat', help='clean test cube [H, W, B]')
parser.add_argument('--key', help='variable name of the clean cube (default: the only variable in the file)')
parser.add_argument('--noise', help="noise type: gauss, blind, noniid, stripe, deadline, impulse, mixture, complex")
parser.add_argument('--noise_level', type=float, help="sigma on the [0, 1] scale for 'gauss' (e.g. 50/255)")
parser.add_argument('--seed', type=int, default=0, help='random seed of the noise')
parser.add_argument('--out', help='output .mat file')


def read_clean_cube(path, key=None):
    mat = loadmat(path)
    variables = [k for k in mat if not k.startswith('__')]
    if key is None:
        if len(variables) != 1:
            raise ValueError(f'{path} holds {variables}; choose one with --key')
        key = variables[0]
    return np.float32(mat[key])


def main():
    args = parser.parse_args()
    config = load_config(args.config_file) if args.config_file else {}
    noise_type = args.noise or config.get('test_noisy_type', config.get('noisy_type'))
    noise_level = args.noise_level if args.noise_level is not None else config.get('noise_level')
    out = args.out or config.get('test_file') or f'../datasets/WDC/test/{noise_type}.mat'
    if noise_type is None:
        raise ValueError('give --noise or --config-file')
    if not out.endswith('.mat'):
        raise ValueError(f'output must be a .mat file, got {out}')

    clean = read_clean_cube(args.src_file, args.key)                     # [H, W, B]
    noise = build_noise(noise_type, noise_level)
    noisy = noise(clean.transpose(2, 0, 1), np.random.default_rng(args.seed)).transpose(1, 2, 0)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    savemat(out, {'gt': clean, 'input': np.float32(noisy)})
    print(f"{out}: {clean.shape[0]}x{clean.shape[1]}x{clean.shape[2]}, noise '{noise_type}'"
          + (f' (sigma {noise_level:.4f})' if noise_type == 'gauss' else '') + f', seed {args.seed}')


if __name__ == '__main__':
    main()
