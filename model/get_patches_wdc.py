"""Generate clean K-band training / validation windows from full-resolution .mat cubes.

    python get_patches_wdc.py                                     # -> ../datasets/WDC/train
    python get_patches_wdc.py --tar_dir ../datasets/WDC/valid --num_patches 128 --seed 1

Every source cube [H, W, B] (key --key) is mirror-padded along the spectral axis
(dataset_utils.pad_spectral) and random ps x ps x K windows are saved as
patch_<file>_<n>.mat with key 'gt'. Window k holds band k at index K // 2.
Noise is added on the fly during training (noise.py).
"""
import argparse
import os
import shutil
from glob import glob

import numpy as np
from joblib import Parallel, delayed
from natsort import natsorted
from scipy.io import loadmat, savemat
from tqdm import tqdm

from dataset_utils import pad_spectral

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution mat files')
parser.add_argument('--src_dir', default='../datasets/WDC/', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../datasets/WDC/train', type=str, help='Directory for image patches')
parser.add_argument('--key', default='data', type=str, help='Variable name of the cube in the source .mat files')
parser.add_argument('--ps', default=64, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=1024, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=8, type=int, help='Number of CPU Cores')
parser.add_argument('--K', default=24, type=int, help='Number of adjacent bands')
parser.add_argument('--seed', default=0, type=int, help='Random seed')


def save_files(i, file_name, args):
    rng = np.random.default_rng((args.seed, i))
    cube = np.float32(loadmat(file_name)[args.key])
    H, W, _ = cube.shape
    if H < args.ps or W < args.ps:
        raise ValueError(f'{file_name} is {H}x{W}, smaller than the patch size {args.ps}')
    bands = cube.shape[-1]
    padded = pad_spectral(cube, args.K, axis=-1)
    for j in range(args.num_patches):
        rr = rng.integers(0, H - args.ps + 1)
        cc = rng.integers(0, W - args.ps + 1)
        kk = rng.integers(0, bands)
        patch = padded[rr:rr + args.ps, cc:cc + args.ps, kk:kk + args.K]
        savemat(os.path.join(args.tar_dir, f'patch_{i + 1}_{j + 1}.mat'), {'gt': patch})


def main():
    args = parser.parse_args()
    files = natsorted(glob(os.path.join(args.src_dir, '*.mat')))
    if not files:
        raise FileNotFoundError(f'no .mat files in {args.src_dir}')
    print(files)

    if os.path.exists(args.tar_dir):
        shutil.rmtree(args.tar_dir)
    os.makedirs(args.tar_dir)

    Parallel(n_jobs=args.num_cores)(delayed(save_files)(i, f, args) for i, f in enumerate(tqdm(files)))


if __name__ == '__main__':
    main()
