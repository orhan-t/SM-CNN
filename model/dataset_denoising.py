import os
import re
from glob import glob

import numpy as np
import torch
from natsort import natsorted
from scipy.io import loadmat
from torch.utils.data import Dataset

from dataset_utils import Augment_RGB_torch
from model import center_band_index

augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


def _list_mat_files(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'data directory not found: {data_dir}')
    files = natsorted(glob(os.path.join(data_dir, '*.mat')))
    if not files:
        raise FileNotFoundError(f'no .mat files in {data_dir}')
    return files


##################################################################################################
class DataLoaderTrain(Dataset):
    """K-band training windows (``gt`` key, [H, W, K]) written by get_patches_wdc.py.

    Returns ``(clean_band [1, ps, ps], noisy_volume [1, K, ps, ps])`` where the clean
    target is the center band (index K // 2) and the noise is generated per sample.

    deterministic=False: random crop, random rot/flip and fresh noise on every call
      (the RNG is seeded from torch, which DataLoader seeds per worker, so workers
      never repeat each other and runs are reproducible from torch.manual_seed).
    deterministic=True: crop, augmentation and noise depend only on (seed, index),
      so the validation loss is comparable between epochs.
    """

    def __init__(self, data_dir, img_options, noise=None, augment=True, deterministic=False, seed=0):
        super().__init__()
        self.files_name = _list_mat_files(data_dir)
        self.patch_size = img_options['patch_size']
        self.K = img_options['K']
        self.center = center_band_index(self.K)
        self.noise = noise
        self.augment = augment
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, index):
        if self.deterministic:
            rng = np.random.default_rng((self.seed, index))
        else:
            rng = np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,))))

        cube = np.float32(loadmat(self.files_name[index])['gt']).transpose(2, 0, 1)  # [K, H, W]
        K, H, W = cube.shape
        if K != self.K:
            raise ValueError(f'{self.files_name[index]} has {K} bands, config K = {self.K}')
        ps = self.patch_size
        if H < ps or W < ps:
            raise ValueError(f'{self.files_name[index]} is {H}x{W}, smaller than patch_size {ps}')
        r = rng.integers(0, H - ps + 1)
        c = rng.integers(0, W - ps + 1)
        clean_volume = np.ascontiguousarray(cube[:, r:r + ps, c:c + ps])

        noisy_volume = self.noise(clean_volume, rng) if self.noise is not None else clean_volume
        clean_volume = torch.from_numpy(clean_volume)
        noisy_volume = torch.from_numpy(np.ascontiguousarray(noisy_volume, dtype=np.float32))

        if self.augment:
            apply_trans = transforms_aug[rng.integers(len(transforms_aug))]
            clean_volume = getattr(augment, apply_trans)(clean_volume)
            noisy_volume = getattr(augment, apply_trans)(noisy_volume)

        clean_band = clean_volume[self.center:self.center + 1].contiguous()  # [1, ps, ps]
        return clean_band, noisy_volume.unsqueeze(0).contiguous()            # [1, K, ps, ps]


##################################################################################################
_WINDOW_NAME = re.compile(r'patch_(\d+)_(\d+)_(\d+)\.mat$')


def load_test_cube(path, key_clean='gt', key_noisy='input'):
    """Load a test HSI as band-first float32 arrays ``(clean [B,H,W], noisy [B,H,W] or None)``.

    `path` is either
      * a .mat file holding the full cube(s) [H, W, B] (written by get_patches_wdc_test.py), or
      * a directory of legacy per-band windows ``patch_{i}_{j}_{k}.mat`` (keys gt / input,
        [ps, ps, K]); the cube is rebuilt from the center band of every window.
    `noisy` is None when the .mat file has no `key_noisy` entry.
    """
    if os.path.isfile(path):
        mat = loadmat(path)
        clean = np.float32(mat[key_clean]).transpose(2, 0, 1)
        noisy = np.float32(mat[key_noisy]).transpose(2, 0, 1) if key_noisy in mat else None
        return clean, noisy

    files = _list_mat_files(path)
    entries = []
    for f in files:
        m = _WINDOW_NAME.search(os.path.basename(f))
        if m is None:
            raise ValueError(f'unexpected file name {f}; expected patch_<i>_<j>_<band>.mat')
        entries.append((tuple(int(v) - 1 for v in m.groups()), f))
    first = loadmat(entries[0][1])
    ps_h, ps_w, K = first[key_clean].shape
    c = center_band_index(K)
    n_i = 1 + max(e[0][0] for e in entries)
    n_j = 1 + max(e[0][1] for e in entries)
    n_b = 1 + max(e[0][2] for e in entries)
    if len(entries) != n_i * n_j * n_b:
        raise ValueError(f'{path}: expected {n_i * n_j * n_b} window files, found {len(entries)}')
    clean = np.zeros((n_b, n_i * ps_h, n_j * ps_w), dtype=np.float32)
    noisy = np.zeros_like(clean)
    for (i, j, b), f in entries:
        mat = first if f == entries[0][1] else loadmat(f)
        clean[b, i * ps_h:(i + 1) * ps_h, j * ps_w:(j + 1) * ps_w] = mat[key_clean][:, :, c]
        noisy[b, i * ps_h:(i + 1) * ps_h, j * ps_w:(j + 1) * ps_w] = mat[key_noisy][:, :, c]
    return clean, noisy


##################################################################################################

def get_training_data(data_dir, img_options, noise=None, augment=True, deterministic=False, seed=0):
    return DataLoaderTrain(data_dir, img_options, noise=noise, augment=augment,
                           deterministic=deterministic, seed=seed)
