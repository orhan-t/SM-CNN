
from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
from joblib import Parallel, delayed
import multiprocessing
import argparse
from scipy.io import savemat, loadmat


parser = argparse.ArgumentParser(description='Generate patches from Full Resolution mat files')
parser.add_argument('--src_dir', default='../../../DataGen/matlab/WDC/blind', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../../datasets/WDC/test/blind', type=str, help='Directory for image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=1, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=8, type=int, help='Number of CPU Cores')
parser.add_argument('--K', default=24, type=int, help='Number of bands')


args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores
K=args.K

if os.path.exists(tar):
    os.removedirs(tar)
os.makedirs(tar)

#get sorted folders
files = natsorted(glob(os.path.join(src, '*.mat')))

print(files)

clean_file = files[0]
print(clean_file)
clean_img = loadmat(clean_file)
bands = clean_img['gt'].shape[-1]
clean = np.concatenate([clean_img['gt'][..., range(int(K / 2), 0, -1)],
                                        clean_img['gt'],
                                        clean_img['gt'][
                                        ..., range(bands - 1, bands - int(K/2),  -1)]], axis=-1)
noisy = np.concatenate([clean_img['input'][..., range(int(K / 2), 0, -1)],
                                        clean_img['input'],
                                        clean_img['input'][
                                        ..., range(bands - 1, bands - int(K/2),  -1)]], axis=-1)
def save_output(dir, clean, noisy, name):
    mdic = {"gt": clean, "input": noisy}
    savemat(dir + "/" + name + ".mat", mdic)

def save_files(i):
    H = clean_img['gt'].shape[0]
    W = clean_img['gt'].shape[1]
    for j in range(NUM_PATCHES):
        for k in range(0, bands, 1):
            clean_patch = clean[i * PS:i * PS + PS, j * PS:j * PS + PS, k:k + int(K)]
            noisy_patch = noisy[i * PS:i * PS + PS, j * PS:j * PS + PS, k:k + int(K)]

            save_output(tar, clean_patch, noisy_patch, 'patch_{}_{}_{}'.format(i + 1, j + 1, k + 1))

Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(NUM_PATCHES)))
