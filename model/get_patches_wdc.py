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
parser.add_argument('--src_dir', default='../datasets/WDC/', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../datasets/WDC/train', type=str, help='Directory for image patches')
parser.add_argument('--ps', default=64, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=1024, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=8, type=int, help='Number of CPU Cores')
parser.add_argument('--train_flag', action='store_true',default=True)
parser.add_argument('--K', default=24, type=int, help='Number of bands')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores
train_flag = args.train_flag
K = args.K

if os.path.exists(tar):
    os.system("rmdir -r {}".format(tar))

os.makedirs(tar)

#get sorted folders
files = natsorted(glob(os.path.join(src, '*.mat')))

print(files)

def save_output(dir, data, name):
    mdic = {"gt": data}
    savemat(dir + "/" + name + ".mat", mdic)

def save_files(i):
    clean_file = files[i]
    clean_img = loadmat(clean_file)

    H = clean_img['data'].shape[0]
    W = clean_img['data'].shape[1]
    bands = clean_img['data'].shape[-1]
    clean = np.concatenate([clean_img['data'][..., range(int(K / 2), 0, -1)],
                            clean_img['data'],
                            clean_img['data'][
                                ..., range(bands - 1, bands - int(K / 2), -1)]], axis=-1)
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        kk = np.random.randint(0, bands)

        clean_patch = clean[rr:rr + PS, cc:cc + PS, kk:kk + K]

        save_output(tar, clean_patch, 'patch_{}_{}'.format(i+1,j+1))

Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(0, 1)))




