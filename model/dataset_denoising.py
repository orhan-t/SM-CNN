import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from scipy.io import loadmat
from natsort import natsorted
from glob import glob
from dataset_utils import Augment_RGB_torch

augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        self.files_name = natsorted(glob(os.path.join(rgb_dir, '*.mat')))
        self.img_options = img_options
        self.tar_size = len(self.files_name)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        clean = np.float32(loadmat(self.files_name[tar_index])['gt'])
        clean = torch.from_numpy(clean)
        clean = clean.permute(2, 0, 1)
        # Crop Input and Target
        ps = self.img_options['patch_size']
        K = self.img_options['K']
        H = clean.shape[1]
        W = clean.shape[2]
        if H - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        spectral_volume = torch.unsqueeze(clean[:, r:r + ps, c:c + ps], dim=0)
        clean = torch.unsqueeze(clean[int(K/2), r:r + ps, c:c + ps], dim=0)
        apply_trans = transforms_aug[random.getrandbits(3)]
        spectral_volume = getattr(augment, apply_trans)(spectral_volume)
        clean = getattr(augment, apply_trans)(clean)

        return clean, spectral_volume


##################################################################################################
class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform
        self.files_name = natsorted(glob(os.path.join(rgb_dir, '*.mat')))
        self.img_options = img_options
        self.tar_size = len(self.files_name)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        clean = np.float32(loadmat(self.files_name[tar_index])['gt'])
        noisy = np.float32(loadmat(self.files_name[tar_index])['input'])

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        clean = torch.unsqueeze(clean.permute(2, 0, 1), dim=0)
        noisy = torch.unsqueeze(noisy.permute(2, 0, 1), dim=0)

        clean_filename = os.path.split(self.files_name[tar_index])[-1]

        return clean, noisy


##################################################################################################

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)


def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options, None)