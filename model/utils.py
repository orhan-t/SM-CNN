import os
from collections import OrderedDict

import numpy as np
import scipy.io as io
import torch
import yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(path):
    with open(path) as fp:
        return AttrDict(yaml.safe_load(fp))


def select_gpus(config):
    """Restrict the visible GPUs to config['gpus'] (e.g. "0,1"); call before any CUDA use."""
    if config.get('gpus') is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwrap(model):
    """Return the underlying model of a DataParallel wrapper."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _np_noise(volume, noise_level):
    shape = volume.shape
    noise = np.random.normal(0.0, noise_level, shape).astype(np.float32)
    noise_simulated_data = volume + noise
    noise_simulated_data = np.clip(noise_simulated_data, 0, 1)
    return noise_simulated_data


def save_output(data, path, train_type):
    """Save a denoised cube [H, W, B] to <path>/hsi_<train_type>.mat under key 'data'."""
    io.savemat(os.path.join(path, 'hsi_' + train_type + '.mat'), {'data': data})


def load_checkpoint(model, checkpoint):
    """Load a checkpoint saved by train.py (with or without the DataParallel 'module.' prefix)."""
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = OrderedDict((k[len('module.'):] if k.startswith('module.') else k, v)
                             for k, v in state_dict.items())
    if any(k.startswith('conv_block.') for k in state_dict):
        raise RuntimeError(
            'This checkpoint comes from the previous SM-CNN implementation (blocks with shared '
            'weights, gamma == beta and an off-by-one target band). It is not compatible with the '
            'fixed model; retrain it with train.py.')
    model.load_state_dict(state_dict)
