import numpy as np
from torch.nn import L1Loss, MSELoss
import torch
import scipy.io as io
from collections import OrderedDict
import threading

def _loss(x,y_out):
    #loss = torch.nn.MSELoss(y_out, x-x_noisy).
    """shape = x.shape
    x = torch.reshape(x, shape=(shape[0], shape[2], shape[3]))
    x_noisy = torch.reshape(x_noisy, shape=(shape[0], shape[2], shape[3]))
    y_out = torch.reshape(y_out, shape=(shape[0], shape[2], shape[3]))"""
    loss = L1Loss()(x, y_out)
    #loss = torch.norm(y_out-(x-x_noisy), p=2)
    #loss = torch.norm(y_out-x, p=2)
    #loss = tf.norm(y_out-x, ord='euclidean')

    return loss

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def _np_noise(volume,noise_level):
    shape = volume.shape
    noise = np.random.normal(0.0, noise_level, shape).astype(np.float32)
    noise_simulated_data = volume + noise
    noise_simulated_data = np.clip(noise_simulated_data, 0, 1)
    return noise_simulated_data

def torch_noise(volume,noise_level):
    shape = volume.shape
    noise = torch.normal(0.0, noise_level, size=shape)
    noise_simulated_data = volume + noise
    #noise_simulated_data = np.clip(noise_simulated_data, 0, 1)
    return noise_simulated_data

def torch_blindnoise(volume, min_sigma, max_sigma):
    shape = volume.shape
    noise = np.random.randn(shape) * np.random.uniform(min_sigma, max_sigma) / 255
    noise_simulated_data = volume + torch.from_numpy(noise)
    #noise_simulated_data = np.clip(noise_simulated_data, 0, 1)
    return noise_simulated_data

def merged_patch(clean_image_list,noise_image_list,hsi_image_list, shape=(200, 200, 191), patch=20):
    clean_image = np.zeros(shape, dtype=np.float32)
    noise_image = np.zeros(shape, dtype=np.float32)
    hsi_image = np.zeros(shape, dtype=np.float32)
    num_patches = int(shape[0] / patch)
    for x in range(0, num_patches):
        for y in range(0, num_patches):
            for z in range(0, shape[2]):
                clean_image[x * patch:x * patch + patch, y * patch:y * patch + patch, z] = \
                    clean_image_list[x*num_patches*shape[2]+ y*shape[2] + z]
                noise_image[x * patch:x * patch + patch, y * patch:y * patch + patch, z] = \
                    noise_image_list[x*num_patches*shape[2]+ y*shape[2] + z]
                hsi_image[x * patch:x * patch + patch, y * patch:y * patch + patch, z] = \
                    hsi_image_list[x*num_patches*shape[2]+ y*shape[2] + z]
    return clean_image, noise_image, hsi_image

def save_output(data, path, train_type):
    mdic = {"data": data}
    io.savemat(path + "/hsi_"+train_type+".mat", mdic)

def load_checkpoint(model, checkpoint):
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except:
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint2(model, checkpoint):
    try:
        model.load_state_dict(checkpoint["net"])
    except:
        state_dict = checkpoint["net"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


class AddNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.

    def __call__(self, img):
        bwsigmas = np.reshape(self.sigmas[np.random.randint(0, len(self.sigmas), img.shape[2])], (-1, 1, 1))
        noise = torch.from_numpy(np.float32(np.random.randn(*img.shape) * bwsigmas))
        return img + noise


class AddNoiseBlind(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""
    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.
        self.pos = LockedIterator(self.__pos(len(sigmas)))

    def __call__(self, img):
        noise = np.random.randn(*img.shape) * self.sigmas[next(self.pos)]
        volume = img + torch.from_numpy(np.float32(noise))
        return volume

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()


class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True:
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out


class AddNoiseMixed(object):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""

    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        X,Y, B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos:pos + num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img


class _AddNoiseImpulse(object):
    """add impulse noise to the given numpy array (B,H,W)"""

    def __init__(self, amounts, s_vs_p=0.5):
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img, bands):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        bwamounts = self.amounts[np.random.randint(0, len(self.amounts), len(bands))]
        for i, amount in zip(bands, bwamounts):
            self.add_noise(img[:,:, i, ...], amount=amount, salt_vs_pepper=self.s_vs_p)
        return img

    def add_noise(self, image, amount, salt_vs_pepper):
        # out = image.copy()
        out = image
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False], size=image.shape,
                                   p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image.shape,
                                  p=[q, 1 - q])
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = 0
        return out


class _AddNoiseStripe(object):
    """add stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        X, Y, B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_stripe = np.random.randint(np.floor(self.min_amount * W), np.floor(self.max_amount * W), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            stripe = np.random.uniform(0, 1, size=(len(loc),)) * 0.5 - 0.25
            img[:, :, i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img


class _AddNoiseDeadline(object):
    """add deadline noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        X, Y, B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_deadline = np.random.randint(np.ceil(self.min_amount * W), np.ceil(self.max_amount * W), len(bands))
        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            img[:, :, i, :, loc] = 0
        return img


class AddNoiseImpulse(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])]
        self.num_bands = [1 / 3]


class AddNoiseStripe(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseStripe(0.05, 0.15)]
        self.num_bands = [1 / 3]


class AddNoiseDeadline(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseDeadline(0.05, 0.15)]
        self.num_bands = [1 / 3]


class AddNoiseComplex(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [
            _AddNoiseStripe(0.05, 0.15),
            _AddNoiseDeadline(0.05, 0.15),
            _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])
        ]
        self.num_bands = [1 / 3, 1 / 3, 1 / 3]