import scipy.io as io
import numpy as np
from utils import _np_noise
from torch.utils.data import Dataset

#hyperparameter K; the number of spectral bands to consider

class dataset:
    def __init__(self,patch_shape, config, training=True, validation=True):

        self.train_files = config.train_file
        self.valid_files = config.valid_file
        self.test_files = config.test_file
        self.training = training
        self.validation = validation

        self.K = config.K
        self.shape = patch_shape
        self.noise_level = config.noise_level
        self.train_type = config.train_type
        self.path = config.checkpoint_path

    def load_data(self):
        if self.training:
            image = io.loadmat(self.train_files)
            self.data = np.array(image['gt'], dtype=np.float32)
            self.bands = self.data.shape[-1]
            self.data_noisy = np.array(image['input'], dtype=np.float32)
            if self.train_type == 'n2n':
                self.data_noisy_2 = np.array(image['input2'], dtype=np.float32)

            self.data = np.concatenate([self.data[..., range(int(self.K / 2), 0, -1)],
                                        self.data,
                                        self.data[
                                            ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]], axis=-1)
            print(self.data.shape)
            self.data_noisy = np.concatenate([self.data_noisy[..., range(int(self.K / 2), 0, -1)],
                                              self.data_noisy,
                                              self.data_noisy[
                                                  ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]],
                                             axis=-1)
            if self.train_type == 'n2n':
                self.data_noisy_2 = np.concatenate([self.data_noisy_2[..., range(int(self.K / 2), 0, -1)],
                                                    self.data_noisy_2,
                                                    self.data_noisy_2[
                                                        ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1,
                                                                   -1)]],
                                                   axis=-1)

        elif self.validation:
            image = io.loadmat(self.valid_files)
            self.data = np.array(image['gt'], dtype=np.float32)
            self.bands = self.data.shape[-1]
            self.data_noisy = np.array(image['input'], dtype=np.float32)
            if self.train_type == 'n2n':
                self.data_noisy_2 = np.array(image['input2'], dtype=np.float32)

            self.data = np.concatenate([self.data[..., range(int(self.K / 2), 0, -1)],
                                        self.data,
                                        self.data[
                                            ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]], axis=-1)
            print(self.data.shape)
            self.data_noisy = np.concatenate([self.data_noisy[..., range(int(self.K / 2), 0, -1)],
                                              self.data_noisy,
                                              self.data_noisy[
                                                  ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]],
                                             axis=-1)
            if self.train_type == 'n2n':
                self.data_noisy_2 = np.concatenate([self.data_noisy_2[..., range(int(self.K / 2), 0, -1)],
                                                    self.data_noisy_2,
                                                    self.data_noisy_2[
                                                        ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1,
                                                                   -1)]],
                                                   axis=-1)
        else:
            image = io.loadmat(self.test_files)
            self.data = np.array(image['gt'], dtype=np.float32)
            self.bands = self.data.shape[-1]
            self.data_noisy = np.array(image['input'], dtype=np.float32)
            if self.train_type == 'n2n':
                self.data_noisy_2 = np.array(image['input2'], dtype=np.float32)

            self.data = np.concatenate([self.data[..., range(int(self.K / 2), 0, -1)],
                                        self.data,
                                        self.data[
                                            ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]], axis=-1)
            print(self.data.shape)
            self.data_noisy = np.concatenate([self.data_noisy[..., range(int(self.K / 2), 0, -1)],
                                              self.data_noisy,
                                              self.data_noisy[
                                                  ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]],
                                             axis=-1)
            if self.train_type == 'n2n':
                self.data_noisy_2 = np.concatenate([self.data_noisy_2[..., range(int(self.K / 2), 0, -1)],
                                                    self.data_noisy_2,
                                                    self.data_noisy_2[
                                                        ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1,
                                                                   -1)]],
                                                   axis=-1)
    def load_real_data(self):
        if self.training:
            image = io.loadmat(self.train_files)
            self.data = np.array(image['test_data'], dtype=np.float32)
            self.bands = self.data.shape[-1]
            self.data_noisy = np.concatenate([self.data[..., range(int(self.K / 2), 0, -1)],
                                              self.data,
                                              self.data[
                                                  ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]],
                                             axis=-1)
            self.data = np.concatenate([self.data[..., range(int(self.K / 2), 0, -1)],
                                        self.data,
                                        self.data[
                                            ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]], axis=-1)
            print(self.data.shape)

        else:

            image = io.loadmat(self.test_files)
            self.data = np.array(image['test_data'], dtype=np.float32)
            self.bands = self.data.shape[-1]
            self.data_noisy = np.concatenate([self.data[..., range(int(self.K / 2), 0, -1)],
                                              self.data,
                                              self.data[
                                                  ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]],
                                             axis=-1)
            self.data = np.concatenate([self.data[..., range(int(self.K / 2), 0, -1)],
                                        self.data,
                                        self.data[
                                            ..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]], axis=-1)
            print(self.data.shape)

    def get_patch(self):
        #shape = [20, 20]
        clean_band = []
        bandwidth = []
        spatial_image = []
        spatial_image_adjacent = []
        spectral_volume = []
        """lambda_x = np.array([*np.linspace(0.4, 0.9, 52, endpoint=True), *np.linspace(1.4, 2.4, 139, endpoint=True)]) \
                   / 2.4"""
        lambda_x = np.array([*np.linspace(0.4, 1.809, 149, endpoint=True), *np.linspace(1.943, 2.5, 57, endpoint=True)]) \
                   / 2.5
        bands = np.full((20, 20, len(lambda_x)), lambda_x[0:len(lambda_x)], dtype=np.float32)
        bands = np.concatenate([bands[..., range(int(self.K / 2), 0, -1)],
                                bands,
                                bands[..., range(self.bands - 1, self.bands - int(self.K / 2) - 1, -1)]],
                               axis=-1)
        for z in range(0, self.data.shape[0] - int(self.shape[0]) + 1, int(self.shape[0])):
            for y in range(0, self.data.shape[1] - int(self.shape[1]) + 1, int(self.shape[1])):
                patches_clean = self.data[z:z + self.shape[0], y:y + self.shape[1], :]
                patches = self.data_noisy[z:z + self.shape[0], y:y + self.shape[1], :]
                if self.train_type == 'n2n':
                    patches_2 = self.data_noisy_2[z:z + self.shape[0], y:y + self.shape[1], :]
                for i in range(0, self.bands):
                    clean_band.append(np.expand_dims(patches_clean[:, :, i + int(self.K / 2)], axis=0))
                    spatial_image.append(np.expand_dims(patches[:, :, i + int(self.K / 2)], axis=0))
                    bandwidth.append(np.transpose(bands[:, :, i:i + int(self.K)], axes=(2, 0, 1)))
                    if self.train_type == 'b2b':
                        spectral_volume.append(np.transpose(np.concatenate(
                                               [patches[:, :, i:i + int(self.K/2)-1],
                                                patches[:, :, i + int(self.K/2):i + int(self.K)+1]],
                                               axis=-1), axes=(2, 0, 1)))
                        spatial_image_adjacent.append(np.expand_dims(patches[:, :, i + int(self.K/2)-1], axis=0))

                    elif self.train_type == 'n2n':
                        spatial_image_adjacent.append(np.expand_dims(patches_2[:, :, i + int(self.K / 2)], axis=0))
                        spectral_volume.append(np.transpose(patches[:, :, i:i + int(self.K)],
                                                                           axes=(2, 0, 1)))
                    else:
                        """spectral_volume.append(np.expand_dims(np.transpose(patches[:, :, i:i + int(self.K)],
                                                                           axes=(2, 0, 1)), axis=0))"""
                        spectral_volume.append(np.transpose(patches[:, :, i:i + int(self.K)],
                                                            axes=(2, 0, 1)))
                        spatial_image_adjacent.append(np.expand_dims(patches[:, :, i + int(self.K / 2)], axis=0))

        return clean_band, spatial_image, spatial_image_adjacent, spectral_volume, bandwidth


    def merged_patch(self,clean_image_list,noise_image_list,hsi_image_list):
        clean_image = np.zeros(self.data.shape, dtype=np.float32)
        noise_image = np.zeros(self.data.shape, dtype=np.float32)
        hsi_image = np.zeros(self.data.shape, dtype=np.float32)
        r = int(self.data.shape[1] / self.shape[0])
        for z in range(0, self.data.shape[0] - self.shape[0] + 1, self.shape[0]):
            for y in range(0, self.data.shape[1] - self.shape[1] + 1, self.shape[1]):
                for k in range(0, self.bands):
                    clean_image[z:z + self.shape[0], y:y + self.shape[1], k] = clean_image_list[
                        int(z / self.shape[0] * self.bands * r + y / self.shape[1] * self.bands) + k]
                    noise_image[z:z + self.shape[0], y:y + self.shape[1], k] = noise_image_list[
                        int(z / self.shape[0] * self.bands * r + y / self.shape[1] * self.bands) + k]
                    hsi_image[z:z + self.shape[0], y:y + self.shape[1], k] = hsi_image_list[
                        int(z / self.shape[0] * self.bands * r + y / self.shape[1] * self.bands) + k]
        return clean_image, noise_image, hsi_image

    def generator(self):
        o_clean_band = []
        o_bandwidth = []
        o_spatial_image = []
        o_spatial_image_adjacent = []
        o_spectral_volume = []
        clean_band, spatial_image, spatial_image_adjacent, spectral_volume, bandwidth = self.get_patch()
        for i in range(0, len(clean_band)):
            if self.training:
                for k in range(0, 4):
                    o_clean_band.append(np.rot90(clean_band[i], k=k, axes=(1,2)).copy())
                    o_spatial_image.append(np.rot90(spatial_image[i], k=k, axes=(1,2)).copy())
                    o_spatial_image_adjacent.append(np.rot90(spatial_image_adjacent[i], k=k, axes=(1,2)).copy())
                    o_spectral_volume.append(np.rot90(spectral_volume[i], k=k, axes=(1,2)).copy())
                    o_bandwidth.append(np.rot90(bandwidth[i], k=k, axes=(1,2)).copy())
            else:
                o_clean_band = clean_band
                o_spatial_image = spatial_image
                o_spatial_image_adjacent = spatial_image_adjacent
                o_spectral_volume = spectral_volume
                o_bandwidth = bandwidth

        return o_clean_band, o_spatial_image, o_spatial_image_adjacent, o_spectral_volume, o_bandwidth

    def save_output(self, data):
        mdic = {"data": data}
        io.savemat(self.path + "/hsi_" + self.train_type + ".mat", mdic)


class MyDataset(Dataset):
    def __init__(self, patch_shape, config, training=True, validation=False, transform=None):
        self.data = dataset(patch_shape, config, training=training, validation=validation)
        self.data.load_data()
        #self.data.load_real_data()
        self.transform = transform
        self.clean_band, self.spatial_image, self.spatial_image_adjacent, self.spectral_volume, self.bandwidth\
            = self.data.generator()
        self.merged_patch = self.data.merged_patch
        self.save_output = self.data.save_output

    def __len__(self):
        return len(self.clean_band)

    def __getitem__(self, idx):
        if self.transform:
            clean_band = self.transform(self.clean_band[idx])
            spatial_image = self.transform(self.spatial_image[idx])
            spatial_image_adjacent = self.transform(self.spatial_image_adjacent[idx])
            spectral_volume = self.transform(self.spectral_volume[idx])
            bandwidth = self.transform(self.bandwidth[idx])
        else:
            clean_band = self.clean_band[idx]
            spatial_image = self.spatial_image[idx]
            spatial_image_adjacent = self.spatial_image_adjacent[idx]
            spectral_volume = self.spectral_volume[idx]
            bandwidth = self.bandwidth[idx]
        return clean_band, spatial_image, spatial_image_adjacent, spectral_volume, bandwidth
