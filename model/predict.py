import os
import os.path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import init_weight, SMCNN
import yaml
import argparse
from utils import _loss, AttrDict, merged_patch, save_output, load_checkpoint
from metrics import _psnr, _ssim
import time
from tqdm import tqdm
from dataset_denoising import get_test_data
from losses import CharbonnierLoss, L1Loss, MSELoss, SmoothL1Loss, L1MSELoss, MSETVLoss, L1TVLoss, SSLoss

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--config-file',help='path to the config file')
args = parser.parse_args()

with open(args.config_file) as fp:
    config = yaml.safe_load(fp)
    config = AttrDict(config)



checkpoint_path = config.checkpoint_path
num_epochs = config.epochs
learning_rate = config.lr
noise_level = config.noise_level
patch_size = config.patch_shape
train_type = config.train_type
K = config.K
arch = config.arch
test_file = config.test_file
loss_type = config.loss
save_dir = config.save_dir

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using {} device".format(device))
print('===> Loading datasets')
img_options_train = {'patch_size':patch_size[0], 'K':K}

test_data = get_test_data(test_file, img_options_train)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

len_trainset = test_dataloader.__len__()
print("Sizeof training set: ", len_trainset)

if arch == 'SM-CNN':
    model = SMCNN(config.num_conv2d_filters, config.num_conv3d_filters, config.num_convolutionblock_filters,
                    config.K)
    model.apply(init_weight)
else:
    print('No model!')

#send model to device
model.to(device)

#loss_fn = _loss
#loss_fn = _loss
if loss_type == 'L1':
    loss_fn = L1Loss().to(device)
elif loss_type == 'MSE':
    loss_fn = MSELoss().to(device)
elif loss_type == 'SL1':
    loss_fn = SmoothL1Loss().to(device)
elif loss_type == 'L1MSE':
    loss_fn = L1MSELoss().to(device)
elif loss_type == 'CHAR':
    loss_fn = CharbonnierLoss().to(device)
elif loss_type == 'MSETV':
    loss_fn = MSETVLoss().to(device)
elif loss_type == 'L1TV':
    loss_fn = L1TVLoss().to(device)
elif loss_type == 'SS':
    loss_fn = SSLoss(alpha=0.9).to(device)
else:
    assert 'Unexpected Loss!'

def test(dataloader, model, loss_fn):
    hsi_image_list = []
    clean_image_list = []
    noisy_image_list = []
    test_loss, average_psnr, average_ssim = 0, 0, 0
    sample_count = 0
    with torch.no_grad():
        model.eval()
        for batch, (clean, spectral_volume) in enumerate(tqdm(dataloader)):
            spatial_image = spectral_volume[..., int(K / 2) - 1, :, :]

            clean_band = clean[..., int(K / 2) - 1, :, :]
            # send gpu
            clean_band, spectral_volume = clean_band.to(device), spectral_volume.to(device)
            spatial_image = spatial_image.to(device)

            # Compute prediction error
            dn_output = model(spatial_image, spectral_volume)
            # dn_output = model(spectral_volume, bandwidth)

            loss = loss_fn(clean_band, dn_output)

            test_loss += loss.item()
            average_psnr += _psnr(clean_band, dn_output)
            average_ssim += _ssim(clean_band, dn_output)
            hsi_image_list.append(np.squeeze(dn_output.detach().cpu().numpy(), axis=0))
            clean_image_list.append(np.squeeze(clean_band.detach().cpu().numpy(), axis=0))
            noisy_image_list.append(np.squeeze(spatial_image.detach().cpu().numpy(), axis=0))
            sample_count += 1

    test_loss /= sample_count
    average_psnr /= sample_count
    average_ssim /= sample_count
    print(f"Test loss: {test_loss:>7f}  mpsnr:{average_psnr:>5.3f} mssim:{average_ssim:>.3f}\n")
    return test_loss, clean_image_list, noisy_image_list, hsi_image_list, average_psnr, average_ssim


if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    #model.load_state_dict(checkpoint['model_state_dict'])
    load_checkpoint(model, checkpoint)
    init_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    print(f"Checkpoint epoch {init_epoch:>3d} is loaded!\n")
    print(checkpoint_path)
else:
    print("No Checkpoint!\n")

print("Testing...\n")
inittime = time.time()
test_loss_avg, clean, noisy, test_output, mpsnr, mssim = test(test_dataloader, model, loss_fn)
elapsed = time.time() - inittime
print(f'Test completed... Elapsed = {elapsed:>.4f}')

clean_image, noise_image, hsi_image = merged_patch(clean, noisy, test_output, shape=(256,256,191), patch=256)
img_ar_noisy = np.clip(noise_image, 0, 1)
img_ar_hsid = np.clip(hsi_image, 0, 1)
img_ar_clean = np.clip(clean_image, 0, 1)

save_output(img_ar_hsid, save_dir, train_type)

fig, (axe11, axe12, axe13) = plt.subplots(1, 3, figsize=(12, 4))
fig.subplots_adjust(wspace=0, hspace=0)

axe11.set_title('Temiz görüntü', fontsize=12)
axe11.imshow(img_ar_clean[:, :, [57, 27, 17]], cmap='gray')
axe12.set_title('Gürültülü görüntü ($\sigma$=%d)' %(round(noise_level*255)), fontsize=12)
axe12.imshow(img_ar_noisy[:, :, [57, 27, 17]], cmap='gray')
axe13.set_title(arch + '(PSNR=%.3f, SSIM=%.3f)' % (mpsnr, mssim), fontsize=12)
axe13.imshow(img_ar_hsid[:, :, [57, 27, 17]], cmap='gray')
plt.tight_layout()
plt.savefig(save_dir + '/rgb_' + train_type + ".png")
#plt.show()
print("Done!")
