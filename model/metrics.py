from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

def _psnr(x,y_out):
    #max_val = tf.maximum(x,y_out)
    x = np.transpose(np.squeeze(x.detach().cpu().numpy(),axis=1), axes=(1, 2, 0))
    y = np.transpose(np.squeeze(y_out.detach().cpu().numpy(),axis=1), axes=(1, 2, 0))
    psnr_val = peak_signal_noise_ratio(x, y, data_range=1)
    return psnr_val

def _ssim(x,y_out):
    #max_val = tf.maximum(x,y_out)
    shape = x.detach().cpu().numpy().shape
    x = np.transpose(np.squeeze(x.detach().cpu().numpy(),axis=1), axes=(1, 2, 0))
    y = np.transpose(np.squeeze(y_out.detach().cpu().numpy(),axis=1), axes=(1, 2, 0))
    ssim_val = structural_similarity(x, y, multichannel=True, data_range=1)
    return ssim_val
