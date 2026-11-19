"""Quality metrics used in the paper (Sec. 4): MPSNR, MSSIM and SAM.

Batch metrics take [N, 1, H, W] tensors (training / validation); cube metrics take
band-first numpy arrays [B, H, W] with values in [0, 1] (evaluation, averaged over
bands for MPSNR / MSSIM and over pixels for SAM, as in the paper).
"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def batch_psnr(clean, output, data_range=1.0):
    """Mean per-image PSNR; the output is clipped to [0, data_range]."""
    output = output.detach().clamp(0, data_range)
    mse = (output - clean.detach()).pow(2).flatten(1).mean(1).clamp_min(1e-12)
    return (10 * torch.log10(data_range ** 2 / mse)).mean().item()


def batch_ssim(clean, output, data_range=1.0):
    """Mean per-image SSIM (scikit-image, on CPU); the output is clipped to [0, data_range]."""
    clean = clean.detach().squeeze(1).cpu().numpy()
    output = output.detach().clamp(0, data_range).squeeze(1).cpu().numpy()
    return float(np.mean([structural_similarity(c, o, data_range=data_range)
                          for c, o in zip(clean, output)]))


def mpsnr(clean, denoised, data_range=1.0):
    return float(np.mean([peak_signal_noise_ratio(c, d, data_range=data_range)
                          for c, d in zip(clean, denoised)]))


def mssim(clean, denoised, data_range=1.0):
    return float(np.mean([structural_similarity(c, d, data_range=data_range)
                          for c, d in zip(clean, denoised)]))


def sam(clean, denoised, eps=1e-8):
    """Mean spectral angle mapper in radians over all pixels."""
    c = clean.reshape(clean.shape[0], -1).astype(np.float64)
    d = denoised.reshape(denoised.shape[0], -1).astype(np.float64)
    cos = (c * d).sum(0) / (np.linalg.norm(c, axis=0) * np.linalg.norm(d, axis=0) + eps)
    return float(np.mean(np.arccos(np.clip(cos, -1.0, 1.0))))


def cube_metrics(clean, denoised):
    return {'MPSNR': mpsnr(clean, denoised), 'MSSIM': mssim(clean, denoised), 'SAM': sam(clean, denoised)}
