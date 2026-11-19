"""Denoise a test HSI with a trained SM-CNN and report MPSNR / MSSIM / SAM.

    python predict.py --config-file gauss_blind_config.yaml

`test_file` (config, or --test-file) is either a .mat file with the full clean cube
`gt` [H, W, B] and noisy cube `input` (written by get_patches_wdc_test.py) or a
directory of legacy per-band windows. If the .mat file has no `input`, the noise of
`test_noisy_type` (default: `noisy_type`) is synthesized with a fixed seed.

As in the paper (Fig. 1), the noisy cube is mirror-padded along the spectral axis
and every band is denoised from its K-band window; the image is processed at full
spatial size since the network is fully convolutional.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_denoising import load_test_cube
from dataset_utils import pad_spectral
from metrics import cube_metrics
from model import build_model, center_band_index
from noise import build_noise
from utils import load_checkpoint, load_config, save_output, select_gpus


def parse_args():
    parser = argparse.ArgumentParser(description='Test SM-CNN')
    parser.add_argument('--config-file', required=True, help='path to the config file')
    parser.add_argument('--checkpoint', help='overrides checkpoint_path of the config')
    parser.add_argument('--test-file', help='overrides test_file of the config')
    parser.add_argument('--batch-size', type=int, default=8, help='bands per forward pass')
    parser.add_argument('--seed', type=int, default=0, help='seed for synthesized test noise')
    return parser.parse_args()


@torch.no_grad()
def denoise_cube(model, noisy, K, device, batch_size=8):
    """Denoise a band-first cube [B, H, W] band by band from K-band windows."""
    B = noisy.shape[0]
    center = center_band_index(K)
    padded = pad_spectral(noisy, K, axis=0)  # window k = padded[k:k + K], band k at index `center`
    denoised = np.empty_like(noisy)
    for start in range(0, B, batch_size):
        stop = min(start + batch_size, B)
        volume = np.stack([padded[k:k + K] for k in range(start, stop)])            # [b, K, H, W]
        volume = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(1).to(device)  # [b, 1, K, H, W]
        output = model(volume[:, :, center], volume)                               # [b, 1, H, W]
        denoised[start:stop] = output.squeeze(1).cpu().numpy()
    return denoised


def false_color(cube, bands):
    """[B, H, W] cube -> [H, W, 3] image from the given band indices."""
    return np.stack([cube[b] for b in bands], axis=-1)


def main():
    args = parse_args()
    config = load_config(args.config_file)
    device = select_gpus(config)
    print(f'Using {device} device')

    K = config.K
    train_type = config.get('train_type', 'n2c')
    arch = config.get('arch', 'SM-CNN')
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(config).to(device).eval()
    checkpoint_path = args.checkpoint or config.checkpoint_path
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path} (train the model first)')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_checkpoint(model, checkpoint)
    print(f"Loaded {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")

    test_file = args.test_file or config.test_file
    print(f'===> Loading test data {test_file}')
    clean, noisy = load_test_cube(test_file)
    noise_type = config.get('test_noisy_type', config.noisy_type)
    if noisy is None:
        noise = build_noise(noise_type, config.get('noise_level'))
        noisy = noise(clean, np.random.default_rng(args.seed))
        print(f"'input' not found in the test file: synthesized '{noise_type}' noise (seed {args.seed})")
    print(f'Test cube: {clean.shape[1]}x{clean.shape[2]} pixels, {clean.shape[0]} bands')

    print('Testing...')
    start = time.time()
    denoised = denoise_cube(model, noisy, K, device, args.batch_size)
    elapsed = time.time() - start
    print(f'Test completed... Elapsed = {elapsed:.4f} s')

    clean = np.clip(clean, 0, 1)
    noisy = np.clip(noisy, 0, 1)
    denoised = np.clip(denoised, 0, 1)
    results = {'noisy': cube_metrics(clean, noisy), arch: cube_metrics(clean, denoised),
               'runtime_s': elapsed, 'checkpoint': checkpoint_path, 'test_file': test_file}
    for name in ('noisy', arch):
        m = results[name]
        print(f"{name:>8}: MPSNR {m['MPSNR']:.3f}  MSSIM {m['MSSIM']:.4f}  SAM {m['SAM']:.4f}")

    save_output(denoised.transpose(1, 2, 0), save_dir, train_type)  # [H, W, B] under key 'data'
    with open(os.path.join(save_dir, f'metrics_{train_type}.json'), 'w') as fp:
        json.dump(results, fp, indent=2)

    B = clean.shape[0]
    rgb_bands = config.get('rgb_bands', [57, 27, 17])  # false-color bands of the paper for WDC
    if max(rgb_bands) >= B:
        rgb_bands = [int(B * 0.75), int(B * 0.45), int(B * 0.15)]
    if noise_type == 'gauss' and config.get('noise_level') is not None:
        noisy_title = r'Gürültülü görüntü ($\sigma$=%d)' % round(config.noise_level * 255)
    else:
        noisy_title = f'Gürültülü görüntü ({noise_type})'
    m = results[arch]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, (clean, noisy, denoised),
                              ('Temiz görüntü', noisy_title,
                               f"{arch} (PSNR={m['MPSNR']:.3f}, SSIM={m['MSSIM']:.3f})")):
        ax.imshow(false_color(img, rgb_bands))
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'rgb_{train_type}.png'), bbox_inches='tight')
    plt.close(fig)
    print('Done!')


if __name__ == '__main__':
    main()
