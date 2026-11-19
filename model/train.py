"""Train SM-CNN.

    python train.py --config-file gauss_blind_config.yaml            # new run
    python train.py --config-file gauss_blind_config.yaml --resume   # continue from <save_dir>/last_model.pt

Checkpoints written to save_dir:
    <train_type>_model.pt   best validation loss (what predict.py loads via checkpoint_path)
    last_model.pt           latest epoch (used by --resume)
    epoch_<N>_model.pt      every `save_every` epochs
"""
import argparse
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_denoising import get_training_data
from losses import build_loss
from metrics import batch_psnr, batch_ssim
from model import build_model, center_band_index, count_parameters
from noise import build_noise
from utils import load_checkpoint, load_config, select_gpus, unwrap


def parse_args():
    parser = argparse.ArgumentParser(description='Train SM-CNN')
    parser.add_argument('--config-file', required=True, help='path to the config file')
    parser.add_argument('--resume', nargs='?', const='last', default=None,
                        help='resume training from a checkpoint (default: <save_dir>/last_model.pt)')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, center, optimizer=None, compute_ssim=False):
    """One pass over `loader`; trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    totals = {'loss': 0.0, 'psnr': 0.0, 'ssim': 0.0}
    count = 0
    with torch.set_grad_enabled(training):
        for clean_band, noisy_volume in tqdm(loader, desc='train' if training else 'valid', leave=False):
            clean_band = clean_band.to(device, non_blocking=True)        # [N, 1, H, W]
            noisy_volume = noisy_volume.to(device, non_blocking=True)    # [N, 1, K, H, W]
            noisy_band = noisy_volume[:, :, center]                      # same band as the target

            output = model(noisy_band, noisy_volume)
            loss = loss_fn(clean_band, output)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            n = clean_band.shape[0]
            totals['loss'] += loss.item() * n
            totals['psnr'] += batch_psnr(clean_band, output) * n
            if compute_ssim:
                totals['ssim'] += batch_ssim(clean_band, output) * n
            count += n
    return {k: v / count for k, v in totals.items()}


def main():
    args = parse_args()
    config = load_config(args.config_file)
    if config.get('train_type', 'n2c') != 'n2c':
        raise ValueError(f"unsupported train_type '{config.train_type}' (only 'n2c' is implemented)")
    train_type = config.get('train_type', 'n2c')

    seed = config.get('seed', 1234)
    set_seed(seed)
    device = select_gpus(config)
    torch.backends.cudnn.benchmark = True
    print(f'Using {device} device')

    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)

    K = config.K
    center = center_band_index(K)
    img_options = {'patch_size': config.patch_shape[0], 'K': K}
    noise = build_noise(config.noisy_type, config.get('noise_level'))

    print('===> Loading datasets')
    train_data = get_training_data(config.train_file, img_options, noise=noise, augment=True)
    valid_data = get_training_data(config.valid_file, img_options, noise=noise, augment=False,
                                   deterministic=True, seed=seed)
    num_workers = config.get('num_workers', 8)
    loader_kwargs = dict(batch_size=config.batch_size, num_workers=num_workers,
                         pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=False, **loader_kwargs)
    valid_loader = DataLoader(valid_data, shuffle=False, drop_last=False, **loader_kwargs)
    print(f'Training patches: {len(train_data)}, validation patches: {len(valid_data)}')

    model = build_model(config).to(device)
    print(f"Model is {config.get('arch', 'SM-CNN')} ({count_parameters(model):,} parameters)")
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    loss_fn = build_loss(config.loss).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get('lr_step', 40),
                                                gamma=config.get('lr_gamma', 0.1))
    print(f'Training is {train_type}, loss is {config.loss}, noise is {config.noisy_type}')

    start_epoch, best_loss = 1, math.inf
    if args.resume:
        resume_path = os.path.join(save_dir, 'last_model.pt') if args.resume == 'last' else args.resume
        checkpoint = torch.load(resume_path, map_location=device)
        load_checkpoint(unwrap(model), checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', checkpoint.get('loss', math.inf))
        print(f'Resumed from {resume_path} (epoch {checkpoint["epoch"]})')

    writer = SummaryWriter(save_dir)
    best_path = os.path.join(save_dir, f'{train_type}_model.pt')
    last_path = os.path.join(save_dir, 'last_model.pt')
    save_every = config.get('save_every', 10)

    for epoch in range(start_epoch, config.epochs + 1):
        print(f'Epoch {epoch}/{config.epochs}  lr={optimizer.param_groups[0]["lr"]:.2e}')
        train_stats = run_epoch(model, train_loader, loss_fn, device, center, optimizer=optimizer)
        scheduler.step()
        valid_stats = run_epoch(model, valid_loader, loss_fn, device, center, compute_ssim=True)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f"  train loss {train_stats['loss']:.6f}  mpsnr {train_stats['psnr']:.3f}")
        print(f"  valid loss {valid_stats['loss']:.6f}  mpsnr {valid_stats['psnr']:.3f}  "
              f"mssim {valid_stats['ssim']:.4f}")
        writer.add_scalar('Train Loss', train_stats['loss'], epoch)
        writer.add_scalar('Train PSNR', train_stats['psnr'], epoch)
        writer.add_scalar('Test Loss', valid_stats['loss'], epoch)
        writer.add_scalar('Test PSNR', valid_stats['psnr'], epoch)
        writer.add_scalar('Test SSIM', valid_stats['ssim'], epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        is_best = valid_stats['loss'] < best_loss
        if is_best:
            best_loss = valid_stats['loss']
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': unwrap(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': float(valid_stats['loss']),
            'best_loss': float(best_loss),
            'config': dict(config),
        }
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, best_path)
            print(f'  new best model saved to {best_path}')
        if save_every and epoch % save_every == 0:
            torch.save(checkpoint, os.path.join(save_dir, f'epoch_{epoch}_model.pt'))

    writer.close()
    print('Done!')


if __name__ == '__main__':  # required on Windows: DataLoader workers re-import this module
    main()
