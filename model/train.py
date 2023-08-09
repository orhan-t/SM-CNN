if __name__ == '__main__':
    import os.path
    import numpy as np
    import torch
    from torchvision.transforms import Compose
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from model import init_weight, SMCNN
    import yaml
    import argparse
    from utils import *
    from metrics import _psnr, _ssim
    import random
    from tqdm import tqdm
    from dataset_denoising import get_training_data
    from losses import CharbonnierLoss, L1Loss, MSELoss, SmoothL1Loss, L1MSELoss, MSETVLoss, L1TVLoss,SSLoss
    from torch.utils.tensorboard import SummaryWriter

    torch.backends.cudnn.benchmark = True #data boyutu değişmediğinde runtime kısaltacak şekilde ayarlıyor? data boyutu değişire işe yaramayabilir.

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)


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
    noise_type = config.noisy_type
    batch_size = config.batch_size
    patch_size = config.patch_shape
    train_type = config.train_type
    K = config.K
    train_file = config.train_file
    valid_file = config.valid_file
    arch = config.arch
    loss_type = config.loss
    gpus = config.gpus
    save_dir = config.save_dir


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    ######### Set GPUs ###########
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))


    print('===> Loading datasets')
    img_options_train = {'patch_size':patch_size[0], 'K':K}

    train_data = get_training_data(train_file, img_options_train)
    valid_data = get_training_data(valid_file, img_options_train)


    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    len_trainset = train_dataloader.__len__()
    len_valset = valid_dataloader.__len__()
    print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

    if arch == 'SM-CNN':
        model = SMCNN(config.num_conv2d_filters, config.num_conv3d_filters, config.num_convolutionblock_filters,
                        config.K)
        model.apply(init_weight)
    else:
        print('No model!')


    ######### DataParallel ###########
    model = torch.nn.DataParallel(model)
    model.to(device)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    print("Training is "+ train_type)
    print("Loss is "+loss_type)
    print("Model is "+arch)

    sigmas = [10, 30, 50, 70]
    blind_noise = AddNoiseBlind(sigmas)
    complex_noise = Compose([
        AddNoiseNoniid(sigmas),
        SequentialSelect(
            transforms=[
                lambda x: x,
                AddNoiseImpulse(),
                AddNoiseStripe(),
                AddNoiseDeadline()
            ]
        )
    ])


    def train(dataloader, model, loss_fn, optimizer):
        model.train()
        loss_train = 0
        psnr = 0
        ssim = 0
        sample_count = 0
        model.train()
        for batch, (clean_band, spectral_volume) in enumerate(tqdm(dataloader)):

            if noise_type == 'blind':
                spectral_volume = blind_noise(spectral_volume)
            elif noise_type == "gauss":
                spectral_volume = torch_noise(spectral_volume, noise_level)
            elif noise_type == "complex":
                spectral_volume = complex_noise(spectral_volume)

            spatial_image = spectral_volume[..., int(K / 2) - 1, :, :]

            # send gpu
            clean_band, spectral_volume = clean_band.to(device), spectral_volume.to(device)
            spatial_image = spatial_image.to(device)

            # Compute prediction error
            dn_output = model(spatial_image, spectral_volume)
            #dn_output = model(spectral_volume, bandwidth)
            if train_type == 'n2c':
                loss = loss_fn(clean_band, dn_output)
            else:
                assert 'Unexpected train!'
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            psnr += _psnr(clean_band,dn_output)
            ssim += _ssim(clean_band,dn_output)
            sample_count += 1
            #print(f"Progress: [{sample_count*len(clean_band):>5d}/{size:>5d}]\n")
        scheduler.step()
        average_loss = loss_train/sample_count
        average_psnr = psnr/sample_count
        average_ssim = ssim/sample_count
        print(f"Train loss: {average_loss:>7f}  mpsnr:{average_psnr:>5.3f} mssim:{average_ssim:>.3f}\n")
        return average_loss, average_psnr, average_ssim

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        model.train()
        loss_test = 0
        psnr = 0
        ssim = 0
        sample_count = 0
        with torch.no_grad():
            model.eval()
            for batch, (clean_band, spectral_volume) in enumerate(tqdm(dataloader)):

                if noise_type == 'blind':
                    spectral_volume = blind_noise(spectral_volume)
                elif noise_type == "gauss":
                    spectral_volume = torch_noise(spectral_volume, noise_level)
                elif noise_type == "complex":
                    spectral_volume = complex_noise(spectral_volume)

                spatial_image = spectral_volume[..., int(K / 2) - 1, :, :]
                # send gpu
                clean_band, spectral_volume = clean_band.to(device), spectral_volume.to(device)
                spatial_image = spatial_image.to(device)

                # Compute prediction error
                dn_output = model(spatial_image, spectral_volume)
                #residue = model(spectral_volume, bandwidth)
                if train_type == 'n2c':
                    loss = loss_fn(clean_band, dn_output)

                loss_test += loss.item()
                psnr += _psnr(clean_band, dn_output)
                ssim += _ssim(clean_band, dn_output)
                sample_count += 1
                # print(f"Progress: [{sample_count*len(clean_band):>5d}/{size:>5d}]\n")

        average_loss = loss_test / sample_count
        average_psnr = psnr / sample_count
        average_ssim = ssim / sample_count
        print(f"Test loss: {average_loss:>7f}  mpsnr:{average_psnr:>5.3f} mssim:{average_ssim:>.3f}\n")
        torch.cuda.empty_cache()
        return average_loss, average_psnr, average_ssim


    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = 0
        best_loss = np.array(1e6)
        print(f"Checkpoint epoch {init_epoch:>3d} is loaded!\n")
        print(checkpoint_path)
    else:
        init_epoch = 0
        best_loss = np.array(1e6)
        print('No checkpoint!')


    writer = SummaryWriter(save_dir)

    for t in range(init_epoch+1,num_epochs+1):
        print(f"Epoch {t}\n-------------------------------")
        train_loss_avg, train_psnr_avg, train_ssim_avg = train(train_dataloader, model, loss_fn, optimizer)
        writer.add_scalar('Train Loss', train_loss_avg, t)
        writer.add_scalar('Train PSNR', train_psnr_avg, t)
        writer.add_scalar('Train SSIM', train_ssim_avg, t)

        test_loss_avg, test_psnr_avg, test_ssim_avg = test(valid_dataloader, model, loss_fn)
        is_best = bool(test_loss_avg < best_loss)
        writer.add_scalar('Test Loss', test_loss_avg, t)
        writer.add_scalar('Test PSNR', test_psnr_avg, t)
        writer.add_scalar('Test SSIM', test_ssim_avg, t)


        if is_best:
            best_loss = test_loss_avg

            torch.save({
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                }, save_dir + '/' + train_type + '_model.pt')
        if t in [10, 20, 30, 40, 50, 80, 100]:
            torch.save({
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_dir + '/' + 'epoch_' + str(t) + '_model.pt')

    print("Done!")
