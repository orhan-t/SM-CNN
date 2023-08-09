import torch
from torch import nn

# initialization
def init_weight(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.Conv3d:
        nn.init.xavier_normal_(m.weight)

#3D spectral conv
class SpectralConv(nn.Module):
    def __init__(self,in_channel, out_channel, K):
        super(SpectralConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.K = K
        self.conv3d_3 = nn.Conv3d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(K, 3, 3), padding=(0, 1, 1))
        self.conv3d_5 = nn.Conv3d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(K, 5, 5), padding=(0, 2, 2))
        self.conv3d_7 = nn.Conv3d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(K, 7, 7), padding=(0, 3, 3))
        self.relu = nn.ReLU()

    def forward(self, spectral_vol):
        conv1 = torch.squeeze(self.conv3d_3(spectral_vol), dim=2)
        conv2 = torch.squeeze(self.conv3d_5(spectral_vol), dim=2)
        conv3 = torch.squeeze(self.conv3d_7(spectral_vol), dim=2)
        concat_volume = torch.cat([conv3, conv2, conv1], dim=1)
        output = self.relu(concat_volume)
        return output

#2D spatial conv
class SpatialConv(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(SpatialConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv2d_3 = nn.Conv2d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d_5 = nn.Conv2d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(5, 5), padding=(2, 2))
        self.conv2d_7 = nn.Conv2d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(7, 7), padding=(3, 3))
        self.relu = nn.ReLU()

    def forward(self, spatial_band):
        conv1 = self.conv2d_3(spatial_band)
        conv2 = self.conv2d_5(spatial_band)
        conv3 = self.conv2d_7(spatial_band)
        concat_volume = torch.cat([conv3, conv2, conv1], dim=1)
        output = self.relu(concat_volume)
        return output


def param_free_norm(x, epsilon=1e-5) :
    x_var, x_mean = torch.var_mean(x, dim=[2, 3], keepdim=True)
    x_std = torch.sqrt(x_var + epsilon)
    return (x - x_mean) / x_std

# spectral self-modulation module
class ssmm(nn.Module):
    def __init__(self,in_channel, out_channel, k):
        super(ssmm, self).__init__()
        self.conv2_3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_5 = nn.Conv2d(in_channels=k, out_channels=out_channel, kernel_size=(5, 5), padding=(2, 2))
        self.relu = nn.ReLU()

    def forward(self, x_init, adj_spectral):
        x = param_free_norm(x_init)
        tmp = self.conv2_5(adj_spectral)
        tmp = self.relu(tmp)
        noisemap_gamma = self.conv2_3(tmp)
        noisemap_beta = self.conv2_3(tmp)
        x = x * (1 + noisemap_gamma) + noisemap_beta
        return x

#spectral self-modulation residual block
class ssmrb(nn.Module):
    def __init__(self, in_channel, out_channel, k=24):
        super(ssmrb, self).__init__()
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), padding=(1, 1))
        self.ssmm = ssmm(in_channel, out_channel, k)
        self.lrelu = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x_init, adj_spectral):
        x = self.ssmm(x_init, adj_spectral)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.ssmm(x, adj_spectral)
        x = self.lrelu(x)
        x = self.conv2(x)
        return x + x_init

#conv block
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d_2 = nn.Conv2d(in_channels=out_channel, out_channels=
                                  out_channel, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d_3 = nn.Conv2d(in_channels=out_channel, out_channels=
                                  int(out_channel/4), kernel_size=(3, 3), padding=(1, 1))
        self.conv2d_4 = nn.Conv2d(in_channels=out_channel, out_channels=
                                  1, kernel_size=(3, 3), padding=(1, 1))
        self.ssmrb = ssmrb(out_channel, out_channel)
        self.relu = nn.ReLU()

    def forward(self,spectral_volume, volume):
        spectral_volume = torch.squeeze(spectral_volume, dim=1)

        conv1 = self.relu(self.conv2d_1(volume))
        conv2 = self.ssmrb(conv1, spectral_volume)
        conv2 = self.ssmrb(conv2, spectral_volume)
        conv3 = self.relu(self.conv2d_2(conv2))
        conv4 = self.ssmrb(conv3, spectral_volume)
        conv4 = self.ssmrb(conv4, spectral_volume)
        conv5 = self.relu(self.conv2d_2(conv4))
        conv6 = self.ssmrb(conv5, spectral_volume)
        conv6 = self.ssmrb(conv6, spectral_volume)
        conv7 = self.relu(self.conv2d_2(conv6))
        conv8 = self.ssmrb(conv7, spectral_volume)
        conv8 = self.ssmrb(conv8, spectral_volume)
        conv9 = self.relu(self.conv2d_2(conv8))

        f_conv3 = self.conv2d_3(conv3)
        f_conv5 = self.conv2d_3(conv5)
        f_conv7 = self.conv2d_3(conv7)
        f_conv9 = self.conv2d_3(conv9)
        final_volume = torch.cat([f_conv3, f_conv5, f_conv7, f_conv9], dim=1)
        final_volume = self.relu(final_volume)
        clean_band = self.conv2d_4(final_volume)
        return clean_band

#self-modulation CNN
class SMCNN(nn.Module):
    def __init__(self,num_3d_filters, num_2d_filters, num_conv_filters, K=24):
        super(SMCNN, self).__init__()
        self.spectral_conv = SpectralConv(in_channel=1, out_channel=num_3d_filters, K=K)
        self.spatial_conv = SpatialConv(in_channel=1, out_channel=num_2d_filters)
        self.conv_block = ConvBlock(in_channel=num_2d_filters*3+num_3d_filters*3, out_channel=num_conv_filters)

    def forward(self, spatial_band, spectral_volume):
        spatial_vol = self.spatial_conv(spatial_band)
        spectral_vol = self.spectral_conv(spectral_volume)
        for_conv_block = torch.cat([spatial_vol, spectral_vol], dim=1)
        residue = self.conv_block(spectral_volume, for_conv_block) + spatial_band
        return residue

