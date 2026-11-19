"""SM-CNN: hyperspectral image denoising via self-modulating CNNs.

Torun et al., "Hyperspectral image denoising via self-modulating convolutional
neural networks", Signal Processing 214 (2024) 109248.

Inputs of the network (Sec. 3.3):
  * spatial_band    [B, 1, H, W]     noisy band to be denoised (y^s)
  * spectral_volume [B, 1, K, H, W]  its K adjacent noisy bands (y^lambda),
                                     with the target band at index K // 2
Output: the denoised band [B, 1, H, W] (residual learning: the network
predicts the residual that is added to spatial_band).

With the default configuration (20/20/60 filters, K = 24, 4 stages x 2 SSMRBs,
15-channel skip convs) the network has 2,404,361 trainable parameters, which is
the number reported in Table 5 of the paper.
"""
import torch
import torch.nn.functional as F
from torch import nn


def center_band_index(K):
    """Index of the target band inside a K-band window.

    Must match the spectral padding in dataset_utils.pad_spectral, which places
    original band k at index K // 2 of window k.
    """
    return K // 2


def init_weight(m):
    """Xavier-normal initialisation of every convolution (Sec. 4.2)."""
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_normal_(m.weight)


class SpatialConv(nn.Module):
    """Multi-scale 2D convolutions on the noisy band (conv2_k3/k5/k7 + ReLU)."""

    def __init__(self, out_channels, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(1, out_channels, k, padding=k // 2) for k in kernel_sizes)

    def forward(self, band):
        return F.relu(torch.cat([conv(band) for conv in self.convs], dim=1))


class SpectralConv(nn.Module):
    """Multi-scale 3D convolutions on the K adjacent bands (conv3_k3/k5/k7 + ReLU).

    The kernel depth equals K and there is no spectral padding, so the spectral
    axis collapses to 1 and the output is a 2D feature map.
    """

    def __init__(self, out_channels, K, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv3d(1, out_channels, (K, k, k), padding=(0, k // 2, k // 2))
            for k in kernel_sizes)

    def forward(self, volume):
        return F.relu(torch.cat([conv(volume).squeeze(2) for conv in self.convs], dim=1))


class SSMM(nn.Module):
    """Spectral self-modulation module (Fig. 3, Eqs. 3-5).

    The feature map is normalised per sample and per channel (Eqs. 4-5, biased
    variance, delta = 1e-5 inside the square root), then scaled and shifted
    pixel-wise by gamma / beta predicted from the K adjacent noisy bands:
    conv_k5 + ReLU (shared), followed by two separate conv_k3 heads.

    gamma is applied as (1 + gamma) so that the module starts close to an
    identity mapping; this is an equivalent re-parameterisation of Eq. 3.
    """

    def __init__(self, channels, K, hidden_channels=None, eps=1e-5):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.shared = nn.Conv2d(K, hidden_channels, 5, padding=2)
        self.gamma = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        self.eps = eps

    def forward(self, x, adj_spectral):
        x = F.instance_norm(x, eps=self.eps)
        h = F.relu(self.shared(adj_spectral))
        return x * (1 + self.gamma(h)) + self.beta(h)


class SSMRB(nn.Module):
    """Spectral self-modulating residual block (Fig. 3).

    SSMM -> LeakyReLU -> conv_k3 -> SSMM -> LeakyReLU -> conv_k3, plus identity skip.
    The two SSMMs and the two convolutions have their own weights.
    """

    def __init__(self, channels, K, negative_slope=0.02):
        super().__init__()
        self.ssmm1 = SSMM(channels, K)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ssmm2 = SSMM(channels, K)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x, adj_spectral):
        out = self.conv1(self.act(self.ssmm1(x, adj_spectral)))
        out = self.conv2(self.act(self.ssmm2(out, adj_spectral)))
        return x + out


class SMCNN(nn.Module):
    """Self-modulating CNN (Fig. 2).

    spatial branch  (3 x 2D conv) --+
                                    +-> concat -> conv_k3+ReLU (head)
    spectral branch (3 x 3D conv) --+
      -> num_stages x [blocks_per_stage x SSMRB -> conv_k3+ReLU] (each stage has a skip)
      -> each skip: conv_k3 (skip_channels) -> concat -> ReLU -> conv_k3 (1 channel)
      -> + spatial_band (global residual)
    Every block has its own weights (nothing is shared between stages).
    """

    def __init__(self, num_conv2d_filters=20, num_conv3d_filters=20,
                 num_convolutionblock_filters=60, K=24, num_stages=4,
                 blocks_per_stage=2, skip_channels=15):
        super().__init__()
        C = num_convolutionblock_filters
        self.K = K
        self.spatial_conv = SpatialConv(num_conv2d_filters)
        self.spectral_conv = SpectralConv(num_conv3d_filters, K)
        self.head = nn.Sequential(
            nn.Conv2d(3 * num_conv2d_filters + 3 * num_conv3d_filters, C, 3, padding=1),
            nn.ReLU())
        self.stages = nn.ModuleList(
            nn.ModuleList(SSMRB(C, K) for _ in range(blocks_per_stage))
            for _ in range(num_stages))
        self.stage_convs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.ReLU())
            for _ in range(num_stages))
        self.skip_convs = nn.ModuleList(
            nn.Conv2d(C, skip_channels, 3, padding=1) for _ in range(num_stages))
        self.tail = nn.Conv2d(num_stages * skip_channels, 1, 3, padding=1)

    def forward(self, spatial_band, spectral_volume):
        if spectral_volume.dim() == 4:  # [B, K, H, W] -> [B, 1, K, H, W]
            spectral_volume = spectral_volume.unsqueeze(1)
        if spectral_volume.shape[2] != self.K:
            raise ValueError(f'expected {self.K} adjacent bands, got {spectral_volume.shape[2]}')
        adj_spectral = spectral_volume.squeeze(1)  # [B, K, H, W], modulation input of every SSMM

        x = torch.cat([self.spatial_conv(spatial_band), self.spectral_conv(spectral_volume)], dim=1)
        x = self.head(x)
        skips = []
        for blocks, stage_conv, skip_conv in zip(self.stages, self.stage_convs, self.skip_convs):
            for block in blocks:
                x = block(x, adj_spectral)
            x = stage_conv(x)
            skips.append(skip_conv(x))
        residual = self.tail(F.relu(torch.cat(skips, dim=1)))
        return residual + spatial_band


MODEL_CONFIG_KEYS = ('num_conv2d_filters', 'num_conv3d_filters', 'num_convolutionblock_filters',
                     'K', 'num_stages', 'blocks_per_stage', 'skip_channels')


def build_model(config):
    """Create the network described by a config (dict/AttrDict) and initialise it."""
    arch = config.get('arch', 'SM-CNN')
    if arch != 'SM-CNN':
        raise ValueError(f"unknown arch '{arch}' (only 'SM-CNN' is implemented)")
    model = SMCNN(**{k: config[k] for k in MODEL_CONFIG_KEYS if k in config})
    model.apply(init_weight)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
