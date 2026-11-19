"""Synthetic HSI noise models (Sec. 4.2 of the paper, following QRNN3D [10]).

Every noise maker operates on a single band-first float32 array of shape
[B, H, W] with values in [0, 1] and takes an explicit numpy Generator, so the
noise is reproducible (fixed validation / test noise) and independent per
sample and per DataLoader worker.

Noise types (``build_noise``):
  gauss     i.i.d. Gaussian, sigma = noise_level (on the [0, 1] scale)
  blind     i.i.d. Gaussian, sigma drawn per sample from `sigmas` / 255
  noniid    Case 1: per-band Gaussian, sigma drawn per band from `sigmas` / 255
  stripe    Case 2: noniid + stripes on 1/3 of the bands (5-15 % of the columns)
  deadline  Case 3: noniid + dead lines on 1/3 of the bands (5-15 % of the columns)
  impulse   Case 4: noniid + salt & pepper on 1/3 of the bands (10-70 %)
  mixture   Case 5: noniid + stripe, deadline and impulse, each on 1/3 of the bands
  complex   training mixture: noniid + one of {none, impulse, stripe, deadline}
"""
import numpy as np

DEFAULT_SIGMAS = (10, 30, 50, 70)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, rng):
        for t in self.transforms:
            img = t(img, rng)
        return img


class RandomChoice:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, rng):
        return self.transforms[rng.integers(len(self.transforms))](img, rng)


class Identity:
    def __call__(self, img, rng):
        return img


class GaussianNoise:
    def __init__(self, sigma):
        self.sigma = float(sigma)

    def __call__(self, img, rng):
        return img + self.sigma * rng.standard_normal(img.shape, dtype=np.float32)


class BlindGaussianNoise:
    def __init__(self, sigmas=DEFAULT_SIGMAS):
        self.sigmas = np.asarray(sigmas, dtype=np.float32) / 255.

    def __call__(self, img, rng):
        return img + rng.choice(self.sigmas) * rng.standard_normal(img.shape, dtype=np.float32)


class NonIIDGaussianNoise:
    def __init__(self, sigmas=DEFAULT_SIGMAS):
        self.sigmas = np.asarray(sigmas, dtype=np.float32) / 255.

    def __call__(self, img, rng):
        band_sigmas = rng.choice(self.sigmas, size=img.shape[0]).reshape(-1, 1, 1)
        return img + band_sigmas * rng.standard_normal(img.shape, dtype=np.float32)


class _ImpulseNoise:
    def __init__(self, amounts=(0.1, 0.3, 0.5, 0.7), salt_vs_pepper=0.5):
        self.amounts = np.asarray(amounts)
        self.salt_vs_pepper = salt_vs_pepper

    def __call__(self, img, rng, bands):
        for b in bands:
            amount = rng.choice(self.amounts)
            flipped = rng.random(img.shape[1:]) < amount
            salted = rng.random(img.shape[1:]) < self.salt_vs_pepper
            img[b][flipped & salted] = 1
            img[b][flipped & ~salted] = 0
        return img


class _StripeNoise:
    def __init__(self, min_amount=0.05, max_amount=0.15):
        self.min_amount, self.max_amount = min_amount, max_amount

    def __call__(self, img, rng, bands):
        W = img.shape[2]
        lo, hi = int(np.floor(self.min_amount * W)), int(np.floor(self.max_amount * W))
        for b in bands:
            cols = rng.permutation(W)[:rng.integers(lo, max(hi, lo + 1))]
            stripe = rng.uniform(0, 1, size=len(cols)).astype(np.float32) * 0.5 - 0.25
            img[b][:, cols] -= stripe
        return img


class _DeadlineNoise:
    def __init__(self, min_amount=0.05, max_amount=0.15):
        self.min_amount, self.max_amount = min_amount, max_amount

    def __call__(self, img, rng, bands):
        W = img.shape[2]
        lo, hi = int(np.ceil(self.min_amount * W)), int(np.ceil(self.max_amount * W))
        for b in bands:
            cols = rng.permutation(W)[:rng.integers(lo, max(hi, lo + 1))]
            img[b][:, cols] = 0
        return img


class MixedNoise:
    """Apply each band-level noise maker to its own disjoint random subset of bands."""

    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img, rng):
        img = np.array(img, dtype=np.float32, copy=True)
        B = img.shape[0]
        all_bands = rng.permutation(B)
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            noise_maker(img, rng, all_bands[pos:pos + num_band])
            pos += num_band
        return img


def build_noise(noise_type, noise_level=None, sigmas=DEFAULT_SIGMAS):
    """Return a callable ``noise(img [B,H,W], rng) -> noisy img`` (None for 'none')."""
    if noise_type in (None, 'none'):
        return None
    if noise_type == 'gauss':
        if noise_level is None:
            raise ValueError("noise type 'gauss' needs noise_level")
        return GaussianNoise(noise_level)
    if noise_type == 'blind':
        return BlindGaussianNoise(sigmas)
    noniid = NonIIDGaussianNoise(sigmas)
    third = 1 / 3
    cases = {
        'noniid': None,
        'stripe': MixedNoise([_StripeNoise()], [third]),
        'deadline': MixedNoise([_DeadlineNoise()], [third]),
        'impulse': MixedNoise([_ImpulseNoise()], [third]),
        'mixture': MixedNoise([_StripeNoise(), _DeadlineNoise(), _ImpulseNoise()], [third] * 3),
        'complex': RandomChoice([Identity(),
                                 MixedNoise([_ImpulseNoise()], [third]),
                                 MixedNoise([_StripeNoise()], [third]),
                                 MixedNoise([_DeadlineNoise()], [third])]),
    }
    if noise_type not in cases:
        raise ValueError(f"unknown noise type '{noise_type}'; expected one of "
                         f"{['none', 'gauss', 'blind'] + list(cases)}")
    extra = cases[noise_type]
    return noniid if extra is None else Compose([noniid, extra])
