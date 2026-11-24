"""Indian household acoustic noise augmenter for NEON Browser Noise-Free Mic.

Synthesizes and injects realistic Indian household interference with probability
`p=0.35`:
- 50 Hz AC mains hum fundamental (1.0x) + 100 Hz (0.5x) + 150 Hz (0.25x) harmonics
- 1/sqrt(f) pink-filtered ceiling fan aerodynamic turbulence
- Mixed at random SNR in `[-2, +15]` dB via RMS scaling, clamped to `[-1.0, 1.0]`.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch


class IndianHouseholdNoiseAugmenter:
    """On-the-fly acoustic noise augmenter for Indian indoor environments."""

    def __init__(
        self,
        sample_rate: int = 16000,
        p: float = 0.35,
        snr_range_db: Tuple[float, float] = (-2.0, 15.0),
        seed: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.p = p
        self.snr_range_db = snr_range_db
        self.rng = np.random.default_rng(seed)

    def synthesize_noise(self, num_samples: int) -> np.ndarray:
        """Synthesize 50/100/150 Hz AC mains hum + 1/sqrt(f) pink ceiling fan turbulence."""
        t = np.arange(num_samples, dtype=np.float64) / float(self.sample_rate)

        # Slight mains frequency drift around 50 Hz (±0.2 Hz) and random phases
        f0 = 50.0 + self.rng.uniform(-0.2, 0.2)
        p1, p2, p3 = self.rng.uniform(0.0, 2.0 * np.pi, size=3)
        hum = (
            1.0 * np.sin(2.0 * np.pi * f0 * t + p1)
            + 0.5 * np.sin(2.0 * np.pi * (2.0 * f0) * t + p2)
            + 0.25 * np.sin(2.0 * np.pi * (3.0 * f0) * t + p3)
        )

        # 1/sqrt(f) pink-filtered ceiling fan turbulence
        white = self.rng.standard_normal(num_samples)
        freqs = np.fft.rfftfreq(num_samples, d=1.0 / self.sample_rate)
        spec = np.fft.rfft(white)
        pink_filter = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
        pink_filter[0] = 0.0  # zero DC offset
        fan = np.fft.irfft(spec * pink_filter, n=num_samples)

        # Subtle blade-pass amplitude modulation (2.5 - 5.0 Hz)
        blade_hz = self.rng.uniform(2.5, 5.0)
        blade_phase = self.rng.uniform(0.0, 2.0 * np.pi)
        fan = fan * (1.0 + 0.12 * np.sin(2.0 * np.pi * blade_hz * t + blade_phase))

        # Normalize components before random blend
        hum = hum / (np.sqrt(np.mean(hum ** 2)) + 1e-8)
        fan = fan / (np.sqrt(np.mean(fan ** 2)) + 1e-8)

        hum_weight = self.rng.uniform(0.25, 0.65)
        noise = hum_weight * hum + (1.0 - hum_weight) * fan
        noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-8)
        return noise.astype(np.float32)

    def __call__(
        self,
        wav: Union[np.ndarray, torch.Tensor],
        force: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inject Indian household noise with probability `self.p` (or always if `force=True`)."""
        if not force and self.rng.random() > self.p:
            return wav

        is_tensor = isinstance(wav, torch.Tensor)
        if is_tensor:
            device = wav.device
            dtype = wav.dtype
            arr = wav.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(wav, dtype=np.float32)

        orig_shape = arr.shape
        flat = arr.reshape(-1)
        sig_rms = float(np.sqrt(np.mean(flat ** 2)) + 1e-8)

        noise = self.synthesize_noise(flat.shape[0])
        noise_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-8)

        snr_db = float(self.rng.uniform(self.snr_range_db[0], self.snr_range_db[1]))
        target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
        scaled_noise = noise * (target_noise_rms / noise_rms)

        mixed = np.clip(flat + scaled_noise, -1.0, 1.0).astype(np.float32).reshape(orig_shape)
        if is_tensor:
            return torch.from_numpy(mixed).to(device=device, dtype=dtype)
        return mixed
