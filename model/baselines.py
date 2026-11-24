"""Baseline speech enhancement models for NEON Browser Noise-Free Mic.

- Model 1: `wiener_denoise(audio_1d, sr=16000)` — Classical Wiener Filter (0 params,
  15th-percentile stationary noise PSD estimate, spectral gain floor 0.05).
- Model 2: `LinearPerceptronMask(nn.Module)` — Unit 2 linear frame-mask baseline
  (exact 132,612 params: Linear(257, 257) + ReLU + Linear(257, 257) + Sigmoid).
"""

import numpy as np
import scipy.signal as spsig
import torch
import torch.nn as nn
import torch.nn.functional as F


def wiener_denoise(
    audio_1d: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    noise_percentile: float = 15.0,
    gain_floor: float = 0.05,
) -> np.ndarray:
    """Classical stationary-noise Wiener Filter (0 learnable parameters).

    Estimates the stationary noise PSD per frequency bin as the 15th percentile of
    the noisy magnitude-squared spectrogram across time, computes the Wiener gain
    with a floor of 0.05, and reconstructs the enhanced waveform via ISTFT.

    Args:
        audio_1d: 1D numpy array of mono audio samples.
        sr: Sample rate in Hz (default: 16000).
        n_fft: FFT size (default: 512).
        hop_length: Hop length in samples (default: 256).
        noise_percentile: Percentile used to estimate stationary noise PSD (default: 15.0).
        gain_floor: Minimum spectral gain floor (default: 0.05).

    Returns:
        1D float32 numpy array of enhanced audio with the same length as `audio_1d`.
    """
    wav = np.asarray(audio_1d, dtype=np.float32).reshape(-1)
    orig_len = wav.shape[0]
    if orig_len == 0:
        return wav.copy()

    noverlap = n_fft - hop_length
    _, _, Zxx = spsig.stft(
        wav,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )

    psd = (np.abs(Zxx) ** 2).astype(np.float64)
    noise_psd = np.percentile(psd, noise_percentile, axis=-1, keepdims=True)

    # Wiener gain: max((P - N) / (P + eps), gain_floor)
    eps = 1e-10
    gain = np.clip((psd - noise_psd) / (psd + eps), gain_floor, 1.0)
    Zxx_enh = Zxx * gain

    _, enh = spsig.istft(
        Zxx_enh,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )

    enh = np.asarray(enh, dtype=np.float32)
    if enh.shape[0] >= orig_len:
        enh = enh[:orig_len]
    else:
        enh = np.pad(enh, (0, orig_len - enh.shape[0]), mode="constant")
    return np.clip(enh, -1.0, 1.0)


class LinearPerceptronMask(nn.Module):
    """Unit 2 linear frame-mask baseline (exact 132,612 parameters).

    Predicts a per-frame, per-bin magnitude ratio mask in `[0, 1]` from the noisy
    magnitude spectrum using a 2-layer MLP (`Linear(257, 257) -> ReLU -> Linear(257, 257) -> Sigmoid`)
    and applies it symmetrically to both real and imaginary STFT channels.
    """

    def __init__(self, n_bins: int = 257) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.fc1 = nn.Linear(n_bins, n_bins)
        self.fc2 = nn.Linear(n_bins, n_bins)

    def forward(self, noisy_spec: torch.Tensor) -> torch.Tensor:
        """Apply learned frame-level magnitude mask.

        Args:
            noisy_spec: `[B, 2, 257, T]` (Real, Imag).

        Returns:
            enh_spec: `[B, 2, 257, T]` (Real, Imag).
        """
        # Magnitude: [B, 257, T]
        mag = torch.sqrt(noisy_spec[:, 0] ** 2 + noisy_spec[:, 1] ** 2 + 1e-8)
        # [B, T, 257]
        h = F.relu(self.fc1(mag.transpose(1, 2)))
        mask = torch.sigmoid(self.fc2(h)).transpose(1, 2).unsqueeze(1)  # [B, 1, 257, T]
        return noisy_spec * mask

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
