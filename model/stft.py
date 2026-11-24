"""16 kHz Mono STFT / ISTFT utilities for NEON Browser Noise-Free Mic.

Uses n_fft=512, hop_length=256 (16 ms), win_length=512 (32 ms), and a periodic
Hann window (`torch.hann_window(512, periodic=True)`), returning real/imaginary
channel tensors of shape `[B, 2, 257, T]` and reconstructing `[B, L]`.
"""

from typing import Optional

import torch
import torch.nn as nn

N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
SAMPLE_RATE = 16000


def stft(
    wav: torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    center: bool = True,
) -> torch.Tensor:
    """Compute STFT of a 16 kHz mono waveform tensor.

    Args:
        wav: Waveform tensor of shape `[B, L]` or `[L]`.
        n_fft: FFT size (default: 512).
        hop_length: Hop length in samples (default: 256, 16 ms at 16 kHz).
        win_length: Window length in samples (default: 512, 32 ms at 16 kHz).
        center: Whether to pad input on both sides so frame `t` is centered at `t * hop_length`.

    Returns:
        Complex spectrogram split into real and imaginary channels with shape `[B, 2, n_fft // 2 + 1, T]`.
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() != 2:
        raise ValueError(f"Expected 1D [L] or 2D [B, L] waveform, got shape {tuple(wav.shape)}")

    window = torch.hann_window(win_length, periodic=True, device=wav.device, dtype=wav.dtype)
    complex_spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    # complex_spec: [B, 257, T] -> stack real & imag to [B, 2, 257, T]
    return torch.stack([complex_spec.real, complex_spec.imag], dim=1)


def istft(
    spec: torch.Tensor,
    length: Optional[int] = None,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    center: bool = True,
) -> torch.Tensor:
    """Reconstruct 16 kHz mono waveform from `[B, 2, 257, T]` real/imag spectrogram.

    Args:
        spec: Spectrogram tensor of shape `[B, 2, n_fft // 2 + 1, T]`.
        length: Optional exact output waveform length `L`.
        n_fft: FFT size (default: 512).
        hop_length: Hop length in samples (default: 256).
        win_length: Window length in samples (default: 512).
        center: Whether STFT used centered padding.

    Returns:
        Reconstructed waveform tensor of shape `[B, L]`.
    """
    if spec.dim() != 4 or spec.shape[1] != 2:
        raise ValueError(f"Expected spec shape [B, 2, F, T], got {tuple(spec.shape)}")

    window = torch.hann_window(win_length, periodic=True, device=spec.device, dtype=spec.dtype)
    complex_spec = torch.complex(spec[:, 0], spec[:, 1])
    return torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
        return_complex=False,
    )


class STFT(nn.Module):
    """Module wrapper for 16 kHz Mono STFT with a registered periodic Hann window."""

    def __init__(
        self,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.register_buffer("window", torch.hann_window(win_length, periodic=True))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        complex_spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        return torch.stack([complex_spec.real, complex_spec.imag], dim=1)


class ISTFT(nn.Module):
    """Module wrapper for 16 kHz Mono ISTFT with a registered periodic Hann window."""

    def __init__(
        self,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.register_buffer("window", torch.hann_window(win_length, periodic=True))

    def forward(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        complex_spec = torch.complex(spec[:, 0], spec[:, 1])
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            length=length,
            return_complex=False,
        )
