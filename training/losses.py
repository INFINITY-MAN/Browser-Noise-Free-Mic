"""Polarity-anchored HybridSpeechLoss for NEON Browser Noise-Free Mic.

Combines:
    L_total = 0.2 * L_SI-SDR + 0.8 * L_Multi-STFT + 0.1 * L1_Waveform

Multi-resolution STFT resolutions (n_fft, win_length, hop_length):
    (256, 128, 64), (512, 256, 128), (1024, 512, 256).
"""

from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Args:
        est: Estimated waveform `[B, L]` or `[L]`.
        ref: Clean reference waveform `[B, L]` or `[L]`.
        eps: Numerical stability epsilon.

    Returns:
        Tensor of shape `[B]` containing SI-SDR in dB.
    """
    if est.dim() == 1:
        est = est.unsqueeze(0)
        ref = ref.unsqueeze(0)

    est_zm = est - est.mean(dim=-1, keepdim=True)
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)

    dot = torch.sum(est_zm * ref_zm, dim=-1, keepdim=True)
    ref_energy = torch.sum(ref_zm * ref_zm, dim=-1, keepdim=True) + eps
    s_target = (dot / ref_energy) * ref_zm
    e_noise = est_zm - s_target

    ratio = (torch.sum(s_target * s_target, dim=-1) + eps) / (
        torch.sum(e_noise * e_noise, dim=-1) + eps
    )
    return 10.0 * torch.log10(ratio + eps)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT spectral convergence + log magnitude L1 loss."""

    def __init__(
        self,
        resolutions: Sequence[Tuple[int, int, int]] = (
            (256, 128, 64),
            (512, 256, 128),
            (1024, 512, 256),
        ),
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.resolutions = list(resolutions)
        self.eps = eps
        for idx, (_, win_len, _) in enumerate(self.resolutions):
            self.register_buffer(f"win_{idx}", torch.hann_window(win_len, periodic=True))

    def _single_res_loss(
        self,
        est: torch.Tensor,
        ref: torch.Tensor,
        n_fft: int,
        win_length: int,
        hop_length: int,
        window: torch.Tensor,
    ) -> torch.Tensor:
        est_spec = torch.stft(
            est,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        ref_spec = torch.stft(
            ref,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        est_mag = torch.sqrt(est_spec.real ** 2 + est_spec.imag ** 2 + self.eps)
        ref_mag = torch.sqrt(ref_spec.real ** 2 + ref_spec.imag ** 2 + self.eps)

        # Spectral convergence (Frobenius norm ratio)
        sc_num = torch.linalg.norm(ref_mag - est_mag, ord="fro", dim=(-2, -1))
        sc_den = torch.linalg.norm(ref_mag, ord="fro", dim=(-2, -1)) + self.eps
        sc_loss = torch.mean(sc_num / sc_den)

        # Log-magnitude L1 loss
        log_mag_loss = F.l1_loss(torch.log(est_mag + self.eps), torch.log(ref_mag + self.eps))
        return 0.5 * (sc_loss + log_mag_loss)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if est.dim() == 1:
            est = est.unsqueeze(0)
            ref = ref.unsqueeze(0)
        total = est.new_tensor(0.0)
        for idx, (n_fft, win_len, hop_len) in enumerate(self.resolutions):
            win = getattr(self, f"win_{idx}")
            total = total + self._single_res_loss(est, ref, n_fft, win_len, hop_len, win)
        return total / len(self.resolutions)


class HybridSpeechLoss(nn.Module):
    """Polarity-anchored HybridSpeechLoss:

        L = 0.2 * (-SI-SDR) + 0.8 * L_Multi-STFT + 0.1 * L1_Waveform
    """

    def __init__(
        self,
        sisdr_weight: float = 0.2,
        stft_weight: float = 0.8,
        l1_weight: float = 0.1,
        resolutions: Sequence[Tuple[int, int, int]] = (
            (256, 128, 64),
            (512, 256, 128),
            (1024, 512, 256),
        ),
    ) -> None:
        super().__init__()
        self.sisdr_weight = sisdr_weight
        self.stft_weight = stft_weight
        self.l1_weight = l1_weight
        self.mr_stft = MultiResolutionSTFTLoss(resolutions=resolutions)

    def forward(
        self,
        est: torch.Tensor,
        ref: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        if est.dim() == 1:
            est = est.unsqueeze(0)
            ref = ref.unsqueeze(0)

        # Polarity-anchored negative SI-SDR (in dB, scaled to keep gradients balanced)
        sisdr_val = si_sdr(est, ref).mean()
        l_sisdr = -sisdr_val
        l_stft = self.mr_stft(est, ref)
        l_l1 = F.l1_loss(est, ref)

        total = (
            self.sisdr_weight * l_sisdr
            + self.stft_weight * l_stft
            + self.l1_weight * l_l1
        )
        if return_components:
            return total, {
                "total": float(total.detach().item()),
                "si_sdr_db": float(sisdr_val.detach().item()),
                "l_stft": float(l_stft.detach().item()),
                "l_l1": float(l_l1.detach().item()),
            }
        return total
