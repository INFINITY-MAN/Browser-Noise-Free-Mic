"""Clean-room strictly causal GTCRN Conv-GRU Complex Ratio Mask (CRM) network.

Architecture summary (~85K–88K parameters):
- Encoder: 4 strictly causal 2D Conv blocks (kt=3, causal left-pad=2 frames, 0 future frames),
  frequency downsampling stride 2: 257 -> 129 -> 65 -> 33 -> 17, channels 2 -> 16 -> 24 -> 32 -> 32,
  with BatchNorm2d + PReLU.
- Bottleneck: Pointwise projection (32x17=544 -> 8x17=136 = 4x34), 4-group Grouped 2-layer GRU
  (num_groups=4, num_layers=2, hidden_size=34 per group), and pointwise projection back (8 -> 32).
- Decoder: 4 strictly causal Conv2d (kt=3) + frequency ConvTranspose2d upsampling blocks with
  additive skip connections from encoder, outputting 2-channel Complex Ratio Mask (m_r, m_i)
  via tanh, applied via complex multiplication:
      enh_r = m_r * x_r - m_i * x_i
      enh_i = m_r * x_i + m_i * x_r
- GTCRNStreamWrapper: Wraps the exact trained GTCRNModel weights for 1-frame (16 ms) stateful
  causal streaming, mathematically identical to batch forward (max_abs_diff < 1e-4).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock2d(nn.Module):
    """Strictly causal 2D Conv block (kf=3, kt=3, causal left-pad=2, stride_f=2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_f: int = 2,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(stride_f, 1),
            padding=(1, 0),
            groups=groups,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(num_parameters=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal left-padding of 2 frames along the time axis (last dim), 0 future frames.
        x_pad = F.pad(x, (2, 0, 0, 0))
        return self.act(self.bn(self.conv(x_pad)))

    def forward_stream(
        self, x_frame: torch.Tensor, cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-frame streaming step with `[1, C_in, F_in, 2]` history cache."""
        x_cat = torch.cat([cache, x_frame], dim=-1)  # [1, C_in, F_in, 3]
        new_cache = x_cat[:, :, :, 1:]  # [1, C_in, F_in, 2]
        y = self.act(self.bn(self.conv(x_cat)))  # [1, C_out, F_out, 1]
        return y, new_cache


class CausalDeconvBlock2d(nn.Module):
    """Strictly causal decoder block: causal kt=3 Conv2d + frequency-only ConvTranspose2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_groups: int = 1,
        final_mask: bool = False,
    ) -> None:
        super().__init__()
        self.final_mask = final_mask
        mid_channels = in_channels if final_mask else out_channels

        self.conv = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 0),
            groups=conv_groups,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.PReLU(num_parameters=mid_channels)

        # Frequency-only upsampling: (F_in - 1) * 2 - 2*1 + 3 = 2*F_in - 1
        # Maps 17 -> 33 -> 65 -> 129 -> 257 with zero temporal mixing (strictly causal).
        up_groups = 1 if final_mask else mid_channels
        self.upsample = nn.ConvTranspose2d(
            mid_channels,
            out_channels,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            groups=up_groups,
            bias=True,
        )
        if not final_mask:
            self.up_bn = nn.BatchNorm2d(out_channels)
            self.up_act = nn.PReLU(num_parameters=out_channels)
        else:
            self.up_bn = None
            self.up_act = None

    def _post_conv(self, h: torch.Tensor) -> torch.Tensor:
        h = self.upsample(h)
        if not self.final_mask:
            h = self.up_act(self.up_bn(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (2, 0, 0, 0))
        h = self.act(self.bn(self.conv(x_pad)))
        return self._post_conv(h)

    def forward_stream(
        self, x_frame: torch.Tensor, cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-frame streaming step with `[1, C_in, F_in, 2]` history cache."""
        x_cat = torch.cat([cache, x_frame], dim=-1)  # [1, C_in, F_in, 3]
        new_cache = x_cat[:, :, :, 1:]  # [1, C_in, F_in, 2]
        h = self.act(self.bn(self.conv(x_cat)))  # [1, mid, F_in, 1]
        y = self._post_conv(h)  # [1, C_out, 2*F_in-1, 1]
        return y, new_cache


class GTCRNModel(nn.Module):
    """Strictly causal GTCRN Conv-GRU Complex Ratio Mask network (~86K parameters).

    Input:
        noisy_spec: `[B, 2, 257, T]` (Real, Imag)
    Output:
        enh_spec: `[B, 2, 257, T]` (Real, Imag)
    """

    NUM_GROUPS = 4
    GRU_LAYERS = 2
    GRU_HIDDEN = 34
    BOTTLENECK_FREQ = 17
    BOTTLENECK_CHANNELS = 32
    BOTTLENECK_PROJ_CHANNELS = 8  # 8 * 17 = 136 = 4 * 34

    def __init__(self) -> None:
        super().__init__()
        # Encoder: 257 -> 129 -> 65 -> 33 -> 17, channels 2 -> 16 -> 24 -> 32 -> 32
        self.enc0 = CausalConvBlock2d(2, 16, stride_f=2, groups=1)
        self.enc1 = CausalConvBlock2d(16, 24, stride_f=2, groups=1)
        self.enc2 = CausalConvBlock2d(24, 32, stride_f=2, groups=2)
        self.enc3 = CausalConvBlock2d(32, 32, stride_f=2, groups=2)

        # Bottleneck projection: [B, 32, 17, T] -> [B, 8, 17, T] (8*17 = 136 = 4*34)
        self.proj_in = nn.Sequential(
            nn.Conv2d(self.BOTTLENECK_CHANNELS, self.BOTTLENECK_PROJ_CHANNELS, kernel_size=1),
            nn.BatchNorm2d(self.BOTTLENECK_PROJ_CHANNELS),
            nn.PReLU(num_parameters=self.BOTTLENECK_PROJ_CHANNELS),
        )

        # 4-group Grouped 2-layer GRU (hidden_size=34 per group)
        self.grus = nn.ModuleList(
            [
                nn.GRU(
                    input_size=self.GRU_HIDDEN,
                    hidden_size=self.GRU_HIDDEN,
                    num_layers=self.GRU_LAYERS,
                    batch_first=True,
                )
                for _ in range(self.NUM_GROUPS)
            ]
        )

        # Project back: [B, 8, 17, T] -> [B, 32, 17, T]
        self.proj_out = nn.Sequential(
            nn.Conv2d(self.BOTTLENECK_PROJ_CHANNELS, self.BOTTLENECK_CHANNELS, kernel_size=1),
            nn.BatchNorm2d(self.BOTTLENECK_CHANNELS),
            nn.PReLU(num_parameters=self.BOTTLENECK_CHANNELS),
        )

        # Decoder: 17 -> 33 -> 65 -> 129 -> 257 with additive encoder skip connections
        self.dec0 = CausalDeconvBlock2d(32, 32, conv_groups=2, final_mask=False)
        self.dec1 = CausalDeconvBlock2d(32, 24, conv_groups=2, final_mask=False)
        self.dec2 = CausalDeconvBlock2d(24, 16, conv_groups=1, final_mask=False)
        self.dec3 = CausalDeconvBlock2d(16, 2, conv_groups=1, final_mask=True)

    def _run_grouped_gru_batch(self, z: torch.Tensor) -> torch.Tensor:
        """Run 4-group 2-layer GRU over `[B, 8, 17, T]` -> `[B, 8, 17, T]`."""
        b, c, f, t = z.shape
        # Permute to [B, T, C*F] = [B, T, 136] -> [B, T, 4, 34]
        z_seq = z.permute(0, 3, 1, 2).reshape(b, t, self.NUM_GROUPS, self.GRU_HIDDEN)
        outs = []
        for g in range(self.NUM_GROUPS):
            out_g, _ = self.grus[g](z_seq[:, :, g, :])  # [B, T, 34]
            outs.append(out_g)
        z_out = torch.stack(outs, dim=2)  # [B, T, 4, 34]
        return z_out.reshape(b, t, c, f).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def apply_crm(noisy_spec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply Complex Ratio Mask via complex multiplication."""
        x_r = noisy_spec[:, 0:1]
        x_i = noisy_spec[:, 1:2]
        m_r = mask[:, 0:1]
        m_i = mask[:, 1:2]
        enh_r = m_r * x_r - m_i * x_i
        enh_i = m_r * x_i + m_i * x_r
        return torch.cat([enh_r, enh_i], dim=1)

    def forward(self, noisy_spec: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(noisy_spec)  # [B, 16, 129, T]
        e1 = self.enc1(e0)          # [B, 24, 65, T]
        e2 = self.enc2(e1)          # [B, 32, 33, T]
        e3 = self.enc3(e2)          # [B, 32, 17, T]

        # Bottleneck
        z = self.proj_in(e3)                  # [B, 8, 17, T]
        z = self._run_grouped_gru_batch(z)    # [B, 8, 17, T]
        b = self.proj_out(z)                  # [B, 32, 17, T]

        # Decoder with additive skip connections
        d0 = self.dec0(b + e3)      # [B, 32, 33, T]
        d1 = self.dec1(d0 + e2)     # [B, 24, 65, T]
        d2 = self.dec2(d1 + e1)     # [B, 16, 129, T]
        mask = torch.tanh(self.dec3(d2 + e0))  # [B, 2, 257, T]

        return self.apply_crm(noisy_spec, mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GTCRNStreamWrapper(nn.Module):
    """Stateful 1-frame (16 ms) causal streaming wrapper sharing exact GTCRNModel weights.

    Inputs:
        frame_spec: `[1, 2, 257, 1]`
        enc_cache_0: `[1, 2, 257, 2]`
        enc_cache_1: `[1, 16, 129, 2]`
        enc_cache_2: `[1, 24, 65, 2]`
        enc_cache_3: `[1, 32, 33, 2]`
        dec_cache_0: `[1, 32, 17, 2]`
        dec_cache_1: `[1, 32, 33, 2]`
        dec_cache_2: `[1, 24, 65, 2]`
        dec_cache_3: `[1, 16, 129, 2]`
        gru_h_in: `[4, 2, 1, 34]`

    Outputs:
        enh_real: `[1, 1, 257, 1]`
        enh_imag: `[1, 1, 257, 1]`
        enc_cache_out_0..3, dec_cache_out_0..3, gru_h_out `[4, 2, 1, 34]`
    """

    INPUT_SPECS: List[Tuple[str, List[int]]] = [
        ("frame_spec", [1, 2, 257, 1]),
        ("enc_cache_0", [1, 2, 257, 2]),
        ("enc_cache_1", [1, 16, 129, 2]),
        ("enc_cache_2", [1, 24, 65, 2]),
        ("enc_cache_3", [1, 32, 33, 2]),
        ("dec_cache_0", [1, 32, 17, 2]),
        ("dec_cache_1", [1, 32, 33, 2]),
        ("dec_cache_2", [1, 24, 65, 2]),
        ("dec_cache_3", [1, 16, 129, 2]),
        ("gru_h_in", [4, 2, 1, 34]),
    ]

    OUTPUT_SPECS: List[Tuple[str, List[int]]] = [
        ("enh_real", [1, 1, 257, 1]),
        ("enh_imag", [1, 1, 257, 1]),
        ("enc_cache_out_0", [1, 2, 257, 2]),
        ("enc_cache_out_1", [1, 16, 129, 2]),
        ("enc_cache_out_2", [1, 24, 65, 2]),
        ("enc_cache_out_3", [1, 32, 33, 2]),
        ("dec_cache_out_0", [1, 32, 17, 2]),
        ("dec_cache_out_1", [1, 32, 33, 2]),
        ("dec_cache_out_2", [1, 24, 65, 2]),
        ("dec_cache_out_3", [1, 16, 129, 2]),
        ("gru_h_out", [4, 2, 1, 34]),
    ]

    def __init__(self, model: GTCRNModel) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def init_states(
        cls, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, ...]:
        """Return zero-initialized state tensors matching `(enc_cache_0..3, dec_cache_0..3, gru_h_in)`."""
        return tuple(
            torch.zeros(shape, device=device, dtype=dtype)
            for name, shape in cls.INPUT_SPECS
            if name != "frame_spec"
        )

    @classmethod
    def io_metadata(cls) -> Dict[str, object]:
        """Return JSON-serializable I/O metadata for browser WebWorker initialization."""
        return {
            "inputs": [{"name": n, "shape": s, "dtype": "float32"} for n, s in cls.INPUT_SPECS],
            "outputs": [{"name": n, "shape": s, "dtype": "float32"} for n, s in cls.OUTPUT_SPECS],
            "state_pairs": [
                {"in": f"enc_cache_{i}", "out": f"enc_cache_out_{i}", "shape": cls.INPUT_SPECS[1 + i][1]}
                for i in range(4)
            ]
            + [
                {"in": f"dec_cache_{i}", "out": f"dec_cache_out_{i}", "shape": cls.INPUT_SPECS[5 + i][1]}
                for i in range(4)
            ]
            + [{"in": "gru_h_in", "out": "gru_h_out", "shape": [4, 2, 1, 34]}],
            "sample_rate": 16000,
            "n_fft": 512,
            "hop_length": 256,
            "win_length": 512,
        }

    def forward(
        self,
        frame_spec: torch.Tensor,
        enc_cache_0: torch.Tensor,
        enc_cache_1: torch.Tensor,
        enc_cache_2: torch.Tensor,
        enc_cache_3: torch.Tensor,
        dec_cache_0: torch.Tensor,
        dec_cache_1: torch.Tensor,
        dec_cache_2: torch.Tensor,
        dec_cache_3: torch.Tensor,
        gru_h_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        m = self.model

        # Encoder streaming blocks
        e0, enc_cache_out_0 = m.enc0.forward_stream(frame_spec, enc_cache_0)
        e1, enc_cache_out_1 = m.enc1.forward_stream(e0, enc_cache_1)
        e2, enc_cache_out_2 = m.enc2.forward_stream(e1, enc_cache_2)
        e3, enc_cache_out_3 = m.enc3.forward_stream(e2, enc_cache_3)

        # Bottleneck: [1, 32, 17, 1] -> [1, 8, 17, 1] -> 4 grouped GRUs
        z = m.proj_in(e3)  # [1, 8, 17, 1]
        z_seq = z.permute(0, 3, 1, 2).reshape(1, 1, m.NUM_GROUPS, m.GRU_HIDDEN)

        gru_outs = []
        gru_h_outs = []
        for g in range(m.NUM_GROUPS):
            h_in_g = gru_h_in[g]  # [2, 1, 34]
            out_g, h_out_g = m.grus[g](z_seq[:, :, g, :], h_in_g)
            gru_outs.append(out_g)
            gru_h_outs.append(h_out_g)

        z_out = torch.stack(gru_outs, dim=2).reshape(
            1, 1, m.BOTTLENECK_PROJ_CHANNELS, m.BOTTLENECK_FREQ
        ).permute(0, 2, 3, 1).contiguous()
        gru_h_out = torch.stack(gru_h_outs, dim=0)  # [4, 2, 1, 34]

        b = m.proj_out(z_out)  # [1, 32, 17, 1]

        # Decoder streaming blocks with additive skip connections
        d0, dec_cache_out_0 = m.dec0.forward_stream(b + e3, dec_cache_0)
        d1, dec_cache_out_1 = m.dec1.forward_stream(d0 + e2, dec_cache_1)
        d2, dec_cache_out_2 = m.dec2.forward_stream(d1 + e1, dec_cache_2)
        mask_raw, dec_cache_out_3 = m.dec3.forward_stream(d2 + e0, dec_cache_3)
        mask = torch.tanh(mask_raw)  # [1, 2, 257, 1]

        x_r = frame_spec[:, 0:1]
        x_i = frame_spec[:, 1:2]
        m_r = mask[:, 0:1]
        m_i = mask[:, 1:2]
        enh_real = m_r * x_r - m_i * x_i  # [1, 1, 257, 1]
        enh_imag = m_r * x_i + m_i * x_r  # [1, 1, 257, 1]

        return (
            enh_real,
            enh_imag,
            enc_cache_out_0,
            enc_cache_out_1,
            enc_cache_out_2,
            enc_cache_out_3,
            dec_cache_out_0,
            dec_cache_out_1,
            dec_cache_out_2,
            dec_cache_out_3,
            gru_h_out,
        )
