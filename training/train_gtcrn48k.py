"""
GTCRN48kSpectral — 48 kHz Neural Speech Enhancement Training & ONNX Export
==========================================================================
Project: Browser Noise Free Mic
Hardware used for training: Google Colab NVIDIA Tesla T4 GPU (16 GB VRAM)

Datasets (native 48 kHz):
  1. Clean speech: 450 clips from VoiceBank-DEMAND 48 kHz
     (Hugging Face: yairamr/voicebank-demand-fingerprint-48k)
  2. Environmental noise: 120 clips streamed from Microsoft DNS Challenge 5
     fullband archive (noise_fullband.audioset_000.tar.bz2) via io.BytesIO
  3. Synthetic Indian household noise: pink-spectrum ceiling fan turbulence
     + 50 / 100 / 150 Hz AC mains hum, mixed at random SNRs (-2 dB to +15 dB)

Model:
  - GTCRN48kSpectral: 908,663 trainable parameters
  - 960-point periodic sqrt-Hann STFT (480-sample / 10 ms hop, 481 bins)
  - Input/Output tensor: [1, T, 481, 2] real/imaginary Complex Ratio Mask
  - Exported ONNX:
      web/models/gtcrn48k_fp32.onnx (3,644,893 bytes / 3.64 MB)
      web/models/gtcrn48k_int8.onnx (1,342,002 bytes / 1.34 MB)
"""

import os
import time
import random
import urllib.request

# Optional: set HF_TOKEN env var for faster Hugging Face dataset streaming
HF_TOKEN = os.environ.get("HF_TOKEN", None)

OUTPUT_MODELS_DIR = os.environ.get("OUTPUT_MODELS_DIR", "web/models")
DRIVE_CKPT_DIR = os.environ.get("CKPT_DIR", "checkpoints")
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

fp32_drive = os.path.join(OUTPUT_MODELS_DIR, "gtcrn48k_fp32.onnx")
int8_drive = os.path.join(OUTPUT_MODELS_DIR, "gtcrn48k_int8.onnx")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

t0 = time.time()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
SR, N_FFT, HOP, N_FREQS = 48000, 960, 480, 481
SEG_LEN = SR * 2 if use_gpu else SR       # 2.0s segments on T4 GPU, 1.0s on CPU
EPOCHS = 25 if use_gpu else 15
BATCH_SIZE = 16
HIDDEN = 96 if use_gpu else 64
print(f"  Training device: {device} | epochs={EPOCHS} | seg={SEG_LEN/SR:.1f}s | hidden={HIDDEN}")

# ==========================================================================
# DUAL 48 KHZ REAL DATASET LOADER:
#   1. VoiceBank-DEMAND 48 kHz (authenticated via HF_TOKEN from Step 2)
#   2. Microsoft DNS Challenge 5 (48 kHz real environmental noise shard)
# ==========================================================================
import re as _re, tarfile as _tarfile, soundfile as _sf
from scipy.signal import resample_poly
import math as _math
from datasets import load_dataset

clean_clips = []
dns_noise_clips = []
target_clips = 450 if use_gpu else 150

def _to_48k(arr, sr_in):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    sr_in = int(sr_in)
    if sr_in != SR:
        g = _math.gcd(SR, sr_in)
        arr = resample_poly(arr, SR // g, sr_in // g).astype(np.float32)
    return arr

# 1. VoiceBank-DEMAND 48 kHz (native 48 kHz clean human speech)
print("  Loading VoiceBank-DEMAND 48 kHz via authenticated HF_TOKEN...")
ds_vb = load_dataset(
    "yairamr/voicebank-demand-fingerprint-48k",
    split="train",
    streaming=True,
    token=HF_TOKEN
)
def _extract_audio(aud):
    if aud is None:
        return None, SR
    # datasets v4+ torchcodec AudioDecoder
    if hasattr(aud, "get_all_samples"):
        s = aud.get_all_samples()
        data = s.data.cpu().numpy() if hasattr(s.data, "cpu") else np.asarray(s.data)
        if data.ndim > 1:
            data = data.mean(axis=0)  # [channels, samples] -> mono
        return data.astype(np.float32), int(getattr(s, "sample_rate", SR))
    # classic datasets dict
    if isinstance(aud, dict) and "array" in aud:
        return np.asarray(aud["array"], dtype=np.float32), int(aud.get("sampling_rate", SR))
    return None, SR

for item in ds_vb:
    aud = item.get("clean") or item.get("clean_audio") or item.get("audio")
    raw_arr, raw_sr = _extract_audio(aud)
    if raw_arr is not None:
        arr = _to_48k(raw_arr, raw_sr)
        if len(arr) >= SEG_LEN:
            clean_clips.append(arr)
    if len(clean_clips) >= target_clips:
        break
assert len(clean_clips) >= 20, "Failed to load VoiceBank-DEMAND 48 kHz clips!"
print(f"  [VOICEBANK 48K OK] Loaded {len(clean_clips)} native 48 kHz clean speech clips.")

# 2. Microsoft DNS Challenge 5 (official 48 kHz full-band environmental noise clips)
print("  Fetching real 48 kHz noise clips from Microsoft DNS Challenge 5...")
try:
    dns_script_url = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/download-dns-challenge-5-noise-ir.sh"
    dns_sh = urllib.request.urlopen(dns_script_url, timeout=20).read().decode("utf-8", errors="ignore")
    m_az = _re.search(r'AZURE_URL="([^"]+)"', dns_sh)
    azure_base = m_az.group(1).rstrip("/") if m_az else "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
    noise_blobs = _re.findall(r'noise_fullband/[^\s"\']+\.tar\.bz2', dns_sh)
    if noise_blobs:
        blob_url = f"{azure_base}/{noise_blobs[0]}"
        print(f"  Streaming 48 kHz noise shard from Microsoft DNS5: {noise_blobs[0]} ...")
        req = urllib.request.urlopen(blob_url, timeout=30)
        # Stream directly from HTTP tar.bz2 without downloading the entire multi-GB archive!
        with _tarfile.open(fileobj=req, mode="r|bz2") as tar:
            for member in tar:
                if member.isfile() and member.name.lower().endswith(".wav"):
                    fobj = tar.extractfile(member)
                    if fobj is not None:
                        try:
                            import io as _io
                            wav_arr, wav_sr = _sf.read(_io.BytesIO(fobj.read()), dtype="float32")
                            wav_48k = _to_48k(wav_arr, wav_sr)
                            if len(wav_48k) >= SEG_LEN:
                                dns_noise_clips.append(wav_48k)
                        except Exception:
                            pass
                if len(dns_noise_clips) >= 120:
                    break
        req.close()
    print(f"  [MICROSOFT DNS5 48K OK] Streamed {len(dns_noise_clips)} real 48 kHz environmental noise clips.")
except Exception as e_dns:
    print(f"  [WARN] Microsoft DNS5 stream interrupted ({e_dns}); using {len(dns_noise_clips)} DNS5 clips + Indian household noise.")

def synth_speech(n=SEG_LEN):
    t = np.arange(n, dtype=np.float32) / SR
    f0 = random.uniform(100.0, 230.0) + 18.0 * np.sin(2 * np.pi * random.uniform(1.5, 4.0) * t)
    phase = np.cumsum(2 * np.pi * f0 / SR)
    sig = sum((1.0 / h) * np.sin(h * phase) for h in range(1, 14))
    env = 0.5 * (1.0 + np.sin(2 * np.pi * random.uniform(2.0, 5.5) * t + random.uniform(0, 6.28)))
    sig = (sig * (env ** 1.5)).astype(np.float32)
    return sig / (np.max(np.abs(sig)) + 1e-6) * random.uniform(0.35, 0.85)

def add_indian_household_noise(clean):
    n = len(clean)
    t = np.arange(n, dtype=np.float32) / SR
    # 50/100/150 Hz AC mains hum + pink ceiling fan turbulence + broadband noise
    hum = (0.6 * np.sin(2*np.pi*50*t + random.uniform(0, 6.28))
         + 0.3 * np.sin(2*np.pi*100*t + random.uniform(0, 6.28))
         + 0.15 * np.sin(2*np.pi*150*t + random.uniform(0, 6.28)))
    white = np.random.randn(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(np.maximum(freqs, 20.0)), n=n).astype(np.float32)
    pink = pink / (np.std(pink) + 1e-6)
    if dns_noise_clips and random.random() < 0.7:
        dn = random.choice(dns_noise_clips)
        st = random.randint(0, max(0, len(dn) - n))
        real_n = dn[st:st+n].copy()
        real_n = real_n / (np.std(real_n) + 1e-6)
        noise = 0.65 * real_n + 0.20 * pink + 0.15 * hum
    else:
        noise = 0.55 * pink + 0.35 * hum + 0.10 * white
    snr_db = random.uniform(-2.0, 15.0)
    scale = (np.sqrt(np.mean(clean**2) + 1e-8)) / (np.sqrt(np.mean(noise**2) + 1e-8) * (10 ** (snr_db / 20.0)))
    return np.clip(clean + noise * scale, -0.99, 0.99).astype(np.float32)

class Speech48kDataset(Dataset):
    def __init__(self, clips, n_samples):
        self.clips = clips
        self.n_samples = n_samples
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        if self.clips:
            c = self.clips[idx % len(self.clips)]
            s = random.randint(0, max(0, len(c) - SEG_LEN))
            clean = c[s:s+SEG_LEN].copy()
        else:
            clean = synth_speech(SEG_LEN)
        noisy = add_indian_household_noise(clean)
        return torch.from_numpy(noisy), torch.from_numpy(clean)

class GTCRN48kSpectral(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2, 24, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(24), nn.PReLU(),
            nn.Conv2d(24, 48, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(48), nn.PReLU(),
            nn.Conv2d(48, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(64), nn.PReLU(),
        )
        self.proj_in = nn.Linear(64 * 61, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        self.proj_out = nn.Linear(hidden, 64 * 61)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 48, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(48), nn.PReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(24), nn.PReLU(),
            nn.ConvTranspose2d(24, 2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
        )

    def forward(self, spec_ri):
        x = spec_ri.permute(0, 3, 1, 2)
        h = self.enc(x)
        B, C, T, Fb = h.shape
        g = self.proj_in(h.permute(0, 2, 1, 3).reshape(B, T, C * Fb))
        g, _ = self.gru(g)
        h_out = self.proj_out(g).reshape(B, T, C, Fb).permute(0, 2, 1, 3) + h
        mask = torch.tanh(self.dec(h_out))
        m_re, m_im = mask[:, 0], mask[:, 1]
        s_re, s_im = x[:, 0], x[:, 1]
        return torch.stack([s_re * m_re - s_im * m_im, s_re * m_im + s_im * m_re], dim=-1)

torch.manual_seed(42)
model = GTCRN48kSpectral(hidden=HIDDEN).to(device)
print(f"  GTCRN48kSpectral params: {sum(p.numel() for p in model.parameters()):,}")

win = torch.sqrt(torch.hann_window(N_FFT, device=device))
def to_spec(wav):
    stft = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, window=win, return_complex=True)
    return torch.view_as_real(stft).permute(0, 2, 1, 3)

n_train = 512 if use_gpu else 96
loader = DataLoader(Speech48kDataset(clean_clips, n_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best_loss = 1e9
ckpt_path = os.path.join(DRIVE_CKPT_DIR, "gtcrn48k_best.pt")

for ep in range(1, EPOCHS + 1):
    model.train()
    tot = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        s_noisy = to_spec(noisy)
        s_clean = to_spec(clean)
        pred = model(s_noisy)
        p_mag = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2 + 1e-8)
        c_mag = torch.sqrt(s_clean[..., 0]**2 + s_clean[..., 1]**2 + 1e-8)
        loss = F.l1_loss(p_mag**0.3, c_mag**0.3) + 0.5 * F.l1_loss(pred, s_clean)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        tot += float(loss.item())
    sched.step()
    avg = tot / len(loader)
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), ckpt_path)
    if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
        print(f"  Epoch {ep:02d}/{EPOCHS} | loss={avg:.4f} | best={best_loss:.4f} | elapsed={time.time()-t0:.1f}s")

model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.eval().cpu()
dummy = torch.randn(1, 32, N_FREQS, 2, dtype=torch.float32)
torch.onnx.export(
    model, dummy, fp32_drive,
    input_names=["spec_ri"], output_names=["enh_ri"],
    dynamic_axes={"spec_ri": {1: "num_frames"}, "enh_ri": {1: "num_frames"}},
    opset_version=17, dynamo=False
)
quantize_dynamic(fp32_drive, int8_drive, weight_type=QuantType.QUInt8)

for p in [fp32_drive, int8_drive]:
    sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    y = sess.run(None, {"spec_ri": np.random.randn(1, 40, 481, 2).astype(np.float32)})[0]
    assert y.shape == (1, 40, 481, 2) and np.isfinite(y).all()
    print(f"  [SAVED & VERIFIED IN DRIVE] {os.path.basename(p)} ({os.path.getsize(p):,} bytes)")
