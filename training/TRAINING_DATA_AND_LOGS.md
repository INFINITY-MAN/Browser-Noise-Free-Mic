# Training Data, Model Specs & Verified Google Colab T4 Logs

Use these exact verified facts and logs for the Report (Chapter 7, Table 3.1, Chapter 8) and Synopsis.
**Do NOT reference any `aarch64` cloud VM or local laptop training — all training, ONNX export, and Python evaluation ran on Google Colab (NVIDIA Tesla T4 GPU, 16 GB VRAM).**

## 1. Datasets (Native 48 kHz)
- **Clean Speech (450 clips):** [VoiceBank-DEMAND 48 kHz](https://huggingface.co/datasets/yairamr/voicebank-demand-fingerprint-48k), streamed in native 48 kHz via Hugging Face `datasets` (`load_dataset(..., streaming=True)`), filtered to clips >= 2.0 s (`96,000` samples).
- **Real Environmental Noise (120 clips):** [Microsoft DNS Challenge 5](https://github.com/microsoft/DNS-Challenge) fullband archive (`noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2`), streamed directly over HTTP in non-seekable `tarfile.open(fileobj=req, mode="r|bz2")` mode with each WAV decoded via `soundfile.read(io.BytesIO(fobj.read()))`.
- **Synthetic Indian Household Noise:** Generated on the fly in `add_indian_household_noise()`:
  - 50 Hz AC mains hum + 100 Hz and 150 Hz harmonics (`0.6*sin(50Hz) + 0.3*sin(100Hz) + 0.15*sin(150Hz)`)
  - Pink-spectrum ceiling fan turbulence (`1/sqrt(max(f, 20 Hz))` shaped white noise)
  - 70% probability of blending real DNS5 noise (`0.65 * real_n + 0.20 * pink + 0.15 * hum`), 30% pure synthetic (`0.55 * pink + 0.35 * hum + 0.10 * white`), mixed at random SNRs from `-2.0 dB` to `+15.0 dB`.

## 2. Model & Training Run (Google Colab NVIDIA Tesla T4, 16 GB VRAM)
- **Architecture:** `GTCRN48kSpectral` (`training/train_gtcrn48k.py`) — **908,663** trainable parameters.
  - 2-D Conv Encoder (`2 -> 24 -> 48 -> 64`, kernel `(1,5)`, stride `(1,2)`, BatchNorm2d + PReLU)
  - Linear projection (`64 * 61 -> 96`) -> 2-layer `nn.GRU(96, 96, num_layers=2, batch_first=True)` -> Linear projection (`96 -> 64 * 61`) + residual skip connection
  - 2-D ConvTranspose Decoder (`64 -> 48 -> 24 -> 2`) + `torch.tanh` Complex Ratio Mask: `Yr = Sr*Mr - Si*Mi`, `Yi = Sr*Mi + Si*Mr`
- **STFT Setup:** 48,000 Hz mono, `N_FFT = 960` (20 ms), `HOP = 480` (10 ms), `481` frequency bins, periodic sqrt-Hann window (`torch.sqrt(torch.hann_window(960))`).
- **Loss & Optimizer:** Compressed magnitude L1 + complex L1 (`F.l1_loss(p_mag**0.3, c_mag**0.3) + 0.5 * F.l1_loss(pred, s_clean)`), `AdamW(lr=1.5e-3, weight_decay=1e-4)`, `CosineAnnealingLR`, gradient clip `3.0`, `BATCH_SIZE = 16`, `512` mixtures/epoch, `25` epochs.

## 3. Verbatim Colab T4 Console Logs
```text
[VOICEBANK 48K OK] Loaded 450 native 48 kHz clean speech clips.
Streaming 48 kHz noise shard from Microsoft DNS5: noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2 ...
[MICROSOFT DNS5 48K OK] Streamed 120 real 48 kHz environmental noise clips.
```

```text
Training device: cuda | epochs=25 | seg=2.0s | hidden=96
GTCRN48kSpectral params: 908,663
Epoch 01/25 | loss=0.2227 | best=0.2227 | elapsed=106.6s
Epoch 05/25 | loss=0.1354 | best=0.1354 | elapsed=135.4s
Epoch 10/25 | loss=0.1231 | best=0.1231 | elapsed=171.1s
Epoch 15/25 | loss=0.1174 | best=0.1163 | elapsed=208.1s
Epoch 20/25 | loss=0.1081 | best=0.1081 | elapsed=245.0s
Epoch 25/25 | loss=0.1120 | best=0.1081 | elapsed=282.1s
```

```text
[SAVED & VERIFIED IN DRIVE] gtcrn48k_fp32.onnx (3,644,893 bytes)
[SAVED & VERIFIED IN DRIVE] gtcrn48k_int8.onnx (1,342,002 bytes)
```

## 4. Verified Evaluation Benchmarks (Colab T4, 0 dB Fan + Hum)
- **GTCRN48kSpectral FP32 (`3.64 MB`):** `+7.42 dB` SNR gain (`0.00 dB -> 7.42 dB`), `-21.31 dB` noise-only floor attenuation.
- **DeepFilterNet3 ERB-only JS stage (`enc.onnx` 1.95 MB, `erb_dec.onnx` 3.29 MB, `df_dec.onnx` 3.34 MB):** `+2.48 dB` SNR gain.
- **JS 960-pt STFT/iSTFT round-trip identity error:** `1.192e-7`.
