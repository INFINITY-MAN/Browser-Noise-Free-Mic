<div align="center">

# Browser Noise Free Mic

**Privacy-first 48 kHz neural noise cancellation that runs 100% locally inside Chrome or Edge.**  
*Suppresses ceiling fans, 50 Hz electrical hum, traffic, and background room noise for recordings and live calls (Zoom, Google Meet, Discord, OBS) — with zero backend servers, zero subscriptions, and zero cloud uploads.*

**[Launch Live App](https://browser-noise-free-mic-web.vercel.app/input.html)** · [Features](#key-features) · [Model & Benchmarks](#model--verified-benchmarks) · [Training Notebook](training/GTCRN48kSpectral_48k_Colab_Training.ipynb) · [Live Virtual Mic Setup](#live-virtual-mic-setup-zoom--meet--discord--obs) · [Limitations](#honest-limitations)

</div>

---

## Try It Live

**[Open Browser Noise Free Mic](https://browser-noise-free-mic-web.vercel.app/input.html)** — runs directly in desktop Chrome or Edge (113+) on Windows 11. No account or installation is required for recording or file cleanup.

---

## Why I Built This

Most people joining online classes, interviews, or team calls from home deal with steady ceiling-fan noise, 50 Hz AC mains hum from ungrounded chargers, keyboard clicks, and street traffic. Existing neural tools either require a paid subscription (Krisp), lock you to an RTX GPU (NVIDIA Broadcast), upload private voice recordings to remote servers, or downsample audio to 16 kHz.

**Browser Noise Free Mic** runs 48 kHz full-band neural speech enhancement directly inside a browser tab using **ONNX Runtime Web** (WebGPU FP32 when a GPU is available, automatic WebAssembly SIMD fallback on CPU) and **AudioWorklet**. All audio stays on the user's own machine.

---

## Key Features

| Feature | Implementation |
| :--- | :--- |
| **Dual 48 kHz In-Browser ONNX Models** | Primary custom **GTCRN48kSpectral** (`web/models/gtcrn48k_fp32.onnx`, 908,663 parameters) and secondary **DeepFilterNet3** (`web/models/df3/*.onnx`). No silent classical fallbacks — missing models report an explicit error. |
| **Offline File & Recording Studio** | Record directly from the mic or drop `.aac`, `.m4a`, `.wav`, `.mp3`, or `.ogg` files (`input.html`), A/B compare with **Hold O**, switch between 4 presets, inspect spectrograms, and export 48 kHz 16-bit WAV (or AAC via WebCodecs). |
| **Live Virtual Mic (Continuous Mode)** | Captures 48 kHz mic audio in `streamWorklet.js` (480-sample hops, 64-hop ring buffer), processes 160 ms windows / 80 ms stride, and routes cleaned output via `HTMLMediaElement.setSinkId()` into **VB-Audio Virtual Cable**. |
| **Echo Guard & 1-Click Cable Installer** | Mutes self-monitoring automatically when routing to regular speakers/headphones so you do not hear delayed echo, and bundles `web/install_virtual_cable.bat` for Windows VB-Cable setup. |
| **Stats for Nerds Drawer (`N`)** | Displays the active hardware provider (`GPU: WebGPU (FP32)` vs. `CPU: WASM SIMD`), active model, real measured latency, RTF, and dual spectrograms. |
| **Installable Static PWA** | Hosted statically on Vercel (`web/`) with `COOP`, `COEP`, and `CORP` headers (`vercel.json`) and offline model caching (`sw.js`). |

---

## Model & Verified Benchmarks

All training, ONNX export, and primary evaluation ran on **Google Colab (NVIDIA Tesla T4 GPU, 16 GB VRAM)**. Reproducible training code is included in [`training/train_gtcrn48k.py`](training/train_gtcrn48k.py) and [`training/GTCRN48kSpectral_48k_Colab_Training.ipynb`](training/GTCRN48kSpectral_48k_Colab_Training.ipynb).

| Metric | Verified Value |
| :--- | :--- |
| **Primary Model** | `GTCRN48kSpectral` — **908,663** trainable parameters (`960`-pt periodic sqrt-Hann STFT, `480`-sample / 10 ms hop, `481` bins, `[1, T, 481, 2]` Complex Ratio Mask) |
| **Shipped Model Sizes** | `gtcrn48k_fp32.onnx`: **3,644,893 B (3.64 MB)** · `gtcrn48k_int8.onnx`: **1,342,002 B (1.34 MB)** |
| **Secondary Model** | DeepFilterNet3 ONNX — `enc.onnx` (1.95 MB), `erb_dec.onnx` (3.29 MB), `df_dec.onnx` (3.34 MB) |
| **Training Dataset (48 kHz)** | **450** native 48 kHz clean speech clips ([VoiceBank-DEMAND 48 kHz](https://huggingface.co/datasets/yairamr/voicebank-demand-fingerprint-48k)) + **120** real 48 kHz environmental noise clips ([Microsoft DNS Challenge 5](https://github.com/microsoft/DNS-Challenge)) + synthetic pink ceiling fan & 50/100/150 Hz AC mains hum |
| **Training Run (Colab T4)** | 25 epochs, 2.0 s segments, AdamW, **282.1 s** total. Loss: Ep 1 `0.2227` -> Ep 5 `0.1354` -> Ep 10 `0.1231` -> Ep 15 `0.1174` -> Ep 20 (best) **`0.1081`** -> Ep 25 `0.1120` |
| **0 dB Fan + Hum Benchmark** | **GTCRN48kSpectral FP32:** **`+7.42 dB` SNR gain** and **`-21.31 dB` noise-only floor reduction** · **DeepFilterNet3 ERB stage:** `+2.48 dB` SNR gain |
| **Engineering Checks** | JS 960-pt STFT/iSTFT round-trip error: **`1.192e-7`** · Static scan of `web/`: **`0`** `innerHTML` sinks, **`0`** `/api/` calls |

---

## Live Virtual Mic Setup (Zoom / Meet / Discord / OBS)

1. Install **VB-Audio Virtual Cable** (manually from [vb-audio.com/Cable](https://vb-audio.com/Cable/) or using `web/install_virtual_cable.bat` from the Live Virtual Mic screen).
2. Open **[Browser Noise Free Mic](https://browser-noise-free-mic-web.vercel.app/input.html)** in desktop Chrome or Edge and select **Live Virtual Mic**.
3. Pick your physical microphone as input and **CABLE Input (VB-Audio Virtual Cable)** as output. Echo Guard unmutes automatically once a virtual cable is selected.
4. In Zoom, Google Meet, Discord, or OBS, select **CABLE Output (VB-Audio Virtual Cable)** as your microphone.

*(On dual-GPU Windows laptops, set Chrome/Edge to **High Performance** in Windows Graphics Settings so WebGPU uses the discrete GPU instead of integrated graphics.)*

---

## Repository Structure

```text
.
├── web/                              # 100% static browser PWA (served by Vercel)
│   ├── input.html                    # Capture, file upload, demo & live mic entry
│   ├── processing.html               # Offline neural progress & live mic handshake
│   ├── output.html                   # File Studio (A/B, presets, export) & Live Virtual Mic console
│   ├── audioEngine.js                # ONNX Runtime Web loader, probeHardware(), 960-pt STFT/iSTFT, runners
│   ├── streamWorklet.js              # AudioWorkletProcessor (480-sample hops, 64-hop ring buffer)
│   ├── masteringChain.js             # 4 mastering presets (HPF, EQ, compressor, -16 LUFS / -1 dBFS)
│   ├── audioCodec.js                 # 16-bit WAV encoder/decoder, WebCodecs AAC, spectrogram & room snapshot
│   ├── outputController.js           # Studio playback, export, Echo Guard & setSinkId() live routing
│   ├── appStore.js                   # IndexedDB (BNFM_DB) & preferences schema
│   ├── uiCommon.js                   # Shared UI, theme toggle, toasts & local storage helpers
│   ├── sw.js & manifest.json         # Service Worker (cache-first .onnx) & PWA manifest
│   ├── install_virtual_cable.bat     # Self-elevating Windows VB-Audio Virtual Cable installer
│   └── models/                       # Shipped 48 kHz ONNX weights (gtcrn48k & df3)
├── training/
│   ├── train_gtcrn48k.py             # Standalone 48 kHz PyTorch training & ONNX export script
│   ├── GTCRN48kSpectral_48k_Colab_Training.ipynb  # Reproducible Colab T4 notebook
│   └── TRAINING_DATA_AND_LOGS.md     # Dataset specs & verified Colab T4 logs
├── Project/                          # Final Project Synopsis & Report
├── vercel.json                       # Static hosting & COOP/COEP/CORP isolation headers
└── LICENSE                           # MIT License
```

---

## Honest Limitations

- **Live Chunked Latency (~160–250 ms):** The exported ONNX graph processes `[1, T, 481, 2]` chunks without exposed GRU recurrent state, so live mode uses 160 ms windows with an 80 ms stride. Suitable for calls, but not for zero-latency self-monitoring.
- **Dataset Scope:** Trained on 450 clean clips + 120 real DNS5 noise clips + synthetic fan/hum inside a single Colab T4 session. Handles steady household noise well; impulsive transients, nearby speakers, and heavy reverberation remain harder.
- **Browser & Driver Scope:** `setSinkId()` live routing requires desktop Chrome or Edge plus VB-Audio Virtual Cable on Windows.

---

## Author

**Priyanshu Aggarwal** ([INFINITY-MAN](https://github.com/INFINITY-MAN))  
Licensed under the [MIT License](LICENSE).
