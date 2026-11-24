# NEON — Browser Noise-Free Mic

**NEON** is a real-time, privacy-first speech enhancement application powered by a strictly causal **GTCRN (Grouped Temporal Convolutional Recurrent Network)** running directly in the browser via **ONNX Runtime Web (WebGPU / WASM)** and **AudioWorklet**, backed by a lightweight **FastAPI** server with COOP/COEP isolation, server-side ONNX denoising, and anonymized **SQLite + JSONL** telemetry.

---

## Architecture Overview

```text
Microphone / Demo WAV (48 kHz)
   │
   ▼
AudioWorklet / Polyphase Resampler (48 kHz -> 16 kHz, 256-sample / 16 ms hop)
   │
   ▼
STFT (512-pt FFT, sqrt-Hann window, 257 freq bins)
   │
   ▼
GTCRN Streaming ONNX Model (WebGPU / WASM INT8 or FP32, < 4 ms/frame)
   │  (Recurrent & Causal Conv Caches carried across 16 ms hops)
   ▼
iSTFT Overlap-Add + Dry/Wet Mixer -> Speakers / VB-Cable Virtual Mic
   │
   ├──► Live Dual Spectrograms (#canvas-noisy, #canvas-denoised)
   ├──► Telemetry Cards (Latency, RTF, Est. SNR Gain, JS Heap)
   └──► FastAPI Backend (/api/denoise, /api/telemetry, /api/feedback, /api/metrics-dashboard)
           └──► monitoring/telemetry.db (SQLite) + monitoring/telemetry.jsonl
```

---

## 9-Point Implementation Summary

1. **Strictly Causal GTCRN Speech Enhancement Model**: Compact grouped-convolution + DPGRNN architecture operating on 16 kHz STFT frames (512-pt FFT, 256-sample / 16 ms hop) with zero lookahead for real-time communication.
2. **Stateful 1-Frame Streaming Export**: Exports both batch (`exports/gtcrn_fp32.onnx`) and cache-carrying streaming (`exports/gtcrn_stream_fp32.onnx`) ONNX graphs with bit-exact causality (`0.0` diff on `t < 20` when `t >= 20` is perturbed).
3. **INT8 Quantization for Edge Deployment**: Post-training quantized `exports/gtcrn_int8.onnx` delivers minimal footprint and sub-4 ms CPU/WASM frame latency with bounded numerical error (`< 0.05` mean abs diff).
4. **Browser AudioWorklet & WebGPU/WASM Runtime**: Zero-copy ring-buffered browser pipeline executing ONNX Runtime Web with `SharedArrayBuffer` enabled via strict COOP/COEP headers.
5. **Interactive Diagnostics & Indian Noise Demo**: Includes a built-in **Fan + 50 Hz Hum Demo** (`#btn-demo-indian`), real-time noisy vs. denoised spectrogram canvases, A/B/C listening modes, and live telemetry cards.
6. **FastAPI Backend with COOP/COEP Middleware (`backend/main.py`)**: Serves static web assets from `web/` alongside `/api/health`, `/api/denoise`, `/api/telemetry`, `/api/feedback`, and `/api/metrics-dashboard` with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` on every response.
7. **SQLite + JSONL Telemetry & Opt-In Clip Store (`backend/telemetry_store.py`)**: Persists anonymized session metrics (`telemetry_events`) and quality ratings (`feedback_events`) to `monitoring/telemetry.db` and `monitoring/telemetry.jsonl`, saving audio clips to `monitoring/opt_in_clips/<uuid>.wav` **only** when users explicitly opt in.
8. **6-Test Numerical, Causality & Latency QA Suite (`tests/test_suite.py`)**: Automated Pytest verification of PyTorch/ONNX parity (`< 1e-4`), 100-frame streaming vs. batch equivalence (`< 1e-4`), zero/clipping stability, 48→16 kHz pitch/duration conservation, 1,000-frame real-time latency & memory stability, and strict causality.
9. **Playwright Headless E2E & One-Click Deployment**: Automated Chromium UI verification (`tests/test_ui_playwright.py`), Windows PowerShell one-command launcher (`run_app.ps1`), and static edge deployment config (`vercel.json`).

---

## VB-Cable Windows Setup Guide (Use NEON as a Virtual Mic)

Route NEON's denoised browser output directly into Zoom, Microsoft Teams, Google Meet, or Discord on Windows:

1. **Install VB-Cable**:
   - Download and install **VB-CABLE Virtual Audio Device** from VB-Audio, then reboot Windows if prompted.
2. **Verify Sample Rate Alignment**:
   - Open **Windows Sound Settings → More sound settings → Playback & Recording**.
   - Set both **CABLE Input** (Playback) and **CABLE Output** (Recording) to **16-bit / 48000 Hz** (or 16000 Hz) to avoid OS resampling artifacts.
3. **Route Browser Audio to VB-Cable**:
   - Launch NEON in Chrome/Edge (`http://localhost:8000`).
   - Open **Windows Settings → System → Sound → Volume mixer**.
   - Locate your browser (Chrome/Edge) and set its **Output device** to **CABLE Input (VB-Audio Virtual Cable)**.
   - In NEON, select your physical microphone as input and enable live monitoring / Mode B (Denoised).
4. **Select Virtual Mic in Your Call App**:
   - In Zoom, Teams, Meet, or Discord, set **Microphone** to **CABLE Output (VB-Audio Virtual Cable)**.
   - Disable the call app's built-in aggressive noise suppression if you want pure NEON output without double-processing.

---

## PowerShell Runbook (Windows)

### One-Command Launch
```powershell
.\run_app.ps1
```
Optional flags:
```powershell
.\run_app.ps1 -Port 8080 -SkipInstall
```

### Manual Server Launch
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Running the QA Suites
```powershell
# 1. Core 6-test Pytest suite (numerical parity, streaming, causality, latency, resampling)
pytest -v tests/test_suite.py

# 2. Playwright headless Chromium E2E test (install Chromium once via `playwright install chromium`)
pytest -v tests/test_ui_playwright.py
```

---

## Vercel Deployment Instructions

NEON's browser-only static frontend (`web/`) can be deployed directly to Vercel with COOP/COEP headers pre-configured in `vercel.json`:

1. **Ensure ONNX Models Are in `web/models/`**:
   - Copy `exports/gtcrn_int8.onnx` (and optionally `exports/gtcrn_stream_fp32.onnx`) into `web/models/`.
2. **Deploy via Vercel CLI**:
   ```bash
   npm i -g vercel
   vercel --prod
   ```
3. **Deploy via Git Integration**:
   - Import the repository in the Vercel Dashboard.
   - Vercel automatically reads `vercel.json`, sets `outputDirectory` to `web`, and attaches:
     - `Cross-Origin-Opener-Policy: same-origin`
     - `Cross-Origin-Embedder-Policy: require-corp`
     - `Cross-Origin-Resource-Policy: same-origin`
4. **Verify Isolation in Production**:
   - Open DevTools Console on your deployed URL and verify `window.crossOriginIsolated === true`.
