<div align="center">

# 🎙️ Browser Noise Free Mic

**Real-time, privacy-first 48 kHz AI noise cancellation that runs 100% locally in your browser.**  
*Kill ceiling fans, 50 Hz electrical hum, traffic, and keyboard noise on Zoom, Google Meet, Discord, and OBS — with zero subscriptions and zero cloud uploads.*

[![Live App](https://img.shields.io/badge/Live%20App-Try%20in%20Browser-FF5A00?style=flat-square&logo=vercel)](https://browser-noise-free-mic-web.vercel.app/input.html)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FINFINITY-MAN%2FBrowser-Noise-Free-Mic&root-directory=web)
[![Releases](https://img.shields.io/github/v/release/INFINITY-MAN/Browser-Noise-Free-Mic?style=flat-square&color=FF5A00)](https://github.com/INFINITY-MAN/Browser-Noise-Free-Mic/releases)
[![WebGPU / WASM](https://img.shields.io/badge/Engine-WebGPU%20%2F%20WASM%2048kHz-0A0A0A?style=flat-square)](https://github.com/INFINITY-MAN/Browser-Noise-Free-Mic)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local%20In--Browser-FF5A00?style=flat-square)](https://github.com/INFINITY-MAN/Browser-Noise-Free-Mic)

[🎙️ Launch Live App](https://browser-noise-free-mic-web.vercel.app/input.html) · [Features](#-key-features) · [Virtual Mic Setup](#-use-as-a-live-virtual-mic-zoom--meet--discord--obs) · [Architecture](#-architecture) · [Keyboard Shortcuts](#%EF%B8%8F-keyboard-shortcuts)

</div>

---

## 🌐 Try It Live Right Now

👉 **[Open Browser Noise Free Mic (https://browser-noise-free-mic-web.vercel.app/input.html)](https://browser-noise-free-mic-web.vercel.app/input.html)** — works immediately in Chrome & Edge with zero install.

---

## ✨ Why Browser Noise Free Mic?

Most noise-cancellation tools require heavy desktop installs, paid monthly subscriptions, or send your voice to remote servers. **Browser Noise Free Mic** runs a strictly causal neural speech enhancement pipeline directly on your GPU/CPU inside Chrome or Edge using **WebGPU / WASM** and **AudioWorklet** — no account, no cloud, no lag.

---

## 🚀 Key Features

| Feature | What It Does |
| :--- | :--- |
| 🎧 **Live Virtual Mic (Continuous Mode)** | Real-time microphone denoising with **AudioWorklet** + `setSinkId` routing straight into **VB-Audio Virtual Cable** for Zoom, Meet, Discord, and OBS. |
| 🛡️ **Built-in Echo Guard** | Automatically detects whether you're routing to a Virtual Cable or regular headphones/speakers — mutes self-monitoring on headphones by default so you never hear annoying voice echo, with a 1-click toggle when you want to test your voice. |
| ⚡ **1-Click Virtual Cable Installer** | Built-in Windows `.bat` & PowerShell installer right inside Continuous Mode that fetches and installs the official VB-Audio Virtual Cable driver, plus a **Refresh Devices** button. |
| 🎛️ **4 Studio Mastering Presets** | Instant one-click presets: **Podcast Studio Cut**, **Natural Voice**, **Heavy Fan / Traffic**, and **Raw Neural**, plus **Trim Silence** and **2s Room Snapshot** noise-floor lock. |
| 🎙️ **Record & File Studio** | Record directly from your mic or drop `.wav`, `.mp3`, `.m4a`, `.aac`, or `.ogg` files, compare **Original vs. Studio Cut** (`Hold O`), and export lossless **48 kHz WAV** (or **AAC 160k `.m4a`** with optional backend). |
| 📊 **Stats for Nerds Drawer** | Collapsible telemetry drawer (`N`) with dual 20 kHz spectrograms, live hardware acceleration badge (WebGPU/WASM), latency, RTF, SNR gain, Dry/Wet blend, and engine selector. |
| 🌓 **Clean Studio UI & Installable PWA** | Crisp 3-color light/dark interface (zero bloat) that installs as a standalone desktop app from Chrome or Edge. |

---

## 🎧 Use as a Live Virtual Mic (Zoom / Meet / Discord / OBS)

1. Open **Browser Noise Free Mic** in Chrome or Edge and click **Live Virtual Mic**.
2. If you don't have a virtual cable installed yet, click **1-Click Install Virtual Cable (Windows)** right inside the app (or click **Copy PowerShell Cmd**), run it as Administrator, and click **Refresh Devices** *(reboot Windows once if `CABLE Input` doesn't show up immediately)*.
3. Select your **physical microphone** as Input and **`CABLE Input (VB-Audio Virtual Cable)`** as Output.
4. In **Zoom, Google Meet, Discord, or OBS**, set your Microphone input to **`CABLE Output (VB-Audio Virtual Cable)`**.
5. *(Tip)* Turn off your call app's built-in aggressive noise suppression so you get clean single-pass voice without double-processing.

---

## ⚡ Quick Start

### Option 1: One-Click Deploy on Vercel (Recommended — $0, Browser-Only)
Click the **Deploy with Vercel** button at the top, or import this repo manually in Vercel with:
- **Application Preset**: `Other`
- **Root Directory**: `web`

`web/vercel.json` automatically configures the required `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers for `SharedArrayBuffer` / WebGPU / WASM execution.

### Option 2: Run Locally on Windows (With Optional Python Backend)
```powershell
git clone https://github.com/INFINITY-MAN/Browser-Noise-Free-Mic.git
cd Browser-Noise-Free-Mic
.\run_app.ps1
```
Or run manually with Python 3.10+:
```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Then open `http://localhost:8000` in Chrome or Edge.

---

## 🧩 Browser-Only vs. Optional FastAPI Backend

The core app — including **Live Virtual Mic** — runs **100% in the browser** and does **not** require a backend.

| Capability | Vercel Static (`web/` only) | Local / Docker + FastAPI (`backend/main.py`) |
| :--- | :---: | :---: |
| Live Virtual Mic + Echo Guard + VB-Cable | ✅ 100% Local | ✅ 100% Local |
| Record Voice & Drop File Denoising | ✅ In-browser GTCRN (WebGPU/WASM) | ✅ + Server-side DeepFilterNet3 48 kHz |
| Studio Mastering Chain & Presets | ✅ Full 48 kHz Web Audio | ✅ Full 48 kHz Web Audio |
| Export Format | ✅ Lossless 48 kHz `.wav` | ✅ `.wav` + FFmpeg 160k `.m4a` |
| Telemetry & History | ✅ Local IndexedDB (`BNFM_DB`) | ✅ + SQLite `/api/metrics-dashboard` |

---

## 🏗️ Architecture

```text
Microphone / Audio File (48 kHz)
       │
       ▼
┌─────────────────────────────────────────────┐
│  Browser Frontend (web/)                    │
│  ├── input.html       (Capture & Intake)    │
│  ├── processing.html  (4-Stage Engine Prep) │
│  ├── output.html      (Review / Live Mic)   │
│  ├── audioEngine.js   (WebGPU / WASM ONNX)  │
│  ├── streamWorklet.js (AudioWorklet Ring)   │
│  ├── masteringChain.js (EQ / Comp / Limit)  │
│  └── appStore.js      (IndexedDB + Prefs)   │
└─────────────────────────────────────────────┘
       │                           │
       ▼                           ▼ (Optional)
VB-Audio Virtual Cable       FastAPI Backend (backend/)
(Zoom / Meet / Discord)      (DeepFilterNet3 / FFmpeg / SQLite)
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `Space` | Play / Pause audio preview |
| `Hold O` | Momentary **Original** bypass (A/B compare) |
| `M` | Toggle **Live / Bypass** in Continuous Mode |
| `N` | Toggle **Stats for Nerds** telemetry drawer |

---

## 📁 Project Structure

```text
Browser-Noise-Free-Mic/
├── web/
│   ├── index.html                  # Entry redirect → input.html
│   ├── input.html                  # Phase 1: Record, Drop File, or Live Mic
│   ├── processing.html             # Phase 2: Engine warmup & processing
│   ├── output.html                 # Phase 3: Studio review, Live routing & Echo Guard
│   ├── audioEngine.js              # WebGPU / WASM ONNX & Web Audio controller
│   ├── streamWorklet.js            # 48 kHz AudioWorklet streaming processor
│   ├── masteringChain.js           # Studio EQ, compression, trim & presets
│   ├── appStore.js                 # Local IndexedDB & preferences store
│   ├── install_virtual_cable.bat   # 1-Click Windows VB-Audio Virtual Cable installer
│   ├── styles.css                  # Light/Dark 3-color Studio UI system
│   ├── manifest.json / sw.js       # Installable Desktop PWA files
│   └── vercel.json                 # COOP / COEP headers for Vercel static deploy
├── backend/                        # Optional FastAPI + DeepFilterNet3 + FFmpeg server
├── run_app.ps1                     # One-command Windows local launcher
├── requirements.txt                # Python dependencies for optional backend
└── vercel.json                     # Root Vercel static config
```

---

## ⭐ Support & Feedback

If **Browser Noise Free Mic** saved your calls from fan or background noise, drop a **⭐ Star** on the repo or open an [Issue](https://github.com/INFINITY-MAN/Browser-Noise-Free-Mic/issues) with ideas for `v2.1`!

---

<div align="center">

Built by **Priyanshu Aggarwal** ([INFINITY-MAN](https://github.com/INFINITY-MAN)) · **Browser Noise Free Mic**

</div>
