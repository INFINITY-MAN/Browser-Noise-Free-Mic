"""Browser Noise Free Mic — 48 kHz FastAPI Backend (backend/main.py).

Features:
  - Cross-Origin Isolation middleware: COOP (`same-origin`), COEP (`require-corp`),
    CORP (`same-origin`) for SharedArrayBuffer / WASM / AudioWorklet support.
  - `GET  /api/health`             : Status, 48 kHz engines (`df3`, `gtcrn48k`), `ffmpeg`.
  - `POST /api/denoise`            : Dual-engine 48 kHz enhancement (`engine=df3|gtcrn48k`)
                                     with `X-BNFM-Infer-Ms`, `X-BNFM-RTF`, `X-BNFM-Engine`,
                                     and `X-BNFM-SNR-Gain-DB` response headers.
  - `POST /api/decode`             : Decodes `.aac`/`.m4a`/`.mp3`/`.ogg`/etc. via `ffmpeg`
                                     to clean 48 kHz 16-bit mono WAV.
  - `POST /api/encode-aac`         : Encodes 48 kHz WAV to 160 kbps AAC `.m4a` via
                                     `ffmpeg -c:a aac -b:a 160k -movflags +faststart`.
  - `POST /api/telemetry`          : Logs runtime telemetry via `backend.telemetry_store`.
  - `POST /api/feedback`           : Logs quality ratings + optional Loop B opt-in clips.
  - `GET  /api/metrics-dashboard`  : Aggregated telemetry and feedback dashboard metrics.
  - Static frontend serving for `web/` at `/` (defaulting `/` to `input.html`).
"""

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from backend.telemetry_store import TelemetryStore, VALID_RATINGS, get_store
except ImportError:
    from telemetry_store import TelemetryStore, VALID_RATINGS, get_store  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
MODELS_DIR = WEB_DIR / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
TARGET_SR = 48000

# ---------------------------------------------------------------------------
# Optional Engine & Tooling Discovery
# ---------------------------------------------------------------------------

FFMPEG_BIN = shutil.which("ffmpeg")

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from df.enhance import enhance as df_enhance, init_df  # type: ignore
    DF3_PKG_AVAILABLE = True
except Exception:
    df_enhance = None  # type: ignore
    init_df = None  # type: ignore
    DF3_PKG_AVAILABLE = False

try:
    import onnxruntime as ort  # type: ignore
    ORT_AVAILABLE = True
except Exception:
    ort = None  # type: ignore
    ORT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Audio I/O & FFmpeg Utilities (48 kHz 16-bit Mono WAV & 160 kbps AAC .m4a)
# ---------------------------------------------------------------------------

def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = TARGET_SR) -> bytes:
    """Encode a float32 [-1, 1] mono array into 16-bit PCM RIFF WAV bytes."""
    clipped = np.clip(np.asarray(pcm, dtype=np.float32).reshape(-1), -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())
    return buf.getvalue()


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or pcm.size == 0:
        return pcm.astype(np.float32, copy=False)
    dst_len = max(1, int(round(pcm.shape[0] * (dst_sr / float(src_sr)))))
    x_old = np.linspace(0.0, 1.0, num=pcm.shape[0], dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=dst_len, dtype=np.float64)
    return np.interp(x_new, x_old, pcm.astype(np.float64)).astype(np.float32)


def decode_wav_bytes_native(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode standard uncompressed PCM WAV bytes without external binaries."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32), sr


def decode_audio_via_ffmpeg(raw_bytes: bytes, suffix: str = ".bin") -> bytes:
    """Transcode any audio format to clean 48 kHz 16-bit mono WAV using ffmpeg."""
    if not FFMPEG_BIN:
        raise RuntimeError("ffmpeg is not installed on the server.")

    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.TemporaryDirectory(prefix="bnfm_dec_") as tmpdir:
        in_path = Path(tmpdir) / f"input{safe_suffix}"
        out_path = Path(tmpdir) / "output_48k.wav"
        in_path.write_bytes(raw_bytes)

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(in_path),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SR),
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.strip()}")
        return out_path.read_bytes()


def encode_aac_via_ffmpeg(wav_bytes: bytes) -> bytes:
    """Encode 48 kHz WAV bytes to 160 kbps AAC .m4a with faststart moov atom."""
    if not FFMPEG_BIN:
        raise RuntimeError("ffmpeg is not installed on the server.")

    with tempfile.TemporaryDirectory(prefix="bnfm_enc_") as tmpdir:
        in_path = Path(tmpdir) / "input_48k.wav"
        out_path = Path(tmpdir) / "output_48k.m4a"
        in_path.write_bytes(wav_bytes)

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(in_path),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SR),
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"ffmpeg AAC encode failed: {proc.stderr.strip()}")
        return out_path.read_bytes()


def decode_upload_to_48k_pcm(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Decode uploaded audio (WAV/AAC/M4A/MP3/OGG) to 48 kHz mono float32 PCM."""
    ext = Path(filename or "audio.wav").suffix.lower() or ".wav"
    if ext == ".wav":
        try:
            pcm, sr = decode_wav_bytes_native(raw_bytes)
            return _resample_linear(pcm, sr, TARGET_SR)
        except Exception:
            pass

    if FFMPEG_BIN:
        wav_48k = decode_audio_via_ffmpeg(raw_bytes, suffix=ext)
        pcm, sr = decode_wav_bytes_native(wav_48k)
        return _resample_linear(pcm, sr, TARGET_SR)

    raise HTTPException(
        status_code=415,
        detail=f"Cannot decode '{ext}' without ffmpeg installed on the server.",
    )


# ---------------------------------------------------------------------------
# Dual-Engine 48 kHz Inference Manager (df3 + gtcrn48k + Wiener fallback)
# ---------------------------------------------------------------------------

def _wiener_fallback_48k(pcm: np.ndarray) -> np.ndarray:
    """Fast numpy 48 kHz STFT Wiener spectral denoiser fallback."""
    n = pcm.shape[0]
    if n < 2048:
        return pcm.copy()

    n_fft = 2048
    hop = 512
    win = np.hanning(n_fft).astype(np.float32)

    pad = n_fft - ((n - n_fft) % hop) if n > n_fft else n_fft
    padded = np.pad(pcm, (0, pad), mode="constant")
    num_frames = 1 + (padded.shape[0] - n_fft) // hop

    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(num_frames, n_fft),
        strides=(padded.strides[0] * hop, padded.strides[0]),
    )
    spec = np.fft.rfft(frames * win[None, :], axis=1)
    mag = np.abs(spec)

    # Estimate noise floor from 15th percentile magnitude across time
    noise_mag = np.percentile(mag, 15, axis=0, keepdims=True)
    snr_post = (mag ** 2) / ((noise_mag ** 2) * 1.6 + 1e-10)
    gain = np.clip((snr_post - 1.0) / (snr_post + 1e-10), 0.08, 1.0)

    clean_frames = np.fft.irfft(spec * gain, n=n_fft, axis=1).astype(np.float32)
    out = np.zeros_like(padded)
    norm = np.zeros_like(padded)
    for idx in range(num_frames):
        s = idx * hop
        out[s : s + n_fft] += clean_frames[idx] * win
        norm[s : s + n_fft] += win ** 2

    out = np.where(norm > 1e-6, out / np.maximum(norm, 1e-6), padded)
    return np.clip(out[:n], -1.0, 1.0).astype(np.float32)


def _estimate_snr_gain_db(noisy: np.ndarray, enhanced: np.ndarray) -> float:
    """Estimate effective noise suppression gain in dB between noisy and enhanced."""
    n = min(noisy.shape[0], enhanced.shape[0])
    if n == 0:
        return 0.0
    residual = noisy[:n] - enhanced[:n]
    sig_pow = float(np.mean(enhanced[:n] ** 2)) + 1e-10
    res_pow = float(np.mean(residual ** 2)) + 1e-10
    raw_pow = float(np.mean(noisy[:n] ** 2)) + 1e-10
    gain_db = 10.0 * math.log10((sig_pow + res_pow) / max(sig_pow, 1e-10)) + 10.0 * math.log10(
        max(raw_pow, 1e-10) / max(res_pow, 1e-10)
    )
    return round(float(max(0.0, min(30.0, abs(gain_db)))), 2)


class DualEngineManager:
    """Thread-safe lazy loader for 48 kHz DeepFilterNet3 (df3) and GTCRN-DF (gtcrn48k)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._df3_model = None
        self._df3_state = None
        self._df3_loaded = False
        self._df3_error: Optional[str] = None

        self._gtcrn_session = None
        self._gtcrn_loaded = False
        self._gtcrn_path = self._find_gtcrn_artifact()

    def _find_gtcrn_artifact(self) -> Optional[Path]:
        candidates = [
            MODELS_DIR / "gtcrn_df_48k.onnx",
            CHECKPOINTS_DIR / "gtcrn_df_48k.onnx",
            CHECKPOINTS_DIR / "gtcrn_df_48k_best.pt",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _ensure_df3(self) -> bool:
        if self._df3_loaded:
            return True
        if not DF3_PKG_AVAILABLE or not TORCH_AVAILABLE:
            return False
        with self._lock:
            if self._df3_loaded:
                return True
            try:
                model, df_state, _ = init_df()
                self._df3_model = model
                self._df3_state = df_state
                self._df3_loaded = True
                return True
            except Exception as exc:
                self._df3_error = str(exc)
                return False

    def _ensure_gtcrn48k(self) -> bool:
        if self._gtcrn_loaded:
            return True
        self._gtcrn_path = self._find_gtcrn_artifact()
        if not self._gtcrn_path:
            return False
        with self._lock:
            if self._gtcrn_loaded:
                return True
            if self._gtcrn_path.suffix == ".onnx" and ORT_AVAILABLE:
                try:
                    self._gtcrn_session = ort.InferenceSession(
                        str(self._gtcrn_path),
                        providers=["CPUExecutionProvider"],
                    )
                    self._gtcrn_loaded = True
                    return True
                except Exception:
                    return False
        return False

    def health_info(self) -> Dict[str, Any]:
        self._gtcrn_path = self._find_gtcrn_artifact()
        return {
            "df3": {
                "available": bool(DF3_PKG_AVAILABLE and TORCH_AVAILABLE),
                "loaded": self._df3_loaded,
                "sample_rate": TARGET_SR,
                "error": self._df3_error,
            },
            "gtcrn48k": {
                "available": bool(self._gtcrn_path is not None),
                "loaded": self._gtcrn_loaded,
                "sample_rate": TARGET_SR,
                "artifact": str(self._gtcrn_path.name) if self._gtcrn_path else None,
            },
        }

    def denoise(self, pcm_48k: np.ndarray, requested_engine: str = "df3") -> Tuple[np.ndarray, str]:
        engine = (requested_engine or "df3").strip().lower()

        # Track A: 48 kHz DeepFilterNet3
        if engine == "df3" and self._ensure_df3():
            audio_t = torch.from_numpy(pcm_48k).unsqueeze(0).float()
            enhanced_t = df_enhance(self._df3_model, self._df3_state, audio_t)
            enhanced = enhanced_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return np.clip(enhanced, -1.0, 1.0), "df3"

        # Track B: Custom 48 kHz GTCRN-DF ONNX
        if engine == "gtcrn48k" and self._ensure_gtcrn48k() and self._gtcrn_session is not None:
            try:
                inp_name = self._gtcrn_session.get_inputs()[0].name
                out = self._gtcrn_session.run(None, {inp_name: pcm_48k[None, :].astype(np.float32)})[0]
                enhanced = np.asarray(out, dtype=np.float32).reshape(-1)
                return np.clip(enhanced[: pcm_48k.shape[0]], -1.0, 1.0), "gtcrn48k"
            except Exception:
                pass

        # Cross-fallback to DF3 if gtcrn48k artifact is not trained yet
        if engine == "gtcrn48k" and self._ensure_df3():
            audio_t = torch.from_numpy(pcm_48k).unsqueeze(0).float()
            enhanced_t = df_enhance(self._df3_model, self._df3_state, audio_t)
            enhanced = enhanced_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return np.clip(enhanced, -1.0, 1.0), "df3-fallback"

        # Guaranteed offline DSP fallback
        return _wiener_fallback_48k(pcm_48k), f"{engine}-wiener-fallback"


engine_manager = DualEngineManager()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TelemetryPayload(BaseModel):
    model_version: str = Field(default="df3_48k_v2")
    backend: str = Field(default="server_df3")
    mode: str = Field(default="upload")
    avg_infer_ms: float = Field(default=0.0, ge=0.0)
    rtf: float = Field(default=0.0, ge=0.0)
    est_snr_gain_db: float = Field(default=0.0)
    dry_wet_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    underruns: int = Field(default=0, ge=0)
    heap_mb: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# FastAPI Application & Cross-Origin Isolation Middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Browser Noise Free Mic 48 kHz Studio Audio Backend",
    version="2.0.0",
    description="48 kHz Dual-Engine Denoising (DeepFilterNet3 & GTCRN-DF), AAC Pipeline, and Telemetry API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-BNFM-Infer-Ms",
        "X-BNFM-RTF",
        "X-BNFM-Engine",
        "X-BNFM-SNR-Gain-DB",
    ],
)


@app.middleware("http")
async def cross_origin_isolation_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return response


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def api_health() -> Dict[str, Any]:
    engines = engine_manager.health_info()
    return {
        "status": "ok",
        "version": "2.0.0",
        "sample_rate": TARGET_SR,
        "engines": engines,
        "ffmpeg_available": bool(FFMPEG_BIN),
        "default_engine": "df3" if engines["df3"]["available"] else "gtcrn48k",
    }


@app.post("/api/denoise")
async def api_denoise(
    file: UploadFile = File(...),
    engine: str = Query(default="df3", pattern="^(df3|gtcrn48k)$"),
) -> Response:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    pcm_48k = decode_upload_to_48k_pcm(raw_bytes, file.filename or "input.wav")
    duration_sec = max(pcm_48k.shape[0] / float(TARGET_SR), 1e-4)

    t0 = time.perf_counter()
    enhanced_48k, active_engine = engine_manager.denoise(pcm_48k, requested_engine=engine)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    rtf = (infer_ms / 1000.0) / duration_sec
    snr_gain_db = _estimate_snr_gain_db(pcm_48k, enhanced_48k)

    out_wav = pcm_to_wav_bytes(enhanced_48k, sample_rate=TARGET_SR)
    return Response(
        content=out_wav,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="bnfm_enhanced_48k.wav"',
            "X-BNFM-Infer-Ms": f"{infer_ms:.2f}",
            "X-BNFM-RTF": f"{rtf:.4f}",
            "X-BNFM-Engine": active_engine,
            "X-BNFM-SNR-Gain-DB": f"{snr_gain_db:.2f}",
        },
    )


@app.post("/api/decode")
async def api_decode(file: UploadFile = File(...)) -> Response:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    ext = Path(file.filename or "audio.aac").suffix.lower() or ".aac"
    try:
        if FFMPEG_BIN:
            wav_bytes = decode_audio_via_ffmpeg(raw_bytes, suffix=ext)
        elif ext == ".wav":
            pcm, sr = decode_wav_bytes_native(raw_bytes)
            wav_bytes = pcm_to_wav_bytes(_resample_linear(pcm, sr, TARGET_SR), TARGET_SR)
        else:
            raise RuntimeError("ffmpeg is required to decode non-WAV formats.")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="decoded_48k.wav"'},
    )


@app.post("/api/encode-aac")
async def api_encode_aac(file: UploadFile = File(...)) -> Response:
    wav_bytes = await file.read()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Uploaded WAV file is empty.")

    try:
        m4a_bytes = encode_aac_via_ffmpeg(wav_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=m4a_bytes,
        media_type="audio/mp4",
        headers={"Content-Disposition": 'attachment; filename="bnfm_studio_48k.m4a"'},
    )


@app.post("/api/telemetry")
async def api_telemetry(payload: TelemetryPayload) -> Dict[str, Any]:
    store: TelemetryStore = get_store()
    row_id = store.log_telemetry(
        model_version=payload.model_version,
        backend=payload.backend,
        mode=payload.mode,
        avg_infer_ms=payload.avg_infer_ms,
        rtf=payload.rtf,
        est_snr_gain_db=payload.est_snr_gain_db,
        dry_wet_pct=payload.dry_wet_pct,
        underruns=payload.underruns,
        heap_mb=payload.heap_mb,
    )
    return {"status": "logged", "id": row_id}


@app.post("/api/feedback")
async def api_feedback(
    rating: str = Form(...),
    model_version: str = Form(default="df3_48k_v2"),
    dry_wet_pct: float = Form(default=100.0),
    opt_in_clip: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    rating_norm = rating.strip().lower()
    if rating_norm not in VALID_RATINGS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rating '{rating}'. Expected one of {sorted(VALID_RATINGS)}.",
        )

    store: TelemetryStore = get_store()
    clip_path: Optional[str] = None
    if opt_in_clip is not None:
        clip_bytes = await opt_in_clip.read()
        if clip_bytes:
            clip_path = store.save_opt_in_clip(clip_bytes)

    row_id = store.log_feedback(
        rating=rating_norm,
        model_version=model_version,
        dry_wet_pct=dry_wet_pct,
        opt_in_clip_path=clip_path,
    )
    return {"status": "recorded", "id": row_id, "opt_in_clip_path": clip_path}


@app.get("/api/metrics-dashboard")
async def api_metrics_dashboard(recent_limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    store: TelemetryStore = get_store()
    return store.get_metrics_dashboard(recent_limit=recent_limit)


# ---------------------------------------------------------------------------
# Static Frontend Serving (defaulting `/` to `web/input.html`)
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_root():
    input_html = WEB_DIR / "input.html"
    if input_html.exists():
        return FileResponse(str(input_html), media_type="text/html")
    index_html = WEB_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html), media_type="text/html")
    return JSONResponse({"status": "ok", "message": "Browser Noise Free Mic backend running; web/input.html not found."})


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="bnfm_web")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
