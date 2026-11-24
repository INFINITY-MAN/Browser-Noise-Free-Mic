"""FastAPI Backend for NEON Browser Noise-Free Mic.

Features:
- Global COOP/COEP middleware (`Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`) so SharedArrayBuffer & threaded
  ONNX Runtime Web / AudioWorklet run smoothly on every page & asset.
- GET  /api/health             : Service health, ONNX model availability & file sizes.
- POST /api/denoise            : Server-side 16 kHz ONNX Runtime WAV denoiser.
- POST /api/telemetry          : Anonymized runtime telemetry ingestion (SQLite + JSONL).
- POST /api/feedback           : User rating (`clean` | `over_suppressed` | `residual_noise`)
                                 + optional opt-in WAV clip (JSON base64 or multipart).
- GET  /api/metrics-dashboard  : Aggregated metrics dashboard JSON.
- Static mount of `web/` at `/`.
"""

from __future__ import annotations

import base64
import io
import math
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy import signal
from starlette.middleware.base import BaseHTTPMiddleware

from backend.telemetry_store import (
    VALID_RATINGS,
    get_metrics_dashboard,
    log_feedback,
    log_telemetry,
    save_opt_in_clip,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = PROJECT_ROOT / "exports"
WEB_DIR = PROJECT_ROOT / "web"

TARGET_SR = 16000
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512

MODEL_CANDIDATES = (
    "exports/gtcrn_int8.onnx",
    "exports/gtcrn_fp32.onnx",
    "exports/gtcrn_stream_fp32.onnx",
    "web/models/gtcrn_int8.onnx",
    "web/models/gtcrn_fp32.onnx",
    "web/models/gtcrn_stream_fp32.onnx",
)

# ---------------------------------------------------------------------------
# Middleware: COOP / COEP on all responses
# ---------------------------------------------------------------------------


class COOPCOEPMiddleware(BaseHTTPMiddleware):
    """Attach COOP/COEP headers to every HTTP response for SharedArrayBuffer."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        return response


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TelemetryPayload(BaseModel):
    model_version: str = Field(default="gtcrn_int8_v1", max_length=128)
    backend: str = Field(default="wasm", max_length=32)
    mode: str = Field(default="A", max_length=32)
    avg_infer_ms: float = Field(default=0.0, ge=0.0)
    rtf: float = Field(default=0.0, ge=0.0)
    est_snr_gain_db: float = Field(default=0.0)
    dry_wet_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    underruns: int = Field(default=0, ge=0)
    heap_mb: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# Audio & ONNX Runtime Helpers
# ---------------------------------------------------------------------------

_ort_lock = threading.Lock()
_ort_sessions: Dict[str, Any] = {}


def _discover_models() -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    for rel in MODEL_CANDIDATES:
        path = PROJECT_ROOT / rel
        exists = path.is_file()
        size_bytes = path.stat().st_size if exists else 0
        info[rel] = {
            "exists": exists,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 4) if exists else 0.0,
        }
    return info


def _resolve_denoise_model(preferred: str = "auto") -> Optional[Path]:
    if preferred == "fp32":
        order = (
            "exports/gtcrn_fp32.onnx",
            "web/models/gtcrn_fp32.onnx",
            "exports/gtcrn_int8.onnx",
            "web/models/gtcrn_int8.onnx",
        )
    else:
        order = (
            "exports/gtcrn_int8.onnx",
            "exports/gtcrn_fp32.onnx",
            "web/models/gtcrn_int8.onnx",
            "web/models/gtcrn_fp32.onnx",
        )
    for rel in order:
        p = PROJECT_ROOT / rel
        if p.is_file():
            return p
    return None


def _get_ort_session(model_path: Path):
    import onnxruntime as ort

    key = str(model_path.resolve())
    with _ort_lock:
        if key not in _ort_sessions:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _ort_sessions[key] = ort.InferenceSession(
                key, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        return _ort_sessions[key]


def _read_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes to mono float32 array in [-1, 1] and sample rate."""
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1, dtype=np.float32)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception:
        pass

    # Fallback: scipy.io.wavfile
    from scipy.io import wavfile

    sr, data = wavfile.read(io.BytesIO(wav_bytes))
    arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        max_abs = float(max(abs(info.min), info.max))
        arr = arr.astype(np.float32) / max_abs
    else:
        arr = arr.astype(np.float32)
    return arr, int(sr)


def _resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample 1D float32 audio to 16 kHz preserving exact duration and pitch."""
    if orig_sr == TARGET_SR or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    g = math.gcd(TARGET_SR, orig_sr)
    up = TARGET_SR // g
    down = orig_sr // g
    resampled = signal.resample_poly(audio.astype(np.float64), up, down)
    return resampled.astype(np.float32)


def _encode_wav_16k(audio: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Encode 1D float32 waveform to 16-bit PCM WAV bytes."""
    clipped = np.clip(np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _stft_sqrt_hann(wav: np.ndarray) -> np.ndarray:
    """Compute complex STFT (F=257, T) using sqrt-Hann window (GTCRN standard)."""
    win = np.sqrt(np.hanning(WIN_LENGTH + 1)[:-1]).astype(np.float32)
    pad = WIN_LENGTH - HOP_LENGTH
    padded = np.pad(wav, (pad, pad), mode="constant")
    n_frames = 1 + max(0, (len(padded) - WIN_LENGTH) // HOP_LENGTH)
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, WIN_LENGTH),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]),
    )
    spec = np.fft.rfft(frames * win[None, :], n=N_FFT, axis=1)  # (T, F)
    return spec.T  # (F, T) complex


def _istft_sqrt_hann(spec_ft: np.ndarray, target_len: int) -> np.ndarray:
    """Inverse STFT from complex (F=257, T) with sqrt-Hann overlap-add."""
    win = np.sqrt(np.hanning(WIN_LENGTH + 1)[:-1]).astype(np.float32)
    frames = np.fft.irfft(spec_ft.T, n=N_FFT, axis=1).astype(np.float32)  # (T, 512)
    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * HOP_LENGTH + WIN_LENGTH
    out = np.zeros(out_len, dtype=np.float32)
    win_sq = np.zeros(out_len, dtype=np.float32)
    w2 = win * win
    for i in range(n_frames):
        start = i * HOP_LENGTH
        out[start : start + WIN_LENGTH] += frames[i] * win
        win_sq[start : start + WIN_LENGTH] += w2
    out /= np.maximum(win_sq, 1e-8)
    pad = WIN_LENGTH - HOP_LENGTH
    trimmed = out[pad : pad + target_len]
    if len(trimmed) < target_len:
        trimmed = np.pad(trimmed, (0, target_len - len(trimmed)))
    return trimmed


def _run_onnx_denoise(audio_16k: np.ndarray, model_path: Path) -> np.ndarray:
    """Run GTCRN ONNX model (auto-detecting STFT [1,F,T,2] / [1,2,F,T] or waveform [1,N])."""
    sess = _get_ort_session(model_path)
    inp_meta = sess.get_inputs()[0]
    inp_name = inp_meta.name
    inp_shape = inp_meta.shape
    orig_len = len(audio_16k)

    # Case 1: 4D STFT input (standard GTCRN: [B, 257, T, 2] or [B, 2, 257, T])
    if len(inp_shape) == 4:
        spec = _stft_sqrt_hann(audio_16k)  # (257, T) complex
        real = spec.real.astype(np.float32)
        imag = spec.imag.astype(np.float32)

        channels_last = (inp_shape[-1] == 2) or (inp_shape[1] == 257)
        if channels_last:
            x = np.stack([real, imag], axis=-1)[None, :, :, :]  # (1, 257, T, 2)
        else:
            x = np.stack([real, imag], axis=0)[None, :, :, :]  # (1, 2, 257, T)

        feeds = {inp_name: x}
        # Zero-initialize any extra state inputs if a streaming model is used
        for extra in sess.get_inputs()[1:]:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in extra.shape]
            feeds[extra.name] = np.zeros(shape, dtype=np.float32)

        out = sess.run(None, feeds)[0]
        if out.ndim == 4:
            if out.shape[-1] == 2:
                out_spec = out[0, :, :, 0] + 1j * out[0, :, :, 1]
            else:
                out_spec = out[0, 0, :, :] + 1j * out[0, 1, :, :]
            return _istft_sqrt_hann(out_spec, target_len=orig_len)
        return np.asarray(out, dtype=np.float32).reshape(-1)[:orig_len]

    # Case 2: 2D/3D raw waveform input ([1, N] or [1, 1, N])
    x_wav = audio_16k[None, :] if len(inp_shape) == 2 else audio_16k[None, None, :]
    out = sess.run(None, {inp_name: x_wav.astype(np.float32)})[0]
    y = np.asarray(out, dtype=np.float32).reshape(-1)
    if len(y) < orig_len:
        y = np.pad(y, (0, orig_len - len(y)))
    return y[:orig_len]


# ---------------------------------------------------------------------------
# FastAPI Application & Routes
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NEON Browser Noise-Free Mic",
    version="1.0.0",
    description="Real-time GTCRN speech enhancement backend, telemetry store, and static web host.",
)
app.add_middleware(COOPCOEPMiddleware)


@app.get("/api/health")
async def api_health() -> Dict[str, Any]:
    models = _discover_models()
    active = _resolve_denoise_model("auto")
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:
        providers = []

    return {
        "status": "ok",
        "model_available": active is not None,
        "active_model": str(active.relative_to(PROJECT_ROOT)) if active else None,
        "onnx_files": models,
        "onnxruntime_providers": providers,
    }


@app.post("/api/denoise")
async def api_denoise(
    file: UploadFile = File(...),
    dry_wet_pct: float = Query(default=100.0, ge=0.0, le=100.0),
    model: str = Query(default="auto", pattern="^(auto|int8|fp32)$"),
) -> Response:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded WAV file is empty.")

    model_path = _resolve_denoise_model(model)
    if model_path is None:
        raise HTTPException(
            status_code=503,
            detail="No ONNX model found in exports/ or web/models/ (expected gtcrn_int8.onnx or gtcrn_fp32.onnx).",
        )

    try:
        audio, orig_sr = _read_wav_bytes(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {exc}") from exc

    audio_16k = _resample_to_16k(audio, orig_sr)
    if audio_16k.size == 0:
        raise HTTPException(status_code=400, detail="Decoded audio has zero samples.")

    t0 = time.perf_counter()
    try:
        enhanced = _run_onnx_denoise(audio_16k, model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    wet = float(dry_wet_pct) / 100.0
    mixed = wet * enhanced + (1.0 - wet) * audio_16k
    out_wav = _encode_wav_16k(mixed, TARGET_SR)

    return Response(
        content=out_wav,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="neon_denoised_16k.wav"',
            "X-Model-Used": str(model_path.relative_to(PROJECT_ROOT)),
            "X-Processing-Time-Ms": f"{elapsed_ms:.2f}",
            "X-Sample-Rate": str(TARGET_SR),
        },
    )


@app.post("/api/telemetry")
async def api_telemetry(payload: TelemetryPayload) -> Dict[str, Any]:
    row_id = log_telemetry(
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
    return {"status": "ok", "id": row_id}


@app.post("/api/feedback")
async def api_feedback(request: Request) -> JSONResponse:
    """Accept user rating (`clean` | `over_suppressed` | `residual_noise`) via JSON or multipart.

    Optional opt-in audio clip is saved ONLY when the user explicitly opts in.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    rating: Optional[str] = None
    model_version = "gtcrn_int8_v1"
    dry_wet_pct = 100.0
    opt_in = False
    audio_bytes: Optional[bytes] = None

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        rating = str(form.get("rating", "")).strip().lower()
        model_version = str(form.get("model_version", model_version))
        dry_wet_pct = float(form.get("dry_wet_pct", dry_wet_pct))
        opt_in_raw = str(form.get("opt_in", "false")).strip().lower()
        opt_in = opt_in_raw in {"1", "true", "yes", "on"}

        upload = form.get("audio_clip") or form.get("file") or form.get("clip")
        if upload is not None and hasattr(upload, "read"):
            raw = await upload.read()
            if raw:
                audio_bytes = raw
                # Presence of an uploaded clip alongside opt_in=true saves it
    else:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

        rating = str(body.get("rating", "")).strip().lower()
        model_version = str(body.get("model_version", model_version))
        dry_wet_pct = float(body.get("dry_wet_pct", dry_wet_pct))
        opt_in = bool(body.get("opt_in", False))
        b64_clip = body.get("opt_in_audio_b64") or body.get("audio_base64") or body.get("audio_clip_b64")
        if opt_in and b64_clip:
            if "," in b64_clip and b64_clip.startswith("data:"):
                b64_clip = b64_clip.split(",", 1)[1]
            try:
                audio_bytes = base64.b64decode(b64_clip)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}") from exc

    if rating not in VALID_RATINGS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid rating '{rating}'. Must be one of {sorted(VALID_RATINGS)}.",
        )

    clip_path: Optional[str] = None
    if opt_in and audio_bytes:
        clip_path = save_opt_in_clip(audio_bytes)

    row_id = log_feedback(
        rating=rating,
        model_version=model_version,
        dry_wet_pct=dry_wet_pct,
        opt_in_clip_path=clip_path,
    )
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "id": row_id,
            "rating": rating,
            "opt_in_clip_saved": clip_path is not None,
            "opt_in_clip_path": clip_path,
        },
    )


@app.get("/api/metrics-dashboard")
async def api_metrics_dashboard() -> Dict[str, Any]:
    return get_metrics_dashboard()


# Mount static web UI at `/` after all `/api/*` routes
WEB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
