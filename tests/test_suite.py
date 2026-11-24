"""Mandatory 6-Test Pytest QA Suite for NEON Browser Noise-Free Mic.

Covers:
1. test_pytorch_onnx_numerical_parity
2. test_streaming_vs_batch_equivalence
3. test_zero_and_clipping_inputs
4. test_resampling_pitch_and_duration_conservation
5. test_runtime_latency_and_memory_stability
6. test_strict_causality
"""

from __future__ import annotations

import importlib
import math
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPORTS_DIR = PROJECT_ROOT / "exports"
FP32_ONNX = EXPORTS_DIR / "gtcrn_fp32.onnx"
INT8_ONNX = EXPORTS_DIR / "gtcrn_int8.onnx"
STREAM_FP32_ONNX = EXPORTS_DIR / "gtcrn_stream_fp32.onnx"


def _load_pytorch_gtcrn():
    """Attempt to import GTCRNModel and load exported weights if present."""
    import torch

    candidate_modules = (
        "model.gtcrn_clean",
        "model.gtcrn",
        "models.gtcrn",
        "src.gtcrn",
        "gtcrn",
        "training.gtcrn",
        "ml.gtcrn",
    )
    model_cls = None
    for mod_name in candidate_modules:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "GTCRNModel"):
                model_cls = getattr(mod, "GTCRNModel")
                break
            if hasattr(mod, "GTCRN"):
                model_cls = getattr(mod, "GTCRN")
                break
        except Exception:
            continue

    if model_cls is None:
        pytest.skip("GTCRNModel Python module not found in project root.")

    model = model_cls()
    ckpt_candidates = (
        PROJECT_ROOT / "exports" / "gtcrn_best.pt",
        PROJECT_ROOT / "exports" / "gtcrn.pt",
        PROJECT_ROOT / "checkpoints" / "gtcrn_best.pt",
        PROJECT_ROOT / "checkpoints" / "best.pt",
        PROJECT_ROOT / "weights" / "gtcrn_best.pt",
    )
    for ckpt in ckpt_candidates:
        if ckpt.is_file():
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            break

    model.eval()
    return model


def _make_ort_session(onnx_path: Path):
    import onnxruntime as ort

    if not onnx_path.is_file():
        pytest.skip(f"ONNX artifact not found: {onnx_path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def _infer_spec_layout(session) -> Tuple[str, Tuple[int, ...]]:
    """Inspect primary ONNX input metadata and return (name, concrete_shape_for_T)."""
    meta = session.get_inputs()[0]
    return meta.name, tuple(meta.shape)


def _make_spec_input(shape_meta: Tuple[Any, ...], n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Create a synthetic STFT tensor matching either [1, 257, T, 2] or [1, 2, 257, T]."""
    if len(shape_meta) == 4:
        if shape_meta[-1] == 2 or shape_meta[1] == 257:
            return rng.standard_normal((1, 257, n_frames, 2)).astype(np.float32) * 0.25
        return rng.standard_normal((1, 2, 257, n_frames)).astype(np.float32) * 0.25
    return rng.standard_normal((1, n_frames * 256)).astype(np.float32) * 0.25


def _run_streaming_frames(
    stream_sess, x_full: np.ndarray, shape_meta: Tuple[Any, ...]
) -> np.ndarray:
    """Execute 1-frame-at-a-time streaming inference carrying all cache states forward."""
    inputs = stream_sess.get_inputs()
    primary_name = inputs[0].name

    state_feeds: Dict[str, np.ndarray] = {}
    for inp in inputs[1:]:
        concrete = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        state_feeds[inp.name] = np.zeros(concrete, dtype=np.float32)

    channels_last = len(shape_meta) == 4 and (shape_meta[-1] == 2 or shape_meta[1] == 257)
    n_frames = x_full.shape[2] if channels_last else x_full.shape[3]

    frame_outputs: List[np.ndarray] = []
    for t in range(n_frames):
        if channels_last:
            frame_in = x_full[:, :, t : t + 1, :]
        else:
            frame_in = x_full[:, :, :, t : t + 1]

        feeds = {primary_name: frame_in, **state_feeds}
        ort_outs = stream_sess.run(None, feeds)
        state_offset = len(ort_outs) - len(inputs[1:])
        if state_offset == 2:
            frame_outputs.append(np.concatenate([ort_outs[0], ort_outs[1]], axis=1))
        else:
            frame_outputs.append(ort_outs[0])

        for inp_meta, out_val in zip(inputs[1:], ort_outs[state_offset:]):
            state_feeds[inp_meta.name] = out_val

    cat_axis = 2 if channels_last else 3
    return np.concatenate(frame_outputs, axis=cat_axis)


def test_pytorch_onnx_numerical_parity():
    import torch

    model = _load_pytorch_gtcrn()
    fp32_sess = _make_ort_session(FP32_ONNX)

    inp_name, shape_meta = _infer_spec_layout(fp32_sess)
    rng = np.random.default_rng(42)
    x_np = _make_spec_input(shape_meta, n_frames=32, rng=rng)

    with torch.no_grad():
        pt_out = model(torch.from_numpy(x_np))
        if isinstance(pt_out, (tuple, list)):
            pt_out = pt_out[0]
        pt_out_np = pt_out.detach().cpu().numpy()

    ort_fp32_out = fp32_sess.run(None, {inp_name: x_np})[0]
    max_abs_diff = float(np.max(np.abs(pt_out_np - ort_fp32_out)))
    assert max_abs_diff < 1e-4, f"PyTorch vs FP32 ONNX max_abs_diff={max_abs_diff:.6e} >= 1e-4"

    if INT8_ONNX.is_file():
        int8_sess = _make_ort_session(INT8_ONNX)
        int8_inp = int8_sess.get_inputs()[0].name
        ort_int8_out = int8_sess.run(None, {int8_inp: x_np})[0]
        int8_mean_diff = float(np.mean(np.abs(ort_fp32_out - ort_int8_out)))
        assert int8_mean_diff < 0.05, (
            f"INT8 vs FP32 ONNX mean_abs_diff={int8_mean_diff:.6f} >= 0.05"
        )


def test_streaming_vs_batch_equivalence():
    import torch

    stream_sess = _make_ort_session(STREAM_FP32_ONNX)
    _, shape_meta = _infer_spec_layout(stream_sess)

    rng = np.random.default_rng(123)
    x_np = _make_spec_input(shape_meta, n_frames=100, rng=rng)

    stream_out = _run_streaming_frames(stream_sess, x_np, shape_meta)

    model = _load_pytorch_gtcrn()
    with torch.no_grad():
        batch_out = model(torch.from_numpy(x_np))
        if isinstance(batch_out, (tuple, list)):
            batch_out = batch_out[0]
        batch_out_np = batch_out.detach().cpu().numpy()

    max_abs_diff = float(np.max(np.abs(stream_out - batch_out_np)))
    assert max_abs_diff < 1e-4, (
        f"100-frame streaming vs batch max_abs_diff={max_abs_diff:.6e} >= 1e-4"
    )


def test_zero_and_clipping_inputs():
    target_onnx = FP32_ONNX if FP32_ONNX.is_file() else (
        INT8_ONNX if INT8_ONNX.is_file() else STREAM_FP32_ONNX
    )
    sess = _make_ort_session(target_onnx)
    inp_name, shape_meta = _infer_spec_layout(sess)

    n_frames = 1 if target_onnx == STREAM_FP32_ONNX else 16
    rng = np.random.default_rng(7)
    template = _make_spec_input(shape_meta, n_frames=n_frames, rng=rng)

    zeros_in = np.zeros_like(template, dtype=np.float32)
    clip_in = np.full_like(template, 1e3, dtype=np.float32)

    extra_feeds = {}
    for extra in sess.get_inputs()[1:]:
        concrete = [d if isinstance(d, int) and d > 0 else 1 for d in extra.shape]
        extra_feeds[extra.name] = np.zeros(concrete, dtype=np.float32)

    out_zero = sess.run(None, {inp_name: zeros_in, **extra_feeds})[0]
    assert not np.isnan(out_zero).any(), "NaN detected on all-zero silence input."
    assert np.all(np.isfinite(out_zero)), "Non-finite value detected on all-zero silence input."

    out_clip = sess.run(None, {inp_name: clip_in, **extra_feeds})[0]
    assert not np.isnan(out_clip).any(), "NaN detected on 1e3 clipped input."
    assert not np.isinf(out_clip).any(), "Inf detected on 1e3 clipped input."


def test_resampling_pitch_and_duration_conservation():
    orig_sr = 48000
    target_sr = 16000
    duration_s = 3.500
    freq_hz = 440.0

    n_samples_48k = int(round(orig_sr * duration_s))
    t = np.arange(n_samples_48k, dtype=np.float64) / orig_sr
    sine_48k = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)

    g = math.gcd(target_sr, orig_sr)
    resampled_16k = signal.resample_poly(
        sine_48k.astype(np.float64), target_sr // g, orig_sr // g
    ).astype(np.float32)

    actual_duration_s = len(resampled_16k) / float(target_sr)
    assert abs(actual_duration_s - duration_s) <= 0.001, (
        f"Duration drift: expected {duration_s:.3f}s ± 0.001s, got {actual_duration_s:.6f}s"
    )

    window = np.hanning(len(resampled_16k))
    mag = np.abs(np.fft.rfft(resampled_16k * window))
    freqs = np.fft.rfftfreq(len(resampled_16k), d=1.0 / target_sr)
    peak_freq = float(freqs[int(np.argmax(mag))])
    assert abs(peak_freq - freq_hz) <= 1.0, (
        f"Pitch drift: expected {freq_hz:.1f}Hz ± 1.0Hz, got {peak_freq:.3f}Hz"
    )


def test_runtime_latency_and_memory_stability():
    target_onnx = STREAM_FP32_ONNX if STREAM_FP32_ONNX.is_file() else INT8_ONNX
    sess = _make_ort_session(target_onnx)
    inputs = sess.get_inputs()
    primary_name, shape_meta = _infer_spec_layout(sess)

    rng = np.random.default_rng(2026)
    frame_in = _make_spec_input(shape_meta, n_frames=1, rng=rng)

    state_feeds: Dict[str, np.ndarray] = {}
    for inp in inputs[1:]:
        concrete = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        state_feeds[inp.name] = np.zeros(concrete, dtype=np.float32)

    state_offset = len(sess.get_outputs()) - len(inputs[1:])
    for _ in range(20):
        outs = sess.run(None, {primary_name: frame_in, **state_feeds})
        for inp_meta, out_val in zip(inputs[1:], outs[state_offset:]):
            state_feeds[inp_meta.name] = out_val

    latencies_ms: List[float] = []
    mem_samples_kb: List[float] = []

    tracemalloc.start()
    try:
        for i in range(1000):
            t0 = time.perf_counter()
            outs = sess.run(None, {primary_name: frame_in, **state_feeds})
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            for inp_meta, out_val in zip(inputs[1:], outs[state_offset:]):
                state_feeds[inp_meta.name] = out_val

            if (i + 1) % 200 == 0:
                current_bytes, _ = tracemalloc.get_traced_memory()
                mem_samples_kb.append(current_bytes / 1024.0)
    finally:
        tracemalloc.stop()

    mean_ms = float(np.mean(latencies_ms))
    p99_ms = float(np.percentile(latencies_ms, 99))
    p50_ms = float(np.percentile(latencies_ms, 50))
    mem_growth_mb = (mem_samples_kb[-1] - mem_samples_kb[0]) / 1024.0 if len(mem_samples_kb) >= 2 else 0.0
    print(f"\n[LATENCY] mean={mean_ms:.3f} ms, p50={p50_ms:.3f} ms, p99={p99_ms:.3f} ms, heap_growth={mem_growth_mb:+.3f} MB")

    assert mean_ms < 8.0, (
        f"Mean streaming latency {mean_ms:.3f} ms exceeded 8.0 ms fallback ceiling (target < 4.0 ms)."
    )
    assert p99_ms < 16.0, (
        f"P99 streaming latency {p99_ms:.3f} ms exceeded 16.0 ms real-time frame budget."
    )

    if len(mem_samples_kb) >= 2:
        mem_growth_mb = (mem_samples_kb[-1] - mem_samples_kb[0]) / 1024.0
        assert mem_growth_mb < 20.0, (
            f"Unbounded Python heap growth detected: +{mem_growth_mb:.2f} MB over 800 frames."
        )


def test_strict_causality():
    sess = _make_ort_session(FP32_ONNX)
    inp_name, shape_meta = _infer_spec_layout(sess)

    rng = np.random.default_rng(999)
    n_frames = 40
    split_t = 20

    x_a = _make_spec_input(shape_meta, n_frames=n_frames, rng=rng)
    x_b = x_a.copy()

    channels_last = len(shape_meta) == 4 and (shape_meta[-1] == 2 or shape_meta[1] == 257)
    if channels_last:
        x_b[:, :, split_t:, :] += rng.standard_normal(x_b[:, :, split_t:, :].shape).astype(np.float32) * 10.0
    else:
        x_b[:, :, :, split_t:] += rng.standard_normal(x_b[:, :, :, split_t:].shape).astype(np.float32) * 10.0

    out_a = sess.run(None, {inp_name: x_a})[0]
    out_b = sess.run(None, {inp_name: x_b})[0]

    if channels_last:
        past_diff = float(np.max(np.abs(out_a[:, :, :split_t, :] - out_b[:, :, :split_t, :])))
    else:
        past_diff = float(np.max(np.abs(out_a[:, :, :, :split_t] - out_b[:, :, :, :split_t])))

    assert past_diff == 0.0, (
        f"Causality violation! Perturbing t >= {split_t} changed t < {split_t} by max diff {past_diff:.6e}"
    )
