"""Generate all 4 mandatory 150-DPI PNG plots for NEON Browser Noise-Free Mic.

Outputs under `artifacts/plots/`:
1. `loss_curves.png` — Training & validation loss + SI-SDR curves from `eval/training_history.json`
2. `metric_comparison.png` — 4-model comparison bar chart from `eval/benchmark_report.json`
3. `spectrogram_comparison.png` — 3-panel aligned Inferno (0–8 kHz) Noisy / GTCRN INT8 / Clean
4. `latency_histogram.png` — 1,000 consecutive 16 ms streaming frames on CPU with Mean, P99, and 16 ms deadline
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import scipy.signal as spsig
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.gtcrn_clean import GTCRNStreamWrapper

DPI = 150
SAMPLE_RATE = 16000


def plot_loss_curves(history_path: Path, out_path: Path) -> None:
    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=DPI)
    gtcrn = hist.get("gtcrn", [])
    perc = hist.get("perceptron", [])

    ax = axes[0]
    if gtcrn:
        ep = [r["epoch"] for r in gtcrn]
        ax.plot(ep, [r["train_loss"] for r in gtcrn], "o-", label="GTCRN Train Loss", linewidth=2)
        ax.plot(ep, [r["val_loss"] for r in gtcrn], "s--", label="GTCRN Val Loss", linewidth=2)
    if perc:
        ep_p = [r["epoch"] for r in perc]
        ax.plot(ep_p, [r["val_loss"] for r in perc], "^:", label="Perceptron Val Loss", linewidth=1.8)
    ax.set_title("HybridSpeechLoss Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    if gtcrn:
        ep = [r["epoch"] for r in gtcrn]
        ax2.plot(ep, [r["val_si_sdr_db"] for r in gtcrn], "o-", color="#2ca02c", label="GTCRN Val SI-SDR (dB)", linewidth=2)
    if perc:
        ep_p = [r["epoch"] for r in perc]
        ax2.plot(ep_p, [r["val_si_sdr_db"] for r in perc], "^:", color="#ff7f0e", label="Perceptron Val SI-SDR (dB)", linewidth=1.8)
    ax2.set_title("Validation SI-SDR Progression (824 Unseen Clips)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SI-SDR (dB)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(report_path: Path, out_path: Path) -> None:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    models_map = report["models"]
    ordered = [
        ("wiener_filter", "Wiener\nFilter"),
        ("linear_perceptron", "Linear\nPerceptron"),
        ("gtcrn_fp32", "GTCRN\nFP32"),
        ("gtcrn_int8_onnx", "GTCRN\nINT8 ONNX"),
    ]
    labels = [lbl for k, lbl in ordered if k in models_map]
    keys = [k for k, _ in ordered if k in models_map]

    metrics_spec = [
        ("pesq_wb", "PESQ (WB)", "#1f77b4"),
        ("stoi", "STOI", "#2ca02c"),
        ("si_sdr_gain_db", "SI-SDR Gain (dB)", "#9467bd"),
        ("snr_gain_db", "SNR Gain (dB)", "#d62728"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), dpi=DPI)
    x = np.arange(len(keys))
    for ax, (m_key, title, color) in zip(axes, metrics_spec):
        vals = [models_map[k].get(m_key, 0.0) for k in keys]
        bars = ax.bar(x, vals, color=color, alpha=0.88, edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("NEON 4-Model Benchmark Comparison (824 Unseen VoiceBank-DEMAND Clips)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_spectrogram_comparison(demos_dir: Path, out_path: Path) -> None:
    panels = [
        ("Noisy Input", demos_dir / "sample_noisy.wav"),
        ("GTCRN INT8 Enhanced", demos_dir / "sample_gtcrn_int8.wav"),
        ("Clean Reference", demos_dir / "sample_clean.wav"),
    ]
    wavs = []
    for _, p in panels:
        w, sr = sf.read(p, dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=-1)
        wavs.append((w, sr))

    min_len = min(len(w) for w, _ in wavs)
    specs_db = []
    times_ref = None
    freqs_ref = None
    for w, sr in wavs:
        f, t, Zxx = spsig.stft(w[:min_len], fs=sr, window="hann", nperseg=512, noverlap=256, nfft=512)
        db = 20.0 * np.log10(np.abs(Zxx) + 1e-6)
        specs_db.append(db)
        freqs_ref, times_ref = f / 1000.0, t  # kHz

    vmax = max(float(np.percentile(s, 99.5)) for s in specs_db)
    vmin = vmax - 80.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), dpi=DPI, sharex=True, sharey=True)
    im = None
    for ax, (title, _), db in zip(axes, panels, specs_db):
        im = ax.pcolormesh(times_ref, freqs_ref, db, shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Frequency (kHz)")
        ax.set_ylim(0.0, 8.0)

    axes[-1].set_xlabel("Time (s)")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, aspect=30)
    cbar.set_label("Magnitude (dB)")
    fig.suptitle("Aligned Spectrogram Comparison (0–8 kHz, Inferno)", fontsize=12.5, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_latency_histogram(stream_onnx_path: Path, out_path: Path, num_frames: int = 1000) -> Dict[str, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(stream_onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    in_specs = GTCRNStreamWrapper.INPUT_SPECS
    out_names = [n for n, _ in GTCRNStreamWrapper.OUTPUT_SPECS]
    state_in_names = [n for n, _ in in_specs if n != "frame_spec"]

    states = {n: np.zeros(shape, dtype=np.float32) for n, shape in in_specs if n != "frame_spec"}
    rng = np.random.default_rng(42)

    # Warmup 20 frames
    for _ in range(20):
        feed = {"frame_spec": rng.standard_normal((1, 2, 257, 1)).astype(np.float32), **states}
        outs = sess.run(out_names, feed)
        for idx, s_name in enumerate(state_in_names):
            states[s_name] = outs[2 + idx]

    latencies_ms = np.empty(num_frames, dtype=np.float64)
    for i in range(num_frames):
        feed = {"frame_spec": rng.standard_normal((1, 2, 257, 1)).astype(np.float32), **states}
        t0 = time.perf_counter()
        outs = sess.run(out_names, feed)
        latencies_ms[i] = (time.perf_counter() - t0) * 1000.0
        for idx, s_name in enumerate(state_in_names):
            states[s_name] = outs[2 + idx]

    mean_ms = float(np.mean(latencies_ms))
    p99_ms = float(np.percentile(latencies_ms, 99.0))
    deadline_ms = 16.0

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=DPI)
    ax.hist(latencies_ms, bins=40, color="#1f77b4", alpha=0.82, edgecolor="black", linewidth=0.5)
    ax.axvline(mean_ms, color="#2ca02c", linestyle="--", linewidth=2.0, label=f"Mean: {mean_ms:.2f} ms")
    ax.axvline(p99_ms, color="#ff7f0e", linestyle="-.", linewidth=2.0, label=f"P99: {p99_ms:.2f} ms")
    ax.axvline(deadline_ms, color="#d62728", linestyle="-", linewidth=2.2, label="16.0 ms Real-Time Deadline")

    ax.set_title(f"GTCRN Streaming CPU Latency ({num_frames} Consecutive 16 ms Frames)")
    ax.set_xlabel("Per-Frame Inference Latency (ms)")
    ax.set_ylabel("Frame Count")
    ax.set_xlim(0.0, max(deadline_ms * 1.15, p99_ms * 1.2))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return {"mean_ms": mean_ms, "p99_ms": p99_ms, "deadline_ms": deadline_ms}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all 4 NEON 150-DPI PNG plots.")
    parser.add_argument("--root-dir", type=str, default=str(REPO_ROOT), help="Project root directory")
    parser.add_argument("--num-frames", type=int, default=1000, help="Streaming frames to benchmark (default: 1000)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    plots_dir = root_dir / "artifacts" / "plots"

    hist_path = root_dir / "eval" / "training_history.json"
    if hist_path.exists():
        plot_loss_curves(hist_path, plots_dir / "loss_curves.png")
        print(f"Generated {plots_dir / 'loss_curves.png'}")

    report_path = root_dir / "eval" / "benchmark_report.json"
    if report_path.exists():
        plot_metric_comparison(report_path, plots_dir / "metric_comparison.png")
        print(f"Generated {plots_dir / 'metric_comparison.png'}")

    demos_dir = root_dir / "artifacts" / "demos"
    if (demos_dir / "sample_noisy.wav").exists():
        plot_spectrogram_comparison(demos_dir, plots_dir / "spectrogram_comparison.png")
        print(f"Generated {plots_dir / 'spectrogram_comparison.png'}")

    stream_onnx = root_dir / "exports" / "gtcrn_stream_int8.onnx"
    if not stream_onnx.exists():
        stream_onnx = root_dir / "exports" / "gtcrn_stream_fp32.onnx"
    if stream_onnx.exists():
        stats = plot_latency_histogram(stream_onnx, plots_dir / "latency_histogram.png", num_frames=args.num_frames)
        print(
            f"Generated {plots_dir / 'latency_histogram.png'} "
            f"(Mean={stats['mean_ms']:.2f} ms, P99={stats['p99_ms']:.2f} ms)"
        )


if __name__ == "__main__":
    main()
