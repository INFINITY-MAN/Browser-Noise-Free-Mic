"""4-Model Evaluation Harness on all 824 unseen VoiceBank-DEMAND-16k test clips.

Evaluates:
1. Classical Wiener Filter (`wiener_denoise`, 0 params)
2. Linear Perceptron Mask (`checkpoints/perceptron_best.pt`, 132,612 params)
3. GTCRN FP32 (`exports/gtcrn_fp32.onnx` or `checkpoints/gtcrn_best.pt`)
4. GTCRN INT8 ONNX (`exports/gtcrn_int8.onnx`)

Computes PESQ (wb), STOI, SI-SDR Gain (dB), SNR Gain (dB), Pearson waveform
correlation, and CPU RTF. Saves `eval/benchmark_report.json` and 4 paired demo WAVs
under `artifacts/demos/`.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.baselines import LinearPerceptronMask, wiener_denoise
from model.gtcrn_clean import GTCRNModel
from model.stft import istft, stft
from training.train import SAMPLE_RATE, extract_audio_1d, load_voicebank_splits


def compute_si_sdr_np(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est_zm = est.astype(np.float64) - np.mean(est)
    ref_zm = ref.astype(np.float64) - np.mean(ref)
    dot = np.sum(est_zm * ref_zm)
    ref_energy = np.sum(ref_zm * ref_zm) + eps
    s_target = (dot / ref_energy) * ref_zm
    e_noise = est_zm - s_target
    ratio = (np.sum(s_target * s_target) + eps) / (np.sum(e_noise * e_noise) + eps)
    return float(10.0 * np.log10(ratio + eps))


def compute_snr_np(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    ref64 = ref.astype(np.float64)
    err = ref64 - est.astype(np.float64)
    return float(10.0 * np.log10((np.sum(ref64 ** 2) + eps) / (np.sum(err ** 2) + eps)))


def compute_pearson_np(est: np.ndarray, ref: np.ndarray) -> float:
    if np.std(est) < 1e-8 or np.std(ref) < 1e-8:
        return 0.0
    r = float(np.corrcoef(est.astype(np.float64), ref.astype(np.float64))[0, 1])
    return r if np.isfinite(r) else 0.0


def run_torch_model_cpu(model: torch.nn.Module, noisy_1d: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        wav_t = torch.from_numpy(noisy_1d).unsqueeze(0)
        spec = stft(wav_t)
        enh_spec = model(spec)
        enh = istft(enh_spec, length=len(noisy_1d)).squeeze(0).numpy()
    return np.clip(enh, -1.0, 1.0).astype(np.float32)


def run_onnx_model_cpu(session: ort.InferenceSession, noisy_1d: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        wav_t = torch.from_numpy(noisy_1d).unsqueeze(0)
        spec_np = stft(wav_t).numpy().astype(np.float32)
        in_name = session.get_inputs()[0].name
        enh_spec_np = session.run(None, {in_name: spec_np})[0]
        enh = istft(torch.from_numpy(enh_spec_np), length=len(noisy_1d)).squeeze(0).numpy()
    return np.clip(enh, -1.0, 1.0).astype(np.float32)


def make_cpu_ort_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def evaluate_clip_metrics(enh: np.ndarray, noisy: np.ndarray, clean: np.ndarray) -> Dict[str, Optional[float]]:
    noisy_sisdr = compute_si_sdr_np(noisy, clean)
    enh_sisdr = compute_si_sdr_np(enh, clean)
    noisy_snr = compute_snr_np(noisy, clean)
    enh_snr = compute_snr_np(enh, clean)
    pearson_r = compute_pearson_np(enh, clean)

    try:
        pesq_val = float(pesq(SAMPLE_RATE, clean, enh, "wb"))
        if not np.isfinite(pesq_val):
            pesq_val = None
    except Exception:
        pesq_val = None

    try:
        stoi_val = float(stoi(clean, enh, SAMPLE_RATE, extended=False))
        if not np.isfinite(stoi_val):
            stoi_val = None
    except Exception:
        stoi_val = None

    return {
        "pesq_wb": pesq_val,
        "stoi": stoi_val,
        "si_sdr_db": enh_sisdr,
        "si_sdr_gain_db": enh_sisdr - noisy_sisdr,
        "snr_db": enh_snr,
        "snr_gain_db": enh_snr - noisy_snr,
        "pearson_r": pearson_r,
    }


def summarize_records(records: List[Dict[str, Optional[float]]], total_infer_sec: float, total_audio_sec: float) -> Dict[str, float]:
    def _mean(key: str) -> float:
        vals = [r[key] for r in records if r[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "pesq_wb": _mean("pesq_wb"),
        "stoi": _mean("stoi"),
        "si_sdr_db": _mean("si_sdr_db"),
        "si_sdr_gain_db": _mean("si_sdr_gain_db"),
        "snr_db": _mean("snr_db"),
        "snr_gain_db": _mean("snr_gain_db"),
        "pearson_r": _mean("pearson_r"),
        "cpu_rtf": float(total_infer_sec / max(total_audio_sec, 1e-8)),
        "num_clips": len(records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all 4 NEON models on VoiceBank-DEMAND-16k test set.")
    parser.add_argument("--root-dir", type=str, default=str(REPO_ROOT), help="Project root directory")
    parser.add_argument("--demo-idx", type=int, default=0, help="Test clip index to export as demo WAVs")
    parser.add_argument("--max-clips", type=int, default=0, help="Optional limit for quick smoke tests (0=all 824)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    torch.set_num_threads(1)

    _, test_split = load_voicebank_splits()
    total_clips = len(test_split) if args.max_clips <= 0 else min(args.max_clips, len(test_split))
    print(f"Evaluating 4 models on {total_clips} unseen VoiceBank-DEMAND-16k test clips...")

    # Load Model 2: LinearPerceptronMask
    perceptron = LinearPerceptronMask().eval()
    perc_ckpt = root_dir / "checkpoints" / "perceptron_best.pt"
    if perc_ckpt.exists():
        perceptron.load_state_dict(torch.load(perc_ckpt, map_location="cpu", weights_only=False)["state_dict"])
    else:
        print(f"WARNING: {perc_ckpt} not found; evaluating untrained LinearPerceptronMask.")

    # Load Model 3: GTCRN FP32 (ONNX preferred, fallback PyTorch)
    fp32_onnx_path = root_dir / "exports" / "gtcrn_fp32.onnx"
    gtcrn_ckpt = root_dir / "checkpoints" / "gtcrn_best.pt"
    fp32_session = make_cpu_ort_session(fp32_onnx_path) if fp32_onnx_path.exists() else None
    gtcrn_torch = None
    if fp32_session is None:
        gtcrn_torch = GTCRNModel().eval()
        if gtcrn_ckpt.exists():
            gtcrn_torch.load_state_dict(torch.load(gtcrn_ckpt, map_location="cpu", weights_only=False)["state_dict"])

    # Load Model 4: GTCRN INT8 ONNX
    int8_onnx_path = root_dir / "exports" / "gtcrn_int8.onnx"
    int8_session = make_cpu_ort_session(int8_onnx_path) if int8_onnx_path.exists() else None
    if int8_session is None:
        raise FileNotFoundError(
            f"Quantized ONNX model not found at {int8_onnx_path}. Run scripts/export_onnx.py first."
        )

    model_keys = ["noisy_input", "wiener_filter", "linear_perceptron", "gtcrn_fp32", "gtcrn_int8_onnx"]
    records: Dict[str, List[Dict[str, Optional[float]]]] = {k: [] for k in model_keys}
    infer_times: Dict[str, float] = {k: 0.0 for k in model_keys}
    total_audio_sec = 0.0

    demo_dir = root_dir / "artifacts" / "demos"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_idx = min(max(args.demo_idx, 0), total_clips - 1)

    for idx in range(total_clips):
        item = test_split[idx]
        noisy = extract_audio_1d(item["noisy"])
        clean = extract_audio_1d(item["clean"])
        min_len = min(len(noisy), len(clean))
        noisy, clean = noisy[:min_len], clean[:min_len]
        total_audio_sec += min_len / float(SAMPLE_RATE)

        # 0. Noisy reference
        records["noisy_input"].append(evaluate_clip_metrics(noisy, noisy, clean))

        # 1. Classical Wiener Filter
        t0 = time.perf_counter()
        enh_wiener = wiener_denoise(noisy, sr=SAMPLE_RATE)
        infer_times["wiener_filter"] += time.perf_counter() - t0
        records["wiener_filter"].append(evaluate_clip_metrics(enh_wiener, noisy, clean))

        # 2. Linear Perceptron Mask
        t0 = time.perf_counter()
        enh_perc = run_torch_model_cpu(perceptron, noisy)
        infer_times["linear_perceptron"] += time.perf_counter() - t0
        records["linear_perceptron"].append(evaluate_clip_metrics(enh_perc, noisy, clean))

        # 3. GTCRN FP32
        t0 = time.perf_counter()
        if fp32_session is not None:
            enh_fp32 = run_onnx_model_cpu(fp32_session, noisy)
        else:
            enh_fp32 = run_torch_model_cpu(gtcrn_torch, noisy)
        infer_times["gtcrn_fp32"] += time.perf_counter() - t0
        records["gtcrn_fp32"].append(evaluate_clip_metrics(enh_fp32, noisy, clean))

        # 4. GTCRN INT8 ONNX
        t0 = time.perf_counter()
        enh_int8 = run_onnx_model_cpu(int8_session, noisy)
        infer_times["gtcrn_int8_onnx"] += time.perf_counter() - t0
        records["gtcrn_int8_onnx"].append(evaluate_clip_metrics(enh_int8, noisy, clean))

        if idx == demo_idx:
            sf.write(demo_dir / "sample_noisy.wav", noisy, SAMPLE_RATE)
            sf.write(demo_dir / "sample_wiener.wav", enh_wiener, SAMPLE_RATE)
            sf.write(demo_dir / "sample_gtcrn_int8.wav", enh_int8, SAMPLE_RATE)
            sf.write(demo_dir / "sample_clean.wav", clean, SAMPLE_RATE)

        if (idx + 1) % 100 == 0 or (idx + 1) == total_clips:
            print(f"  Evaluated {idx + 1}/{total_clips} clips...")

    report = {
        "dataset": "JacobLinCool/VoiceBank-DEMAND-16k (unseen test split)",
        "num_clips": total_clips,
        "total_audio_sec": total_audio_sec,
        "demo_clip_index": demo_idx,
        "models": {
            k: summarize_records(records[k], infer_times[k], total_audio_sec)
            for k in model_keys
        },
    }

    report_path = root_dir / "eval" / "benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved benchmark report to {report_path}")
    print(f"Saved 4 paired demo WAVs to {demo_dir}")
    for k, m in report["models"].items():
        print(
            f"  {k:20s} | PESQ={m['pesq_wb']:.3f} | STOI={m['stoi']:.3f} | "
            f"SI-SDR+={m['si_sdr_gain_db']:+.2f} dB | SNR+={m['snr_gain_db']:+.2f} dB | "
            f"r={m['pearson_r']:.3f} | RTF={m['cpu_rtf']:.4f}"
        )


if __name__ == "__main__":
    main()
