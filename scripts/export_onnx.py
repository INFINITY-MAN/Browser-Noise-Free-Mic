"""Export all 4 GTCRN ONNX models, streaming metadata, weights manifest, and parity checks.

Exports from `checkpoints/gtcrn_best.pt`:
1. `exports/gtcrn_fp32.onnx` & `web/models/gtcrn_fp32.onnx` (batch FP32)
2. `exports/gtcrn_int8.onnx` & `web/models/gtcrn_int8.onnx` (batch QUInt8 dynamic quantized)
3. `exports/gtcrn_stream_fp32.onnx` & `web/models/gtcrn_stream_fp32.onnx` (stateful 1-frame FP32)
4. `exports/gtcrn_stream_int8.onnx` & `web/models/gtcrn_stream_int8.onnx` (stateful 1-frame QUInt8)

Also writes:
- `web/models/stream_io_meta.json`: exact streaming input/output names & shapes for `audioWorker.js`
- `exports/weights_manifest.json`: SHA-256 hashes + exact byte sizes of all `.pt` and `.onnx` files

Verifies numerical parity (`max_abs_diff < 1e-4`) for both batch FP32 ORT vs PyTorch
and 100-frame streaming FP32 ORT vs PyTorch batch.
"""

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.gtcrn_clean import GTCRNModel, GTCRNStreamWrapper


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _onnx_export_legacy(
    model: torch.nn.Module,
    args: Any,
    output_path: Path,
    input_names: list,
    output_names: list,
    dynamic_axes: Dict[str, Dict[int, str]] = None,
    opset_version: int = 17,
) -> None:
    """Export to ONNX using TorchScript exporter (`dynamo=False` when available) for ORT QUInt8 compatibility."""
    kwargs: Dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    if dynamic_axes is not None:
        kwargs["dynamic_axes"] = dynamic_axes
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        kwargs["dynamo"] = False

    torch.onnx.export(model, args, str(output_path), **kwargs)
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def verify_parity(
    model: GTCRNModel,
    batch_onnx_path: Path,
    stream_onnx_path: Path,
    num_frames: int = 100,
    tol: float = 1e-4,
) -> Dict[str, float]:
    """Verify batch FP32 ORT vs PyTorch AND 100-frame streaming FP32 ORT vs PyTorch batch."""
    torch.manual_seed(123)
    x = torch.randn(1, 2, 257, num_frames, dtype=torch.float32)

    with torch.no_grad():
        y_torch = model(x).numpy()

    # 1. Batch FP32 ONNX parity
    sess_batch = ort.InferenceSession(str(batch_onnx_path), providers=["CPUExecutionProvider"])
    y_ort_batch = sess_batch.run(["enh_spec"], {"noisy_spec": x.numpy()})[0]
    batch_diff = float(np.max(np.abs(y_torch - y_ort_batch)))

    # 2. 100-frame streaming FP32 ONNX parity vs PyTorch batch
    sess_stream = ort.InferenceSession(str(stream_onnx_path), providers=["CPUExecutionProvider"])
    in_specs = GTCRNStreamWrapper.INPUT_SPECS
    out_names = [n for n, _ in GTCRNStreamWrapper.OUTPUT_SPECS]
    state_names = [n for n, _ in in_specs if n != "frame_spec"]
    states = {n: np.zeros(shape, dtype=np.float32) for n, shape in in_specs if n != "frame_spec"}

    x_np = x.numpy()
    stream_frames = []
    for t in range(num_frames):
        feed = {"frame_spec": x_np[:, :, :, t : t + 1], **states}
        outs = sess_stream.run(out_names, feed)
        enh_r, enh_i = outs[0], outs[1]
        for idx, s_name in enumerate(state_names):
            states[s_name] = outs[2 + idx]
        stream_frames.append(np.concatenate([enh_r, enh_i], axis=1))

    y_ort_stream = np.concatenate(stream_frames, axis=-1)
    stream_diff = float(np.max(np.abs(y_torch - y_ort_stream)))

    print(f"Parity check (batch FP32 ORT vs PyTorch):          max_abs_diff = {batch_diff:.2e}")
    print(f"Parity check ({num_frames}-frame stream ORT vs PyTorch): max_abs_diff = {stream_diff:.2e}")

    if batch_diff >= tol:
        raise AssertionError(f"Batch ONNX parity failed: {batch_diff:.2e} >= {tol}")
    if stream_diff >= tol:
        raise AssertionError(f"Streaming ONNX parity failed: {stream_diff:.2e} >= {tol}")

    return {"batch_fp32_max_abs_diff": batch_diff, "stream_100f_fp32_max_abs_diff": stream_diff}


def build_weights_manifest(root_dir: Path) -> Dict[str, Any]:
    entries = []
    for sub in ("checkpoints", "exports", "web/models"):
        d = root_dir / sub
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix in (".pt", ".onnx"):
                entries.append(
                    {
                        "path": str(p.relative_to(root_dir)),
                        "bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                    }
                )
    return {"files": entries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NEON GTCRN models to ONNX (FP32 + QUInt8).")
    parser.add_argument("--root-dir", type=str, default=str(REPO_ROOT), help="Project root directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to gtcrn_best.pt (default: <root-dir>/checkpoints/gtcrn_best.pt)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (root_dir / "checkpoints" / "gtcrn_best.pt")

    exports_dir = root_dir / "exports"
    web_models_dir = root_dir / "web" / "models"
    exports_dir.mkdir(parents=True, exist_ok=True)
    web_models_dir.mkdir(parents=True, exist_ok=True)

    model = GTCRNModel().eval()
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Loaded GTCRN weights from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1. Batch FP32 ONNX
    batch_fp32 = exports_dir / "gtcrn_fp32.onnx"
    dummy_batch = torch.randn(1, 2, 257, 16, dtype=torch.float32)
    _onnx_export_legacy(
        model=model,
        args=(dummy_batch,),
        output_path=batch_fp32,
        input_names=["noisy_spec"],
        output_names=["enh_spec"],
        dynamic_axes={"noisy_spec": {0: "batch", 3: "frames"}, "enh_spec": {0: "batch", 3: "frames"}},
        opset_version=args.opset,
    )
    shutil.copy2(batch_fp32, web_models_dir / "gtcrn_fp32.onnx")

    # 2. Batch QUInt8 dynamic quantized ONNX
    batch_int8 = exports_dir / "gtcrn_int8.onnx"
    quantize_dynamic(
        model_input=str(batch_fp32),
        model_output=str(batch_int8),
        weight_type=QuantType.QUInt8,
    )
    shutil.copy2(batch_int8, web_models_dir / "gtcrn_int8.onnx")

    # 3. Stateful 1-frame streaming FP32 ONNX
    streamer = GTCRNStreamWrapper(model).eval()
    stream_fp32 = exports_dir / "gtcrn_stream_fp32.onnx"
    dummy_frame = torch.randn(1, 2, 257, 1, dtype=torch.float32)
    dummy_states = GTCRNStreamWrapper.init_states()
    _onnx_export_legacy(
        model=streamer,
        args=(dummy_frame, *dummy_states),
        output_path=stream_fp32,
        input_names=[n for n, _ in GTCRNStreamWrapper.INPUT_SPECS],
        output_names=[n for n, _ in GTCRNStreamWrapper.OUTPUT_SPECS],
        dynamic_axes=None,
        opset_version=args.opset,
    )
    shutil.copy2(stream_fp32, web_models_dir / "gtcrn_stream_fp32.onnx")

    # 4. Stateful 1-frame streaming QUInt8 dynamic quantized ONNX
    stream_int8 = exports_dir / "gtcrn_stream_int8.onnx"
    quantize_dynamic(
        model_input=str(stream_fp32),
        model_output=str(stream_int8),
        weight_type=QuantType.QUInt8,
    )
    shutil.copy2(stream_int8, web_models_dir / "gtcrn_stream_int8.onnx")

    # 5. Write web/models/stream_io_meta.json
    meta_path = web_models_dir / "stream_io_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(GTCRNStreamWrapper.io_metadata(), f, indent=2)

    # 6. Verify numerical parity (< 1e-4)
    parity = verify_parity(model, batch_fp32, stream_fp32, num_frames=100, tol=1e-4)

    # 7. Write exports/weights_manifest.json
    manifest = build_weights_manifest(root_dir)
    manifest["parity_verification"] = parity
    manifest_path = exports_dir / "weights_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved streaming I/O metadata to {meta_path}")
    print(f"Saved weights manifest to {manifest_path}")
    print("ONNX EXPORT & VERIFICATION COMPLETE!")


if __name__ == "__main__":
    main()
