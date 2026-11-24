"""Full CLI trainer for NEON Browser Noise-Free Mic on VoiceBank-DEMAND-16k.

Trains:
1. Model 2 (`LinearPerceptronMask`, default 10 epochs) -> `checkpoints/perceptron_best.pt`
2. Model 3 (`GTCRNModel`, default 15 epochs, AdamW lr=2e-3, wd=1e-4, CosineAnnealingLR
   T_max=epochs, eta_min=1e-5, grad clip 3.0) -> `checkpoints/gtcrn_best.pt`

Logs epoch-by-epoch metrics to `eval/training_history.json`.
"""

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader, Dataset

# Allow direct CLI execution from repo root or scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.baselines import LinearPerceptronMask
from model.gtcrn_clean import GTCRNModel
from model.stft import ISTFT, STFT
from training.augment import IndianHouseholdNoiseAugmenter
from training.losses import HybridSpeechLoss, si_sdr

SAMPLE_RATE = 16000
CROP_SAMPLES = 32000  # 2.0 s @ 16 kHz


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_audio_1d(field: Any) -> np.ndarray:
    """Safely extract 1D float32 waveform from HuggingFace datasets Audio entry."""
    if isinstance(field, dict):
        if "array" in field and field["array"] is not None:
            wav = np.asarray(field["array"], dtype=np.float32)
        elif "bytes" in field and field["bytes"] is not None:
            wav, _ = sf.read(io.BytesIO(field["bytes"]), dtype="float32")
        elif "path" in field and field["path"]:
            wav, _ = sf.read(field["path"], dtype="float32")
        else:
            raise ValueError(f"Unsupported audio dict keys: {list(field.keys())}")
    elif hasattr(field, "get_all_samples"):
        wav = field.get_all_samples().data.squeeze().cpu().numpy().astype(np.float32)
    else:
        wav = np.asarray(field, dtype=np.float32)

    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav.reshape(-1).astype(np.float32)


class VoiceBankDemandDataset(Dataset):
    """VoiceBank-DEMAND-16k dataset wrapper with 2.0 s random crops and Indian noise augmentation."""

    def __init__(
        self,
        hf_split: Any,
        crop_samples: int = CROP_SAMPLES,
        train: bool = True,
        augmenter: IndianHouseholdNoiseAugmenter = None,
    ) -> None:
        self.hf_split = hf_split
        self.crop_samples = crop_samples
        self.train = train
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.hf_split[idx]
        noisy = extract_audio_1d(item["noisy"])
        clean = extract_audio_1d(item["clean"])

        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        if self.train and self.crop_samples > 0:
            if min_len >= self.crop_samples:
                start = random.randint(0, min_len - self.crop_samples)
                noisy = noisy[start : start + self.crop_samples]
                clean = clean[start : start + self.crop_samples]
            else:
                pad_len = self.crop_samples - min_len
                noisy = np.pad(noisy, (0, pad_len), mode="constant")
                clean = np.pad(clean, (0, pad_len), mode="constant")

            if self.augmenter is not None:
                noisy = self.augmenter(noisy)

        return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy())


def load_voicebank_splits():
    """Load all 11,572 train clips and 824 unseen test clips from JacobLinCool/VoiceBank-DEMAND-16k."""
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    # Disable automatic torchcodec decoding so soundfile reads raw bytes cleanly across datasets>=4.x
    for split_name in ("train", "test"):
        if split_name in ds:
            ds[split_name] = ds[split_name].cast_column("noisy", Audio(decode=False))
            ds[split_name] = ds[split_name].cast_column("clean", Audio(decode=False))
    return ds["train"], ds["test"]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    stft_mod: STFT,
    istft_mod: ISTFT,
    criterion: HybridSpeechLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_sisdr = 0.0
    total_sisdr_gain = 0.0
    n_clips = 0

    for noisy, clean in val_loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        length = clean.shape[-1]

        noisy_spec = stft_mod(noisy)
        enh_spec = model(noisy_spec)
        enh_wav = istft_mod(enh_spec, length=length)

        loss = criterion(enh_wav, clean)
        enh_sisdr = si_sdr(enh_wav, clean)
        noisy_sisdr = si_sdr(noisy, clean)

        bsz = noisy.shape[0]
        total_loss += float(loss.item()) * bsz
        total_sisdr += float(enh_sisdr.sum().item())
        total_sisdr_gain += float((enh_sisdr - noisy_sisdr).sum().item())
        n_clips += bsz

    return {
        "val_loss": total_loss / max(n_clips, 1),
        "val_si_sdr_db": total_sisdr / max(n_clips, 1),
        "val_si_sdr_gain_db": total_sisdr_gain / max(n_clips, 1),
    }


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    ckpt_path: Path,
    device: torch.device,
) -> List[Dict[str, float]]:
    stft_mod = STFT().to(device)
    istft_mod = ISTFT().to(device)
    criterion = HybridSpeechLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-5
    )

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []

    print(
        f"\n=== Training {model_name} ({sum(p.numel() for p in model.parameters()):,} params) "
        f"for {epochs} epochs on {device} ==="
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_sisdr = 0.0
        n_samples = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            length = clean.shape[-1]

            optimizer.zero_grad(set_to_none=True)
            noisy_spec = stft_mod(noisy)
            enh_spec = model(noisy_spec)
            enh_wav = istft_mod(enh_spec, length=length)

            loss, comps = criterion(enh_wav, clean, return_components=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            bsz = noisy.shape[0]
            running_loss += comps["total"] * bsz
            running_sisdr += comps["si_sdr_db"] * bsz
            n_samples += bsz

        scheduler.step()
        train_loss = running_loss / max(n_samples, 1)
        train_sisdr = running_sisdr / max(n_samples, 1)

        val_metrics = evaluate_model(model, val_loader, stft_mod, istft_mod, criterion, device)
        elapsed = time.time() - t0
        cur_lr = float(optimizer.param_groups[0]["lr"])

        record = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": train_loss,
            "train_si_sdr_db": train_sisdr,
            "val_loss": val_metrics["val_loss"],
            "val_si_sdr_db": val_metrics["val_si_sdr_db"],
            "val_si_sdr_gain_db": val_metrics["val_si_sdr_gain_db"],
            "epoch_sec": elapsed,
        }
        history.append(record)

        improved = val_metrics["val_loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["val_loss"]
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": model_name,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_si_sdr_db": val_metrics["val_si_sdr_db"],
                    "val_si_sdr_gain_db": val_metrics["val_si_sdr_gain_db"],
                    "param_count": sum(p.numel() for p in model.parameters()),
                },
                ckpt_path,
            )

        star = " *BEST*" if improved else ""
        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} (SI-SDR={train_sisdr:.2f} dB) | "
            f"val_loss={val_metrics['val_loss']:.4f} (SI-SDR={val_metrics['val_si_sdr_db']:.2f} dB, "
            f"gain={val_metrics['val_si_sdr_gain_db']:+.2f} dB) | "
            f"lr={cur_lr:.2e} | {elapsed:.1f}s{star}"
        )

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NEON Perceptron & GTCRN models.")
    parser.add_argument("--epochs", type=int, default=15, help="GTCRN training epochs (default: 15)")
    parser.add_argument(
        "--perceptron-epochs",
        type=int,
        default=10,
        help="LinearPerceptronMask training epochs (default: 10)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-3, help="Initial AdamW learning rate (default: 2e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--grad-clip", type=float, default=3.0, help="Gradient clipping norm (default: 3.0)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--skip-perceptron", action="store_true", help="Skip Model 2 Perceptron training")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=str(REPO_ROOT),
        help="Project root directory containing checkpoints/ and eval/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = Path(args.root_dir)

    print("Loading JacobLinCool/VoiceBank-DEMAND-16k...")
    train_split, test_split = load_voicebank_splits()
    print(f"Loaded {len(train_split)} train clips and {len(test_split)} unseen test clips.")

    augmenter = IndianHouseholdNoiseAugmenter(sample_rate=SAMPLE_RATE, p=0.35, seed=args.seed)
    train_ds = VoiceBankDemandDataset(train_split, crop_samples=CROP_SAMPLES, train=True, augmenter=augmenter)
    val_ds = VoiceBankDemandDataset(test_split, crop_samples=0, train=False, augmenter=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    # Full-length clips have variable lengths, so evaluate with batch_size=1
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    history_payload: Dict[str, Any] = {
        "dataset": "JacobLinCool/VoiceBank-DEMAND-16k",
        "train_clips": len(train_split),
        "test_clips": len(test_split),
        "crop_samples": CROP_SAMPLES,
        "batch_size": args.batch_size,
        "device": str(device),
        "perceptron": [],
        "gtcrn": [],
    }

    if not args.skip_perceptron and args.perceptron_epochs > 0:
        perceptron = LinearPerceptronMask().to(device)
        history_payload["perceptron"] = train_model(
            model=perceptron,
            model_name="LinearPerceptronMask",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.perceptron_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            ckpt_path=root_dir / "checkpoints" / "perceptron_best.pt",
            device=device,
        )

    if args.epochs > 0:
        gtcrn = GTCRNModel().to(device)
        history_payload["gtcrn"] = train_model(
            model=gtcrn,
            model_name="GTCRNModel",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            ckpt_path=root_dir / "checkpoints" / "gtcrn_best.pt",
            device=device,
        )

    hist_path = root_dir / "eval" / "training_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"\nSaved training history to {hist_path}")


if __name__ == "__main__":
    main()
