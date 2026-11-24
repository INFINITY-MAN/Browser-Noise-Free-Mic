"""Loop B Replay-Safe Teacher-Student continual fine-tuning for NEON.

Workflow:
1. Scans `monitoring/opt_in_clips/` for opt-in user WAV clips and extracts noise-only
   segments via an energy-based Voice Activity Detector (VAD).
2. Builds a replay-safe training mixture: 30% real user noise mixed with clean VoiceBank
   speech + 70% VoiceBank-DEMAND replay buffer.
3. Fine-tunes `checkpoints/gtcrn_best.pt` (`--epochs 3 --lr 3e-4`) using `HybridSpeechLoss`
   plus frozen teacher spectrogram distillation to prevent catastrophic forgetting.
4. Runs a strict regression gate on the 824-clip unseen VoiceBank-DEMAND holdout set:
   ONLY promotes the new checkpoint and versions `model_registry.json` if BOTH PESQ (wb)
   and SI-SDR (dB) strictly improve over the incumbent model.
"""

import argparse
import copy
import datetime
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from pesq import pesq
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.gtcrn_clean import GTCRNModel
from model.stft import ISTFT, STFT
from training.losses import HybridSpeechLoss, si_sdr
from training.train import CROP_SAMPLES, SAMPLE_RATE, extract_audio_1d, load_voicebank_splits, seed_everything


def extract_noise_segments_vad(
    clips_dir: Path,
    frame_len: int = 512,
    hop_len: int = 256,
    energy_percentile: float = 25.0,
    min_noise_samples: int = 8000,
) -> List[np.ndarray]:
    """Extract non-speech noise segments from opt-in clips using frame-energy VAD."""
    noise_bank: List[np.ndarray] = []
    if not clips_dir.exists():
        return noise_bank

    for wav_path in sorted(clips_dir.glob("*.wav")):
        try:
            wav, sr = sf.read(wav_path, dtype="float32")
        except Exception:
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != SAMPLE_RATE or len(wav) < frame_len:
            continue

        # Compute short-time frame energy
        num_frames = 1 + (len(wav) - frame_len) // hop_len
        energies = np.empty(num_frames, dtype=np.float32)
        for i in range(num_frames):
            frame = wav[i * hop_len : i * hop_len + frame_len]
            energies[i] = float(np.mean(frame ** 2))

        threshold = float(np.percentile(energies, energy_percentile))
        noise_frames = [
            wav[i * hop_len : i * hop_len + hop_len]
            for i in range(num_frames)
            if energies[i] <= max(threshold, 1e-8)
        ]
        if noise_frames:
            seg = np.concatenate(noise_frames, axis=0).astype(np.float32)
            if len(seg) >= min_noise_samples:
                noise_bank.append(seg)

    return noise_bank


class ReplaySafeContinualDataset(Dataset):
    """Mixes 30% real opt-in user noise + clean VoiceBank speech with 70% VoiceBank replay."""

    def __init__(
        self,
        train_split: Any,
        noise_bank: List[np.ndarray],
        user_noise_ratio: float = 0.30,
        crop_samples: int = CROP_SAMPLES,
        snr_range_db: Tuple[float, float] = (0.0, 15.0),
    ) -> None:
        self.train_split = train_split
        self.noise_bank = noise_bank
        self.user_noise_ratio = user_noise_ratio
        self.crop_samples = crop_samples
        self.snr_range_db = snr_range_db

    def __len__(self) -> int:
        return len(self.train_split)

    def _sample_noise_crop(self, length: int) -> np.ndarray:
        seg = random.choice(self.noise_bank)
        if len(seg) < length:
            reps = int(np.ceil(length / len(seg)))
            seg = np.tile(seg, reps)
        start = random.randint(0, len(seg) - length)
        return seg[start : start + length].astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.train_split[idx]
        noisy = extract_audio_1d(item["noisy"])
        clean = extract_audio_1d(item["clean"])

        min_len = min(len(noisy), len(clean))
        noisy, clean = noisy[:min_len], clean[:min_len]
        if min_len >= self.crop_samples:
            start = random.randint(0, min_len - self.crop_samples)
            noisy = noisy[start : start + self.crop_samples]
            clean = clean[start : start + self.crop_samples]
        else:
            pad_len = self.crop_samples - min_len
            noisy = np.pad(noisy, (0, pad_len), mode="constant")
            clean = np.pad(clean, (0, pad_len), mode="constant")

        # 30% real user noise + clean speech; 70% VoiceBank-DEMAND replay buffer
        if self.noise_bank and random.random() < self.user_noise_ratio:
            noise = self._sample_noise_crop(self.crop_samples)
            clean_rms = float(np.sqrt(np.mean(clean ** 2)) + 1e-8)
            noise_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-8)
            snr_db = random.uniform(self.snr_range_db[0], self.snr_range_db[1])
            scale = (clean_rms / (10.0 ** (snr_db / 20.0))) / noise_rms
            noisy = np.clip(clean + noise * scale, -1.0, 1.0).astype(np.float32)

        return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy())


@torch.no_grad()
def evaluate_holdout_gate(
    model: GTCRNModel,
    test_split: Any,
    stft_mod: STFT,
    istft_mod: ISTFT,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate PESQ (wb) and SI-SDR (dB) on all 824 unseen VoiceBank-DEMAND holdout clips."""
    model.eval()
    pesq_scores: List[float] = []
    sisdr_scores: List[float] = []

    for i in range(len(test_split)):
        item = test_split[i]
        noisy = extract_audio_1d(item["noisy"])
        clean = extract_audio_1d(item["clean"])
        min_len = min(len(noisy), len(clean))
        noisy, clean = noisy[:min_len], clean[:min_len]

        noisy_t = torch.from_numpy(noisy).unsqueeze(0).to(device)
        enh_spec = model(stft_mod(noisy_t))
        enh = istft_mod(enh_spec, length=min_len).squeeze(0).cpu().numpy()
        enh = np.clip(enh, -1.0, 1.0)

        sisdr_val = float(si_sdr(torch.from_numpy(enh), torch.from_numpy(clean)).item())
        sisdr_scores.append(sisdr_val)

        try:
            p_val = float(pesq(SAMPLE_RATE, clean, enh, "wb"))
            if np.isfinite(p_val):
                pesq_scores.append(p_val)
        except Exception:
            pass

    return {
        "pesq_wb": float(np.mean(pesq_scores)) if pesq_scores else 0.0,
        "si_sdr_db": float(np.mean(sisdr_scores)) if sisdr_scores else 0.0,
        "num_clips": len(test_split),
    }


def update_model_registry(
    registry_path: Path,
    incumbent_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
    promoted: bool,
    version_tag: str,
    checkpoint_relpath: str,
) -> None:
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"active_version": "v1.0.0", "versions": []}

    entry = {
        "version": version_tag,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": checkpoint_relpath,
        "promoted": promoted,
        "incumbent_metrics": incumbent_metrics,
        "candidate_metrics": candidate_metrics,
    }
    registry["versions"].append(entry)
    if promoted:
        registry["active_version"] = version_tag

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loop B Replay-Safe Continual Fine-Tuning.")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Fine-tuning learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.15,
        help="Teacher-student spectrogram L1 distillation weight (default: 0.15)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--root-dir", type=str, default=str(REPO_ROOT), help="Project root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = Path(args.root_dir)

    ckpt_path = root_dir / "checkpoints" / "gtcrn_best.pt"
    clips_dir = root_dir / "monitoring" / "opt_in_clips"
    registry_path = root_dir / "model_registry.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Incumbent checkpoint not found: {ckpt_path}")

    noise_bank = extract_noise_segments_vad(clips_dir)
    print(f"Extracted {len(noise_bank)} noise segments via VAD from {clips_dir}.")

    train_split, test_split = load_voicebank_splits()
    ds = ReplaySafeContinualDataset(train_split, noise_bank=noise_bank, user_noise_ratio=0.30)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    stft_mod = STFT().to(device)
    istft_mod = ISTFT().to(device)
    criterion = HybridSpeechLoss().to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    student = GTCRNModel().to(device)
    student.load_state_dict(ckpt["state_dict"])

    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Evaluating incumbent model on 824-clip unseen holdout gate...")
    incumbent_metrics = evaluate_holdout_gate(teacher, test_split, stft_mod, istft_mod, device)
    print(
        f"Incumbent -> PESQ: {incumbent_metrics['pesq_wb']:.4f}, "
        f"SI-SDR: {incumbent_metrics['si_sdr_db']:.2f} dB"
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        student.train()
        running_loss = 0.0
        n_batches = 0
        for noisy, clean in loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            length = clean.shape[-1]

            optimizer.zero_grad(set_to_none=True)
            noisy_spec = stft_mod(noisy)
            student_spec = student(noisy_spec)
            with torch.no_grad():
                teacher_spec = teacher(noisy_spec)

            enh_wav = istft_mod(student_spec, length=length)
            task_loss = criterion(enh_wav, clean)
            distill_loss = F.l1_loss(student_spec, teacher_spec)
            loss = task_loss + args.distill_weight * distill_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        print(f"[Continual] Epoch {epoch}/{args.epochs} | loss={running_loss / max(n_batches, 1):.4f}")

    print("Evaluating fine-tuned candidate on 824-clip holdout gate...")
    candidate_metrics = evaluate_holdout_gate(student, test_split, stft_mod, istft_mod, device)
    print(
        f"Candidate -> PESQ: {candidate_metrics['pesq_wb']:.4f}, "
        f"SI-SDR: {candidate_metrics['si_sdr_db']:.2f} dB"
    )

    promoted = (
        candidate_metrics["pesq_wb"] > incumbent_metrics["pesq_wb"]
        and candidate_metrics["si_sdr_db"] > incumbent_metrics["si_sdr_db"]
    )
    version_tag = "v" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_ckpt = root_dir / "checkpoints" / f"gtcrn_{version_tag}.pt"

    if promoted:
        torch.save(
            {
                "model_name": "GTCRNModel",
                "version": version_tag,
                "state_dict": student.state_dict(),
                "pesq_wb": candidate_metrics["pesq_wb"],
                "si_sdr_db": candidate_metrics["si_sdr_db"],
                "param_count": student.count_parameters(),
            },
            versioned_ckpt,
        )
        shutil.copy2(versioned_ckpt, ckpt_path)
        print(f"REGRESSION GATE PASSED! Promoted {version_tag} -> {ckpt_path}")
    else:
        print("REGRESSION GATE REJECTED! Incumbent checkpoint retained.")

    update_model_registry(
        registry_path=registry_path,
        incumbent_metrics=incumbent_metrics,
        candidate_metrics=candidate_metrics,
        promoted=promoted,
        version_tag=version_tag,
        checkpoint_relpath=str(versioned_ckpt.relative_to(root_dir)) if promoted else "checkpoints/gtcrn_best.pt",
    )


if __name__ == "__main__":
    main()
