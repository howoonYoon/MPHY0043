#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from datasets_taskB import TOOL_COLS, Cholec80ToolDataset, Cholec80ToolDatasetWithTimePred, Cholec80ToolDatasetWithOracleTime
from models_taskB import ResNet18ToolDet
from sklearn.metrics import average_precision_score, roc_auc_score


PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderRetraction",
    "CleaningCoagulation",
    "GallbladderPackaging",
]

def paired_bootstrap_video_pvalue(
    y: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    vid: np.ndarray,
    metric_key: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Paired bootstrap at the surgery/video level.
    Resample videos with replacement, then include all frames from sampled videos.
    H1: B > A (one-sided).
    """
    rng = np.random.default_rng(seed)
    vids = np.array(sorted(set(vid.tolist())))
    diffs = np.empty(n_boot, dtype=np.float64)

    # precompute indices per vid for speed
    idx_map = {v: np.where(vid == v)[0] for v in vids}

    for k in range(n_boot):
        sampled_vids = rng.choice(vids, size=len(vids), replace=True)
        idx = np.concatenate([idx_map[v] for v in sampled_vids], axis=0)

        m_a = compute_metrics(y[idx], prob_a[idx])[metric_key]
        m_b = compute_metrics(y[idx], prob_b[idx])[metric_key]
        diffs[k] = m_b - m_a

    mean = float(diffs.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5]).tolist()
    p = float((diffs <= 0).mean())
    return {"metric": metric_key, "delta_mean": mean, "ci95": [float(lo), float(hi)], "p_one_sided": p}



def ap_per_class(y: np.ndarray, prob: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(TOOL_COLS):
        if y[:, i].sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(average_precision_score(y[:, i], prob[:, i]))
    return out

def macro_f1_at_threshold(y: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> float:
    """
    Macro F1 computed only over tools that have at least one positive label in y.
    This matches the NaN-exclusion behavior used for macro AUPRC (mAP).
    """
    pred = (prob >= thr).astype(np.float32)
    eps = 1e-8

    tp = (pred * y).sum(axis=0)
    fp = (pred * (1 - y)).sum(axis=0)
    fn = ((1 - pred) * y).sum(axis=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    has_pos = (y.sum(axis=0) > 0)
    if not np.any(has_pos):
        return float("nan")
    return float(np.mean(f1[has_pos]))


def compute_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    """Return the key scalar metrics used in the report."""
    ap_per_tool = ap_per_class(y, prob)
    mAP = float(np.nanmean(list(ap_per_tool.values())))
    micro_auprc = float(average_precision_score(y.ravel(), prob.ravel()))
    macro_f1 = macro_f1_at_threshold(y, prob, thr=0.5)


    return {
        "mAP(AUPRC_macro)": mAP,
        "AUPRC_micro": micro_auprc,
        "macro_f1@0.5": macro_f1,
    }


def paired_bootstrap_pvalue(
    y: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric_key: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Paired bootstrap on frames (with replacement).
    Returns mean delta, 95% CI, and one-sided p-value for H1: B > A.
    """
    rng = np.random.default_rng(seed)
    N = y.shape[0]
    diffs = np.empty(n_boot, dtype=np.float64)

    for k in range(n_boot):
        idx = rng.integers(0, N, size=N)  # sample frames with replacement
        m_a = compute_metrics(y[idx], prob_a[idx])[metric_key]
        m_b = compute_metrics(y[idx], prob_b[idx])[metric_key]
        diffs[k] = m_b - m_a

    mean = float(diffs.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5]).tolist()
    # one-sided p-value: probability that improvement <= 0
    p = float((diffs <= 0).mean())
    return {"metric": metric_key, "delta_mean": mean, "ci95": [float(lo), float(hi)], "p_one_sided": p}



# -------------------------
# Load timepreds (pred + valid)
# -------------------------
def load_timepreds(timepred_root: Path, split_json: Path, split: str) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load time predictions from:
      timepred_root/{split}/{vid}.pt

    Supports:
      - dict with {"pred": Tensor(T,F), "valid": BoolTensor(T,)}
      - Tensor(T,F)  (then valid is all True)
    """
    split_map = json.loads(Path(split_json).read_text())
    vids = split_map[split]

    split_dir = timepred_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")

    out: dict[str, dict[str, torch.Tensor]] = {}
    for vid in vids:
        p = split_dir / f"{vid}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing timepred file: {p}")

        obj = torch.load(p, map_location="cpu")

        if isinstance(obj, dict) and "pred" in obj and "valid" in obj:
            pred = obj["pred"].to(dtype=torch.float32)
            valid = obj["valid"].to(dtype=torch.bool)
        elif torch.is_tensor(obj):
            pred = obj.to(dtype=torch.float32)
            valid = torch.ones((pred.shape[0],), dtype=torch.bool)
        else:
            raise RuntimeError(f"{p} must be Tensor or dict(pred, valid)")

        out[vid] = {"pred": pred, "valid": valid}
    return out






# -------------------------
# Model: image + timefeat
# -------------------------
class ResNet18ToolDetWithTimePred(nn.Module):
    def __init__(self, time_feat_dim: int, num_tools: int = 7, pretrained: bool = True, dropout: float = 0.2, time_emb_dim: int = 64):
        super().__init__()
        if pretrained:
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            m = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.img_dim = m.fc.in_features
        self.time_emb = nn.Sequential(
            nn.LayerNorm(time_feat_dim),
            nn.Linear(time_feat_dim, time_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(self.img_dim + time_emb_dim, self.img_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.img_dim, num_tools),
        )

    def forward(self, x: torch.Tensor, tfeat: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x).flatten(1)
        t = self.time_emb(tfeat)
        z = torch.cat([f, t], dim=1)
        return self.head(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cholec80_dir", type=str, required=True)
    ap.add_argument("--labels_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out_json", type=str, default="eval_taskB.json")

    # unified timed option
    ap.add_argument(
        "--time_source",
        type=str,
        default="none",
        choices=["none", "pred", "oracle"],
        help="none: baseline, pred: use saved timepreds, oracle: build GT time features from labels npz on-the-fly"
    )
    ap.add_argument("--timepred_root", type=str, default="", help="required if time_source=pred")
    ap.add_argument("--time_emb_dim", type=int, default=64)
    
    ap.add_argument("--dump_npz", type=str, default="", help="save y/prob arrays to .npz for stats")
    ap.add_argument("--compare_npz", type=str, default="", help="paired bootstrap vs another .npz dump")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=0)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cholec80_dir = Path(args.cholec80_dir)
    labels_dir = Path(args.labels_dir)
    split_json = Path(args.split_json)

    use_time = (args.time_source != "none")
    use_pred = (args.time_source == "pred")
    use_oracle = (args.time_source == "oracle")

    # dataset
    if use_pred:
        if not args.timepred_root:
            raise ValueError("--timepred_root is required when time_source=pred")
        preds = load_timepreds(Path(args.timepred_root), split_json, args.split)
        ds = Cholec80ToolDatasetWithTimePred(
            cholec80_dir, labels_dir, split_json, args.split,
            timepreds=preds, stride=args.stride, transform=tfm
        )
    elif use_oracle:
        # oracle IO 많음: 기본값 workers=4면 느리거나 문제가 날 수 있음.
        # 필요하면 CLI로 --num_workers 0 줘.
        ds = Cholec80ToolDatasetWithOracleTime(
            cholec80_dir, labels_dir, split_json, args.split,
            stride=args.stride, transform=tfm
        )
    else:
        ds = Cholec80ToolDataset(cholec80_dir, labels_dir, split_json, args.split, stride=args.stride, transform=tfm)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # model
    if use_time:
        sample = ds[0]
        time_feat_dim = int(sample[3].shape[-1])
        model = ResNet18ToolDetWithTimePred(
            time_feat_dim=time_feat_dim, num_tools=7, pretrained=False, dropout=0.2, time_emb_dim=args.time_emb_dim
        )
    else:
        model = ResNet18ToolDet(num_tools=7, pretrained=False)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_logits, all_y, all_phase, all_vid = [], [], [], []

    with torch.no_grad():
        for batch in dl:
            if use_time:
                x, y, phase_id, tfeat, _, tvalid, vid = batch
                all_vid.extend(list(vid))
                x = x.to(device, non_blocking=True)
                tfeat = tfeat.to(device, non_blocking=True)
                tvalid = tvalid.to(device, non_blocking=True).float().unsqueeze(1)  # (B,1)
                tfeat = tfeat * tvalid
                logits = model(x, tfeat).detach().cpu()
            else:
                x, y, phase_id, vid = batch
                all_vid.extend(list(vid))

                x = x.to(device, non_blocking=True)
                logits = model(x).detach().cpu()

            all_logits.append(logits)
            all_y.append(y)
            all_phase.append(phase_id)

    logits = torch.cat(all_logits, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    phase = torch.cat(all_phase, dim=0).numpy()

    prob = 1 / (1 + np.exp(-logits))

    # --- optional dump for statistical testing ---
    if args.dump_npz:
        Path(args.dump_npz).parent.mkdir(parents=True, exist_ok=True)
        vid_arr = np.array(all_vid, dtype=str)
        np.savez_compressed(
            args.dump_npz,
            y=y.astype(np.float32),
            prob=prob.astype(np.float32),
            phase=phase.astype(np.int64),
            vid=vid_arr,
        )


    ap_per_tool = ap_per_class(y, prob)
    mAP = float(np.nanmean(list(ap_per_tool.values())))
    micro_auprc = float(average_precision_score(y.ravel(), prob.ravel()))

    phasewise = {}
    for p in range(7):
        idx = (phase == p)
        if idx.sum() == 0:
            continue
        y_p = y[idx]
        prob_p = prob[idx]
        AP_p = ap_per_class(y_p, prob_p)
        vals = np.array(list(AP_p.values()), dtype=np.float32)
        mAP_p = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

        phasewise[PHASE_NAMES[p]] = {
            "n_samples": int(idx.sum()),
            "mAP(AUPRC_macro)": mAP_p,
            "AP_per_tool": AP_p,
        }

    macro_f1 = macro_f1_at_threshold(y, prob, thr=0.5)

    result = {
        "split": args.split,
        "stride": args.stride,
        "n_samples": int(len(ds)),
        "time_source": args.time_source,
        "mAP(AUPRC_macro)": mAP,
        "AUPRC_micro": micro_auprc,
        "AP_per_tool": ap_per_tool,
        "macro_f1@0.5": macro_f1,
        "phasewise": phasewise,
    }

    
    # --- optional paired bootstrap test against another model ---
    if args.compare_npz:
        other = np.load(args.compare_npz, allow_pickle=False)
        y2 = other["y"]
        prob2 = other["prob"]

        if "vid" not in other.files:
            raise RuntimeError("compare_npz must contain 'vid'. Re-run with --dump_npz after enabling vid saving.")
        vid2 = other["vid"].astype(str)

        vid_this = np.array(all_vid, dtype=str)

        if y2.shape != y.shape:
            raise RuntimeError(f"Shape mismatch: this y={y.shape}, other y={y2.shape}. "
                            f"Make sure split/stride/order are identical.")
        if not np.array_equal(vid2, vid_this):
            raise RuntimeError("Dataset order mismatch: 'vid' arrays differ. Ensure identical split/stride and deterministic ordering.")

        boot = {
            "mAP(AUPRC_macro)": paired_bootstrap_video_pvalue(y, prob2, prob, vid_this, "mAP(AUPRC_macro)", args.n_boot, args.boot_seed),
            "AUPRC_micro":      paired_bootstrap_video_pvalue(y, prob2, prob, vid_this, "AUPRC_micro",      args.n_boot, args.boot_seed),
            "macro_f1@0.5":     paired_bootstrap_video_pvalue(y, prob2, prob, vid_this, "macro_f1@0.5",     args.n_boot, args.boot_seed),
        }
        result["bootstrap_vs_other"] = boot
        result["bootstrap_unit"] = "video"

    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
