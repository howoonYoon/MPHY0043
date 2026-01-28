#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets_tooldet import Cholec80ToolDataset, TOOL_COLS
from models_tooldet import ResNet18ToolDet
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


def ap_per_class(y: np.ndarray, prob: np.ndarray) -> dict:
    """Compute per-tool AP; NaN if no positives for that tool."""
    out = {}
    for i, name in enumerate(TOOL_COLS):
        if y[:, i].sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(average_precision_score(y[:, i], prob[:, i]))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cholec80_dir", type=str, required=True)
    ap.add_argument("--labels_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out_json", type=str, default="eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = Cholec80ToolDataset(
        Path(args.cholec80_dir),
        Path(args.labels_dir),
        Path(args.split_json),
        args.split,
        stride=args.stride,
        transform=tfm,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = ResNet18ToolDet(num_tools=7, pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    all_logits = []
    all_y = []
    all_phase = []

    for x, y, phase_id in dl:
        x = x.to(device, non_blocking=True)
        logits = model(x).cpu()
        all_logits.append(logits)
        all_y.append(y)
        all_phase.append(phase_id)

    logits = torch.cat(all_logits, dim=0).numpy()     # (N,7)
    y = torch.cat(all_y, dim=0).numpy()               # (N,7)
    phase = torch.cat(all_phase, dim=0).numpy()       # (N,)

    prob = 1 / (1 + np.exp(-logits))
    pred = (prob >= 0.5).astype(np.float32)

    # F1@0.5 (optional)
    eps = 1e-8
    tp = (pred * y).sum(axis=0)
    fp = (pred * (1 - y)).sum(axis=0)
    fn = ((1 - pred) * y).sum(axis=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # Overall AP/mAP + micro AUPRC
    AP_overall = ap_per_class(y, prob)
    mAP_overall = float(np.nanmean(list(AP_overall.values())))
    micro_auprc = float(average_precision_score(y.ravel(), prob.ravel()))

    # Optional AUROC (overall)
    AUROC_overall = {}
    for i, name in enumerate(TOOL_COLS):
        if len(np.unique(y[:, i])) < 2:
            AUROC_overall[name] = float("nan")
        else:
            AUROC_overall[name] = float(roc_auc_score(y[:, i], prob[:, i]))
    macro_auroc = float(np.nanmean(list(AUROC_overall.values())))

    # -------- Phase-wise AP / mAP --------
    phasewise = {}
    for p in range(7):
        idx = (phase == p)
        if idx.sum() == 0:
            continue

        y_p = y[idx]
        prob_p = prob[idx]

        AP_p = ap_per_class(y_p, prob_p)
        mAP_p = float(np.nanmean(list(AP_p.values())))

        phasewise[PHASE_NAMES[p]] = {
            "n_samples": int(idx.sum()),
            "mAP(AUPRC_macro)": mAP_p,
            "AP_per_tool": AP_p,
        }

    result = {
        "split": args.split,
        "stride": args.stride,
        "n_samples": int(len(ds)),

        "mAP(AUPRC_macro)": mAP_overall,
        "AUPRC_micro": micro_auprc,
        "AP_per_tool": AP_overall,

        "AUROC_macro": macro_auroc,
        "AUROC_per_tool": AUROC_overall,

        "macro_f1@0.5": float(np.mean(f1)),
        "per_tool_f1@0.5": {TOOL_COLS[i]: float(f1[i]) for i in range(7)},

        "phasewise": phasewise,
    }

    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
