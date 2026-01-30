#!/usr/bin/env python3
"""
Plot Task A remaining-time curve (GT vs prediction) for a single video.

Figure 1:
- x: elapsed surgery time (seconds or minutes)
- y: remaining time (seconds or minutes)
- lines: GT vs prediction

Assumes:
- labels npz contains keys: "T_1fps" and "rt_current" (shape (T,) or (T,1))
- features file is the same one you use in feat_seq mode (e.g., .npy/.npz as your TaskADataset expects)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from datasets_taskA import TaskADataset
from models_taskA import FeatLSTMTime


@torch.no_grad()
def predict_one_video(
    npz_path: Path,
    feat_path: Path,
    ckpt_path: Path,
    device: torch.device,
    seq_len: int,
    time_feat: str,
    tnorm_div: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pred: (N,) float32 in seconds, aligned to times t = (seq_len-1) .. (T-1)
      gt:   (N,) float32 in seconds, same alignment
    """
    d = np.load(npz_path, allow_pickle=False)
    T = int(np.asarray(d["T_1fps"]).reshape(-1)[0])

    # --- GT remaining time in seconds ---
    gt_full = d["rt_current"].astype(np.float32).reshape(-1)[:T]

    # --- dataset that yields sliding windows aligned to each time t ---
    ds = TaskADataset(
        npz_path=npz_path,
        mode="feat_seq",
        seq_len=seq_len,
        time_feat=time_feat,
        tnorm_div=tnorm_div,
        feat_path=feat_path,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- infer dims from dataset sample (safe) ---
    feat_seq0, time_seq0, _, _ = ds[0]
    feat_dim = int(feat_seq0.shape[-1])
    time_dim = int(time_seq0.shape[-1])

    # --- model (match training hyperparams) ---
    model: nn.Module = FeatLSTMTime(
        feat_dim=feat_dim,
        time_dim=time_dim,
        hidden=128,
        layers=1,
        out_dim=15,
        dropout=0.4,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    preds = []
    for batch in dl:
        feat_seq, time_seq, _, _ = batch  # (B,L,512), (B,L,Ft)
        feat_seq = feat_seq.to(device, non_blocking=True)
        time_seq = time_seq.to(device, non_blocking=True)

        yhat = model(feat_seq, time_seq)   # (B,15) log1p space
        yhat_rt_log = yhat[:, 0]           # rt_current channel
        preds.append(yhat_rt_log.detach().cpu().numpy())

    pred_log = np.concatenate(preds, axis=0).astype(np.float32)
    pred = np.expm1(np.maximum(pred_log, 0.0)).astype(np.float32)  # log1p -> seconds

    # --- ALIGNMENT ---
    # ds produces predictions for t = (seq_len-1) .. (T-1)
    t0 = seq_len - 1
    gt = gt_full[t0:T]                 # (T - t0,)
    pred = pred[: len(gt)]             # match length exactly (safety)

    return pred, gt


def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return y
    k = int(k)
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(ypad, w, mode="valid").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True, help="labels npz for one video (e.g., xxx_labels.npz)")
    ap.add_argument("--feat", type=Path, required=True, help="feature file for the same video (as TaskADataset expects)")
    ap.add_argument("--ckpt", type=Path, required=True, help="model checkpoint (.pt)")
    ap.add_argument("--out_png", type=Path, required=True, help="output figure path (.png)")
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--time_feat", type=str, default="u_u2_tnorm")
    ap.add_argument("--tnorm_div", type=float, default=6000.0, help="normalization divisor used during training")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--minutes", action="store_true", help="plot in minutes instead of seconds")
    ap.add_argument("--smooth_k", type=int, default=1, help="optional moving average window (e.g., 11). 1 disables.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred, gt = predict_one_video(
        npz_path=args.npz,
        feat_path=args.feat,
        ckpt_path=args.ckpt,
        device=device,
        seq_len=args.seq_len,
        time_feat=args.time_feat,
        tnorm_div=args.tnorm_div,
        batch_size=args.batch_size,
    )

    if args.smooth_k > 1:
        pred_plot = moving_average(pred, args.smooth_k)
        gt_plot = moving_average(gt, args.smooth_k)
    else:
        pred_plot, gt_plot = pred, gt

    # x-axis aligned to real elapsed time (starts at t0 = seq_len-1)
    t0 = args.seq_len - 1
    N = min(len(gt_plot), len(pred_plot))
    gt_plot = gt_plot[:N]
    pred_plot = pred_plot[:N]

    x = (np.arange(N, dtype=np.float32) + t0)  # seconds at 1fps

    if args.minutes:
        x = x / 60.0
        pred_plot = pred_plot / 60.0
        gt_plot = gt_plot / 60.0
        xlabel = "Elapsed time (min)"
        ylabel = "Remaining time (min)"
    else:
        xlabel = "Elapsed time (s)"
        ylabel = "Remaining time (s)"

    plt.figure(figsize=(7.0, 3.2))
    plt.plot(x, gt_plot, label="GT")
    plt.plot(x, pred_plot, label="Prediction", linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=200)
    print(f"[OK] saved: {args.out_png}")


if __name__ == "__main__":
    main()
