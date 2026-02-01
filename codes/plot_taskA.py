#!/usr/bin/env python3
"""
Plot Task A remaining-time curve (GT vs prediction) for a single video.

This script:
1) Loads a single video label NPZ containing 1fps targets.
2) Runs a sliding-window prediction using the feat_seq setup (FeatLSTMTime).
3) Plots GT vs predicted rt_current over elapsed time and saves a PNG.

Figure:
- x-axis: elapsed surgery time (seconds or minutes)
- y-axis: remaining time (seconds or minutes)
- curves: GT vs Prediction

Assumptions / Inputs
--------------------
- Label NPZ contains:
    - "T_1fps": number of 1fps frames
    - "rt_current": remaining time of current phase in seconds, shape (T,) or (T,1)
- Feature file corresponds to the same video at 1fps alignment and is compatible
  with TaskADataset(mode="feat_seq") (e.g., .npy of shape (T, Ffeat)).

Outputs
-------
- A PNG figure saved to --out_png.
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
    Run sliding-window inference for one video and return prediction/GT arrays.

    Parameters
    ----------
    npz_path:
        Path to label NPZ for one video (e.g., xxx_labels.npz).
    feat_path:
        Path to cached feature file aligned to the same video at 1fps.
    ckpt_path:
        Path to trained model checkpoint (.pt).
        Supports either {"model": state_dict, ...} or a raw state_dict.
    device:
        Torch device ("cuda" or "cpu").
    seq_len:
        Sliding window length L. Predictions are produced for t >= L-1.
    time_feat:
        Time feature configuration string passed to TaskADataset.
    tnorm_div:
        Normalization divisor used for tnorm feature; must match training.
    batch_size:
        Batch size for inference.

    Returns
    -------
    pred:
        (N,) float32 predicted rt_current in seconds,
        aligned to times t = (seq_len-1) .. (T-1).
    gt:
        (N,) float32 ground-truth rt_current in seconds, same alignment.

    Notes
    -----
    - Model outputs rt_current in log1p space; conversion back to seconds uses expm1.
    - We clip the final seconds to be non-negative (remaining time cannot be < 0).
    """
    d = np.load(npz_path, allow_pickle=False)
    T = int(np.asarray(d["T_1fps"]).reshape(-1)[0])

    # Ground-truth remaining time in seconds
    gt_full = d["rt_current"].astype(np.float32).reshape(-1)[:T]

    # Dataset yields windows aligned to each time t
    ds = TaskADataset(
        npz_path=npz_path,
        mode="feat_seq",
        seq_len=seq_len,
        time_feat=time_feat,
        tnorm_div=tnorm_div,
        feat_path=feat_path,
    )

    pin = (device.type == "cuda")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # Infer input dims from one dataset sample (robust)
    feat_seq0, time_seq0, _, _ = ds[0]
    feat_dim = int(feat_seq0.shape[-1])
    time_dim = int(time_seq0.shape[-1])

    # Model definition MUST match training hyperparams used for the checkpoint
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

    preds_log = []
    for batch in dl:
        feat_seq, time_seq, _, _ = batch  # (B,L,Ffeat), (B,L,Ft)
        feat_seq = feat_seq.to(device, non_blocking=pin)
        time_seq = time_seq.to(device, non_blocking=pin)

        yhat = model(feat_seq, time_seq)   # (B,15) in log1p space
        preds_log.append(yhat[:, 0].detach().cpu().numpy())  # rt_current channel

    pred_log = np.concatenate(preds_log, axis=0).astype(np.float32)

    # Convert log1p -> seconds, then clip in seconds domain (correct behavior)
    pred = np.expm1(pred_log).astype(np.float32)
    pred = np.clip(pred, 0.0, None)

    # Alignment: ds produces predictions for t = (seq_len-1) .. (T-1)
    t0 = seq_len - 1
    gt = gt_full[t0:T]
    pred = pred[: len(gt)]  # safety

    return pred, gt


def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    """
    Centered moving average with edge padding (for visualization only).

    Parameters
    ----------
    y:
        Input 1D array.
    k:
        Window size. If k <= 1, returns y unchanged.

    Returns
    -------
    y_smooth:
        Smoothed 1D array with the same length as y.
    """
    if k <= 1:
        return y
    k = int(k)
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(ypad, w, mode="valid").astype(np.float32)


def main():
    """
    CLI entry point.

    Steps:
      1) Parse args (paths + inference config)
      2) Run predict_one_video() to get (pred, gt)
      3) Optionally smooth for visualization
      4) Build x-axis in seconds (or minutes) aligned to t0=seq_len-1
      5) Plot and save to --out_png
    """
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

    # Optional smoothing for visualization
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

    # Unit conversion
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
    plt.close()
    print(f"[OK] saved: {args.out_png}")


if __name__ == "__main__":
    main()
