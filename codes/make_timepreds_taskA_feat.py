#!/usr/bin/env python3
"""
Batch-export Task A predictions for multiple videos into per-video .pt files.

What this script does
---------------------
- Loads a trained FeatLSTMTime checkpoint.
- For each video in a specified split (train/val/test):
  1) Builds a TaskADataset in feat_seq mode (sliding windows).
  2) Runs the model to produce predictions for each valid timestep.
  3) Writes a padded full-length tensor (T, 15) plus a boolean valid mask (T,)
     so downstream code can align predictions to the original 1fps timeline.

Output format (.pt)
-------------------
Each output file is a dict:
  {
    "pred":  FloatTensor of shape (T, 15), padded with zeros for early timesteps,
    "valid": BoolTensor  of shape (T,), True where predictions are valid
  }

Alignment convention
--------------------
- feat_seq dataset yields predictions for timesteps t = (seq_len - 1) .. (T - 1)
- The script returns/stores a full-length tensor by placing predictions at:
    pred_full[start_t:] = pred_tail
  and marking:
    valid[start_t:] = True
  where start_t = seq_len - 1.

Prediction space
----------------
- pred_space="log": stores model outputs directly (log1p space).
- pred_space="seconds": converts outputs via expm1() and stores seconds.

Expected naming
---------------
- Labels: {labels_dir}/{vid}_labels.npz
- Features: {feats_dir}/{vid}_{feat_suffix}.npy  (default suffix: resnet18)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets_taskA import TaskADataset
from models_taskA import FeatLSTMTime


@torch.no_grad()
def predict_one_video(
    npz_path: Path,
    feat_path: Path,
    model: nn.Module,
    device: torch.device,
    seq_len: int,
    time_feat: str,
    tnorm_div: float,
    batch_size: int,
    pred_space: str,   # "log" or "seconds"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict Task A targets for a single video and return padded full-length output.

    Parameters
    ----------
    npz_path:
        Path to {vid}_labels.npz containing 1fps timeline length (T_1fps).
    feat_path:
        Path to cached features aligned to 1fps (e.g., {vid}_resnet18.npy).
    model:
        A loaded FeatLSTMTime model with weights already restored.
    device:
        Torch device ("cuda" or "cpu").
    seq_len:
        Sliding window length L used during training/inference.
    time_feat:
        Time feature configuration string (must match dataset implementation).
    tnorm_div:
        Normalization divisor for t_norm feature (must match training).
    batch_size:
        Inference batch size.
    pred_space:
        "log" to return log1p-space predictions,
        "seconds" to convert to seconds via expm1().

    Returns
    -------
    pred_full:
        FloatTensor of shape (T, 15).
        Predictions are placed starting at t = seq_len-1.
        Early timesteps [0, seq_len-2] are padded with zeros.
    valid:
        BoolTensor of shape (T,).
        valid[t] is True iff pred_full[t] contains a real prediction.

    Notes
    -----
    TaskADataset(mode="feat_seq") produces one item per timestep:
      t in [seq_len-1, ..., T-1]
    so the number of predicted rows is (T - (seq_len-1)).
    """
    d = np.load(npz_path, allow_pickle=False)
    T = int(d["T_1fps"][0])

    ds = TaskADataset(
        npz_path=npz_path,
        mode="feat_seq",
        seq_len=seq_len,
        time_feat=time_feat,
        tnorm_div=tnorm_div,
        feat_path=feat_path,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    preds = []
    model.eval()
    for feat_seq, time_seq, _, _ in dl:
        feat_seq = feat_seq.to(device, non_blocking=True)
        time_seq = time_seq.to(device, non_blocking=True)

        pred_log = model(feat_seq, time_seq)  # (B, 15) in log1p space

        if pred_space == "seconds":
            pred = torch.expm1(pred_log)       # convert to seconds
        else:
            pred = pred_log                    # keep log1p-space

        preds.append(pred.detach().cpu())

    if len(preds) == 0:
        raise RuntimeError(f"No predictions for {npz_path.name}: seq_len={seq_len}, T={T}")

    pred_tail = torch.cat(preds, dim=0)  # (T-(L-1), 15)

    # Pad to full length timeline
    start_t = seq_len - 1
    pred_full = torch.zeros((T, pred_tail.shape[1]), dtype=torch.float32)
    valid = torch.zeros((T,), dtype=torch.bool)

    pred_full[start_t:, :] = pred_tail
    valid[start_t:] = True

    return pred_full, valid


def infer_time_dim(sample_npz: Path, sample_feat: Path, seq_len: int, time_feat: str, tnorm_div: float) -> int:
    """
    Infer time feature dimensionality from a dataset sample.

    Parameters
    ----------
    sample_npz:
        NPZ path for one example video.
    sample_feat:
        Feature file path for the same video.
    seq_len:
        Window length used in feat_seq mode.
    time_feat:
        Time feature configuration string.
    tnorm_div:
        Normalization divisor for time features.

    Returns
    -------
    time_dim:
        Integer time feature dimension Ft (last dim of time_seq).
    """
    ds = TaskADataset(
        npz_path=sample_npz,
        mode="feat_seq",
        seq_len=seq_len,
        time_feat=time_feat,
        tnorm_div=tnorm_div,
        feat_path=sample_feat,
    )
    _, time_seq, _, _ = ds[0]
    return int(time_seq.shape[-1])


def load_model_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    time_dim: int,
    feat_dim: int = 512,
    out_dim: int = 15,
) -> nn.Module:
    """
    Load FeatLSTMTime model from a checkpoint and move it to the target device.

    This function tries to infer architecture hyperparameters in a robust way:
    - If ckpt is a dict and contains "args", read hidden/layers/dropout from it.
    - Otherwise (or additionally), infer hidden size and number of layers from
      LSTM weight shapes in the state_dict.

    Parameters
    ----------
    ckpt_path:
        Path to checkpoint (.pt).
        Accepted formats:
          - dict with key "model" containing state_dict
          - plain state_dict
    device:
        Torch device to load the model onto.
    time_dim:
        Time feature dimension Ft.
    feat_dim:
        Visual feature dimension (default 512 for ResNet18 features).
    out_dim:
        Output dimension (default 15 for Task A targets).

    Returns
    -------
    model:
        FeatLSTMTime instance with loaded weights, set to eval() mode.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Defaults (used if nothing can be inferred)
    hidden, layers, dropout = 256, 1, 0.2

    # Try to read from ckpt["args"] if present
    if isinstance(ckpt, dict):
        a = ckpt.get("args", {})
        hidden = int(a.get("hidden", a.get("lstm_hidden", hidden)))
        layers = int(a.get("layers", a.get("num_layers", layers)))
        dropout = float(a.get("dropout", dropout))

    # Infer from state dict if possible
    if isinstance(state, dict) and "lstm.weight_ih_l0" in state:
        hidden = int(state["lstm.weight_ih_l0"].shape[0] // 4)

        layer_ids = []
        for k in state.keys():
            if k.startswith("lstm.weight_ih_l"):
                try:
                    layer_ids.append(int(k.split("lstm.weight_ih_l")[1]))
                except ValueError:
                    pass
        if layer_ids:
            layers = max(layer_ids) + 1

    model = FeatLSTMTime(
        feat_dim=feat_dim,
        time_dim=time_dim,
        hidden=hidden,
        layers=layers,
        out_dim=out_dim,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def feat_path_for_vid(feats_dir: Path, vid: str, suffix: str) -> Path:
    """
    Build feature file path for a given video id.

    Parameters
    ----------
    feats_dir:
        Root directory that stores per-video feature files.
    vid:
        Video id string like "video01".
    suffix:
        Feature suffix string like "resnet18".

    Returns
    -------
    path:
        feats_dir / f"{vid}_{suffix}.npy"
    """
    return feats_dir / f"{vid}_{suffix}.npy"


def main():
    """
    CLI entry point.

    Workflow
    --------
    1) Read split list from --split_json and select --split.
    2) Optionally override seq_len/time_feat/tnorm_div from checkpoint args if present.
    3) Infer time_dim from the first video in the split.
    4) Load FeatLSTMTime model.
    5) For each video:
       - run predict_one_video()
       - save {"pred": ..., "valid": ...} to {out_dir}/{split}/{vid}.pt

    Notes
    -----
    - Use --overwrite to regenerate existing files.
    - Use --pred_space seconds/log depending on what downstream expects.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", type=str, required=True, help="dir with *_labels.npz")
    ap.add_argument("--feats_dir", type=str, required=True, help="dir with {vid}_resnet18.npy")
    ap.add_argument("--feat_suffix", type=str, default="resnet18", help="e.g. resnet18 (file is {vid}_{suffix}.npy)")
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=180)
    ap.add_argument(
        "--time_feat",
        type=str,
        default="u_u2_tnorm",
        choices=[
            "u", "u_u2", "tnorm", "u_tnorm", "u_u2_tnorm",
        ],
    )
    ap.add_argument("--tnorm_div", type=float, default=6000.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--pred_space", type=str, default="seconds", choices=["log", "seconds"])

    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    feats_dir = Path(args.feats_dir)
    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # If checkpoint contains training args, prefer them for consistency
    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    if isinstance(ckpt, dict) and "args" in ckpt:
        ckpt_args = ckpt["args"]
        args.seq_len = int(ckpt_args.get("seq_len", args.seq_len))
        args.time_feat = ckpt_args.get("time_feat", args.time_feat)
        args.tnorm_div = float(ckpt_args.get("tnorm_div", args.tnorm_div))

    split_map = json.loads(Path(args.split_json).read_text())
    vids = split_map[args.split]
    if len(vids) == 0:
        raise RuntimeError(f"No videos found for split={args.split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer time feature dimension from one sample
    sample_vid = vids[0]
    sample_npz = labels_dir / f"{sample_vid}_labels.npz"
    sample_feat = feat_path_for_vid(feats_dir, sample_vid, args.feat_suffix)
    if not sample_feat.exists():
        raise FileNotFoundError(f"Missing feature file: {sample_feat}")

    time_dim = infer_time_dim(sample_npz, sample_feat, args.seq_len, args.time_feat, args.tnorm_div)
    model = load_model_from_ckpt(Path(args.ckpt), device=device, time_dim=time_dim)

    n_done = 0
    for vid in vids:
        out_path = out_dir / f"{vid}.pt"
        if out_path.exists() and not args.overwrite:
            continue

        npz_path = labels_dir / f"{vid}_labels.npz"
        feat_path = feat_path_for_vid(feats_dir, vid, args.feat_suffix)

        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature file for {vid}: {feat_path}")

        pred_full, valid = predict_one_video(
            npz_path=npz_path,
            feat_path=feat_path,
            model=model,
            device=device,
            seq_len=args.seq_len,
            time_feat=args.time_feat,
            tnorm_div=args.tnorm_div,
            batch_size=args.batch_size,
            pred_space=args.pred_space,
        )

        torch.save({"pred": pred_full.cpu(), "valid": valid.cpu()}, out_path)
        n_done += 1

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Saved preds for {n_done} videos to {out_dir}")
    ex = out_dir / f"{vids[0]}.pt"
    obj = torch.load(ex, map_location="cpu")
    print(f"Example {ex.name}: pred_shape={tuple(obj['pred'].shape)} valid_shape={tuple(obj['valid'].shape)}")
    print("valid true count:", int(obj["valid"].sum()))


if __name__ == "__main__":
    main()
