#!/usr/bin/env python3
"""
Task A evaluation script (RST regression) for train/val/test split.

This script loads a trained checkpoint and reports masked MAE in seconds:
  - rt_mae     : MAE for rt_current (target index 0)
  - bounds_mae : MAE for boundary-related targets (target indices 1:)
  - total_mae  : MAE for all targets (0..14) with mask

Supported modes match the training script:
  - time_mlp, time_lstm, time_phase, feat_lstm

Notes
-----
- Targets and model outputs are expected in log-space (log1p). Evaluation converts back using expm1().
- The dataset provides a mask `m` to ignore invalid (already-ended) phase targets.
- Checkpoint can be either:
    (a) a dict with key "model" (your training checkpoints), or
    (b) a raw state_dict (just in case).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from models_taskA import TimeOnlyMLP, TimeOnlyLSTM, FeatLSTMTime
from datasets_taskA import TaskADataset

# Example:
#   python test_taskA.py --mode time_mlp --label_dir /path/labels --split_json /path/split.json --ckpt /path/best.pt
#   python test_taskA.py --mode time_lstm --label_dir /path/labels --split_json /path/split.json --ckpt /path/best.pt
#   python test_taskA.py --mode time_phase --label_dir /path/labels --split_json /path/split.json --ckpt /path/best.pt
#   python test_taskA.py --mode feat_lstm --label_dir /path/labels --split_json /path/split.json --feat_dir /path/feats --ckpt /path/best.pt

MODES = ("time_mlp", "time_lstm", "time_phase", "feat_lstm")


def forward_adapter(
    model: nn.Module,
    batch: Any,
    device: torch.device,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize per-mode batch/model signatures into a common output.

    Parameters
    ----------
    model:
        The model for the chosen mode.
    batch:
        One minibatch from the DataLoader. Structure depends on mode:
          - time_mlp   : (x, y_log, m)
          - time_lstm  : (x_seq, y_log, m)
          - time_phase : (x_seq, y_log, m)  # x_seq may include phase-related features if enabled
          - feat_lstm  : (feat_seq, time_seq, y_log, m)
    device:
        Device to move tensors to.
    mode:
        One of MODES.

    Returns
    -------
    pred_log:
        Predicted targets in log-space, shape (B, 15).
    y_log:
        Ground-truth targets in log-space, shape (B, 15).
    m:
        Mask tensor indicating valid targets, shape (B, 15).
    """
    if mode == "time_mlp":
        x, y_log, m = batch
        x = x.to(device)
        y_log = y_log.to(device)
        m = m.to(device)
        pred_log = model(x)
        return pred_log, y_log, m

    if mode in ("time_lstm", "time_phase"):
        x_seq, y_log, m = batch
        x_seq = x_seq.to(device)
        y_log = y_log.to(device)
        m = m.to(device)
        pred_log = model(x_seq)
        return pred_log, y_log, m

    if mode == "feat_lstm":
        feat_seq, time_seq, y_log, m = batch
        feat_seq = feat_seq.to(device)
        time_seq = time_seq.to(device)
        y_log = y_log.to(device)
        m = m.to(device)
        pred_log = model(feat_seq, time_seq)
        return pred_log, y_log, m

    raise ValueError(f"Unknown mode: {mode}")


@torch.no_grad()
def eval_mae_seconds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> Tuple[float, float, float]:
    """
    Compute masked MAE in seconds on the provided loader.

    Returns
    -------
    rt_mae:
        Mean absolute error for rt_current (target index 0).
    bounds_mae:
        Mean absolute error for boundary targets (indices 1:), masked by m[:, 1:].
    total_mae:
        Mean absolute error for all targets (0..14), masked by m.

    Implementation details
    ----------------------
    - Model outputs/labels are log1p-space; convert back with expm1().
    - Mask `m` prevents ended phases from contributing to bounds/total MAE.
    """
    model.eval()

    rt_abs, bounds_abs, bounds_mask = [], [], []
    total_abs, total_mask = [], []

    for batch in loader:
        pred_log, y_log, m = forward_adapter(model, batch, device, mode)

        # Convert from log-space back to seconds
        pred = torch.expm1(pred_log)
        y = torch.expm1(y_log)
        abs_err = torch.abs(pred - y)

        # rt_current error (index 0)
        rt_abs.append(abs_err[:, 0].detach().cpu())

        # boundary errors (indices 1:) with masking
        b_err = abs_err[:, 1:] * m[:, 1:]
        bounds_abs.append(b_err.detach().cpu())
        bounds_mask.append(m[:, 1:].detach().cpu())

        # total error (all dims) with masking
        total_abs.append((abs_err * m).detach().cpu())
        total_mask.append(m.detach().cpu())

    rt_mae = torch.cat(rt_abs).mean().item()

    b_err_all = torch.cat(bounds_abs, dim=0)
    b_m_all = torch.cat(bounds_mask, dim=0)
    bounds_mae = (b_err_all.sum() / b_m_all.sum().clamp_min(1.0)).item()

    t_err_all = torch.cat(total_abs, dim=0)
    t_m_all = torch.cat(total_mask, dim=0)
    total_mae = (t_err_all.sum() / t_m_all.sum().clamp_min(1.0)).item()

    return rt_mae, bounds_mae, total_mae


def build_test_loader(args) -> DataLoader:
    """
    Build a DataLoader for the requested split (train/val/test).

    The split file is expected to be JSON with keys: "train", "val", "test",
    and values are lists of video IDs. For each video ID `vid`, labels live at:
      {label_dir}/{vid}_labels.npz

    Returns
    -------
    dl_test:
        DataLoader over concatenated per-video datasets.
    """
    label_dir = Path(args.label_dir)
    split = json.loads(Path(args.split_json).read_text())

    split_name = args.split
    if split_name not in split:
        raise KeyError(f"Split '{split_name}' not found in split_json. Available: {list(split.keys())}")

    test_vids = split[split_name]

    # Build per-video datasets depending on mode
    if args.mode == "time_mlp":
        test_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="time_point",
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
            )
            for vid in test_vids
        ]

    elif args.mode == "time_lstm":
        test_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="time_seq",
                seq_len=args.seq_len,
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
            )
            for vid in test_vids
        ]

    elif args.mode == "time_phase":
        test_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="timephase_seq",
                seq_len=args.seq_len,
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
                use_phase_onehot=args.use_phase_onehot,
                use_elapsed_phase=args.use_elapsed_phase,
                n_phases=args.n_phases,
            )
            for vid in test_vids
        ]

    elif args.mode == "feat_lstm":
        feat_dir = Path(args.feat_dir)
        test_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="feat_seq",
                seq_len=args.seq_len,
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
                feat_path=feat_dir / f"{vid}_resnet18.npy",
            )
            for vid in test_vids
        ]

    else:
        raise ValueError(args.mode)

    # Concat videos so a single DataLoader can iterate over them
    ds_test = ConcatDataset(test_sets)
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return dl_test


def build_model(args, sample_batch: Any) -> nn.Module:
    """
    Instantiate a model and infer input dimensions from `sample_batch`.
    This avoids hardcoding feature dims and keeps evaluation robust.

    Parameters
    ----------
    sample_batch:
        One minibatch from the loader; its tensor shapes depend on mode.
    """
    if args.mode == "time_mlp":
        x, _, _ = sample_batch
        in_dim = int(x.shape[-1])
        return TimeOnlyMLP(in_dim=in_dim, out_dim=15, hidden=args.hidden)

    if args.mode == "time_lstm":
        x_seq, _, _ = sample_batch
        in_dim = int(x_seq.shape[-1])
        return TimeOnlyLSTM(in_dim=in_dim, hidden=args.hidden, num_layers=args.layers, out_dim=15, dropout=args.dropout)

    if args.mode == "time_phase":
        # Same LSTM architecture; dataset packs extra phase features into x_seq if enabled.
        x_seq, _, _ = sample_batch
        in_dim = int(x_seq.shape[-1])
        return TimeOnlyLSTM(in_dim=in_dim, hidden=args.hidden, num_layers=args.layers, out_dim=15, dropout=args.dropout)

    if args.mode == "feat_lstm":
        feat_seq, time_seq, _, _ = sample_batch
        feat_dim = int(feat_seq.shape[-1])
        time_dim = int(time_seq.shape[-1])
        return FeatLSTMTime(feat_dim=feat_dim, time_dim=time_dim, hidden=args.hidden, layers=args.layers, out_dim=15, dropout=args.dropout)

    raise ValueError(args.mode)


def apply_mode_defaults(args) -> None:
    """
    Fill mode-specific defaults for evaluation.

    These defaults mirror the training setup so evaluation works out-of-the-box
    without specifying every hyperparameter from the CLI.
    """
    if args.mode == "time_mlp":
        if args.batch is None:
            args.batch = 1024
        if args.seq_len is None:
            args.seq_len = 1
        if args.time_feat is None:
            args.time_feat = "u"
        if not args.pin_memory:
            args.pin_memory = True

    elif args.mode == "time_lstm":
        if args.batch is None:
            args.batch = 512
        if args.seq_len is None:
            args.seq_len = 180
        if args.time_feat is None:
            args.time_feat = "u_u2"
        if not args.pin_memory:
            args.pin_memory = True

    elif args.mode == "time_phase":
        if args.batch is None:
            args.batch = 512
        if args.seq_len is None:
            args.seq_len = 180
        if args.time_feat is None:
            args.time_feat = "u_u2"
        if not args.use_phase_onehot:
            args.use_phase_onehot = True
        if args.n_phases is None:
            args.n_phases = 7
        args.pin_memory = False

    elif args.mode == "feat_lstm":
        if args.batch is None:
            args.batch = 64
        if args.seq_len is None:
            args.seq_len = 60
        if args.time_feat is None:
            args.time_feat = "u_u2_tnorm"
        if not args.pin_memory:
            args.pin_memory = True

    else:
        raise ValueError(args.mode)


def parse_args():
    """
    Parse CLI args for evaluation.

    Key args
    --------
    --split:
        Which split to evaluate: train/val/test (default: test)
    --ckpt:
        Path to checkpoint. If omitted, uses {out_dir}/{run_name}/best.pt
    --metrics_json:
        If set, writes metrics payload to a JSON file for easy logging.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=MODES)

    ap.add_argument("--label_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--feat_dir", type=str, default="", help="required for feat_lstm")
    ap.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))

    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument(
        "--time_feat",
        type=str,
        default=None,
        choices=(
            "u",
            "u_u2",
            "tnorm",
            "u_tnorm",
            "u_u2_tnorm",
            "u_u2_u3",
            "u_u2_u3_sin_cos",
            "u_u2_u3_sin_cos_tnorm",
            "u_u2_u3_sin_cos_clip",
        ),
    )
    ap.add_argument("--tnorm_div", type=float, default=6000.0)

    ap.add_argument("--use_phase_onehot", action="store_true")
    ap.add_argument("--use_elapsed_phase", action="store_true")
    ap.add_argument("--n_phases", type=int, default=None)

    ap.add_argument("--out_dir", type=str, default="./runs_taskA")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--metrics_json", type=str, default="", help="write metrics to this path")

    return ap.parse_args()


def main():
    """
    Entry point for evaluation.

    Steps:
      1) parse args + fill defaults
      2) build loader for requested split
      3) infer model input dims from first batch
      4) load checkpoint weights
      5) compute metrics and (optionally) write JSON
    """
    args = parse_args()
    apply_mode_defaults(args)

    if args.mode == "feat_lstm" and not args.feat_dir:
        raise ValueError("feat_lstm requires --feat_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    run_name = args.run_name if args.run_name else args.mode
    run_dir = out_dir / run_name

    # If --ckpt is not provided, default to best.pt under run directory
    ckpt_path = Path(args.ckpt) if args.ckpt else (run_dir / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dl_test = build_test_loader(args)

    # Infer dimensions from one batch (keeps evaluation robust to feature packing changes)
    sample_batch = next(iter(dl_test))
    model = build_model(args, sample_batch).to(device)

    # Load checkpoint: support both {"model": state_dict} and raw state_dict
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)

    scores = eval_mae_seconds(model, dl_test, device, args.mode)
    split_tag = args.split.upper()
    print(f"[{split_tag}] rt={scores[0]:.1f}s | bounds={scores[1]:.1f}s | total={scores[2]:.1f}s")

    # Optional: write metrics for logging/reproducibility
    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.mode,
            "split": args.split,
            "ckpt": str(ckpt_path),
            "rt_mae_s": scores[0],
            "bounds_mae_s": scores[1],
            "total_mae_s": scores[2],
        }
        metrics_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
