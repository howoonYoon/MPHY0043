#!/usr/bin/env python3
"""
Task A training script (RST regression) with unified training loop across model modes.

This script supports four modes:
  - time_mlp   : MLP on per-frame time features (no sequence)
  - time_lstm  : LSTM on time-feature sequences
  - time_phase : LSTM on time(+phase) feature sequences (phase features are pre-packed by dataset)
  - feat_lstm  : LSTM that consumes visual feature sequences + time-feature sequences

Key design:
  - Different dataset/model signatures are normalized via `forward_adapter`.
  - Targets are assumed to be in log-space (log1p) and predictions are evaluated in seconds via expm1.
  - Best checkpoint is selected by validation total MAE (masked).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import random

from models_taskA import TimeOnlyMLP, TimeOnlyLSTM, FeatLSTMTime
from datasets_taskA import TaskADataset
from losses import masked_smooth_l1, weighted_masked_smooth_l1

# Supported training configurations
MODES = ("time_mlp", "time_lstm", "time_phase", "feat_lstm")


# =========================
# Forward adapter
# =========================
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
        The model for the selected mode.
    batch:
        One minibatch from the DataLoader. Its structure depends on `mode`:
          - time_mlp   : (x, y_log, m)
          - time_lstm  : (x_seq, y_log, m)
          - time_phase : (x_seq, y_log, m)  # x_seq already includes phase-related features if enabled
          - feat_lstm  : (feat_seq, time_seq, y_log, m)
    device:
        Target device for tensors.
    mode:
        One of MODES.

    Returns
    -------
    pred_log : torch.Tensor
        Predicted targets in log-space, shape (B, 15).
    y_log : torch.Tensor
        Ground-truth targets in log-space, shape (B, 15).
    m : torch.Tensor
        Mask indicating valid targets, shape (B, 15).
        Mask is used both for loss and masked MAE computation.
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


# =========================
# Evaluation
# =========================
@torch.no_grad()
def eval_mae_seconds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> Tuple[float, float, float]:
    """
    Evaluate masked MAE in *seconds* (not log-space) on a loader.

    Metrics
    -------
    rt_mae:
        MAE of rt_current (index 0), averaged over all samples (unmasked).
    bounds_mae:
        MAE over boundary-related targets (indices 1:), masked by m[:, 1:].
    total_mae:
        MAE over all 15 targets, masked by m.

    Notes
    -----
    - Predictions and labels are expected in log-space (log1p). We invert with expm1().
    - `m` is used to ignore invalid/ended phases.
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

        # rt_current (index 0) is always present in this setup
        rt_abs.append(abs_err[:, 0].detach().cpu())

        # boundary targets (index 1:)
        b_err = abs_err[:, 1:] * m[:, 1:]
        bounds_abs.append(b_err.detach().cpu())
        bounds_mask.append(m[:, 1:].detach().cpu())

        # total masked error across all 15 dims
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


# =========================
# Training (single epoch)
# =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    opt: torch.optim.Optimizer,
    w_bounds: float,
) -> float:
    """
    Train for one epoch and return mean training loss (in log-space).

    Loss choice
    ----------
    - time_mlp   : masked_smooth_l1 (legacy/simple)
    - others     : weighted_masked_smooth_l1 (optionally upweights boundary terms)

    Gradient handling
    ---------------
    - zero_grad(set_to_none=True) for efficiency
    - clip_grad_norm_ to prevent exploding gradients (esp. LSTMs)
    """
    model.train()
    running = 0.0

    for batch in loader:
        pred_log, y_log, m = forward_adapter(model, batch, device, mode)

        if mode == "time_mlp":
            loss = masked_smooth_l1(pred_log, y_log, m)
        else:
            loss = weighted_masked_smooth_l1(pred_log, y_log, m, w_bounds=w_bounds)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running += float(loss.item())

    return running / max(len(loader), 1)


# =========================
# DataLoader builders
# =========================
def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders for the selected mode.

    The split file is a JSON with at least:
      - split["train"] : list of video ids
      - split["val"]   : list of video ids

    Each video has a corresponding label NPZ:
      {vid}_labels.npz

    Returns
    -------
    dl_train, dl_val:
        DataLoaders over concatenated per-video datasets.
    """
    label_dir = Path(args.label_dir)
    split = json.loads(Path(args.split_json).read_text())

    train_vids = split["train"]
    val_vids = split["val"]

    # Build per-video datasets (keeps logic explicit and debuggable)
    if args.mode == "time_mlp":
        train_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="time_point",
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
            )
            for vid in train_vids
        ]
        val_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="time_point",
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
            )
            for vid in val_vids
        ]

    elif args.mode == "time_lstm":
        train_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="time_seq",
                seq_len=args.seq_len,
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
            )
            for vid in train_vids
        ]
        val_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="time_seq",
                seq_len=args.seq_len,
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
            )
            for vid in val_vids
        ]

    elif args.mode == "time_phase":
        train_sets = [
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
            for vid in train_vids
        ]
        val_sets = [
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
            for vid in val_vids
        ]

    elif args.mode == "feat_lstm":
        feat_dir = Path(args.feat_dir)
        train_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="feat_seq",
                seq_len=args.seq_len,
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
                feat_path=feat_dir / f"{vid}_resnet18.npy",
            )
            for vid in train_vids
        ]
        val_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="feat_seq",
                seq_len=args.seq_len,
                time_feat=args.time_feat,
                tnorm_div=args.tnorm_div,
                feat_path=feat_dir / f"{vid}_resnet18.npy",
            )
            for vid in val_vids
        ]

    else:
        raise ValueError(args.mode)

    # Concatenate per-video datasets so we can shuffle globally in the DataLoader
    ds_train = ConcatDataset(train_sets)
    ds_val = ConcatDataset(val_sets)

    # pin_memory helps GPU training; harmless on CPU but can be disabled per mode
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_val,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return dl_train, dl_val


# =========================
# Model builder
# =========================
def build_model(args, sample_batch: Any) -> nn.Module:
    """
    Instantiate the correct model for the selected mode.

    We infer input dimensions from one sample batch (safe, avoids hardcoding).
    """
    if args.mode == "time_mlp":
        x, _, _ = sample_batch
        in_dim = int(x.shape[-1])
        return TimeOnlyMLP(in_dim=in_dim, out_dim=15, hidden=args.hidden)

    if args.mode == "time_lstm":
        x_seq, _, _ = sample_batch
        in_dim = int(x_seq.shape[-1])
        return TimeOnlyLSTM(
            in_dim=in_dim,
            hidden=args.hidden,
            num_layers=args.layers,
            out_dim=15,
            dropout=args.dropout,
        )

    if args.mode == "time_phase":
        # time_phase uses the same LSTM architecture; dataset packs extra features into x_seq if enabled
        x_seq, _, _ = sample_batch
        in_dim = int(x_seq.shape[-1])
        return TimeOnlyLSTM(
            in_dim=in_dim,
            hidden=args.hidden,
            num_layers=args.layers,
            out_dim=15,
            dropout=args.dropout,
        )

    if args.mode == "feat_lstm":
        feat_seq, time_seq, _, _ = sample_batch
        feat_dim = int(feat_seq.shape[-1])
        time_dim = int(time_seq.shape[-1])
        return FeatLSTMTime(
            feat_dim=feat_dim,
            time_dim=time_dim,
            hidden=args.hidden,
            layers=args.layers,
            out_dim=15,
            dropout=args.dropout,
        )

    raise ValueError(args.mode)


# =========================
# Checkpoint IO
# =========================
def save_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, epoch: int, best_total: float) -> None:
    """
    Save a training checkpoint.

    Stored fields
    -------------
    - model state_dict
    - optimizer state_dict
    - epoch (last completed epoch)
    - best_total (best validation total MAE so far)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "best_total": best_total,
        },
        path,
    )


def load_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, device: torch.device) -> Tuple[int, float]:
    """
    Load checkpoint into model (and optimizer if provided).

    Returns
    -------
    start_epoch:
        Next epoch index to run (checkpoint epoch + 1).
    best_total:
        Best validation total MAE recorded in the checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt and opt is not None:
        opt.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_total = float(ckpt.get("best_total", float("inf")))
    return start_epoch, best_total


# =========================
# Default hyperparameters per mode
# =========================
def apply_mode_defaults(args) -> None:
    """
    Fill in unspecified arguments with mode-specific defaults.

    Convention:
      - CLI defaults are None for tunables, then we set good defaults here.
      - Early stopping is disabled unless --early_stop_patience is provided.
    """
    if args.mode == "time_mlp":
        if args.epochs is None:
            args.epochs = 50
        if args.batch is None:
            args.batch = 512
        if args.batch_val is None:
            args.batch_val = 1024
        if args.lr is None:
            args.lr = 1e-3
        if args.weight_decay is None:
            args.weight_decay = 1e-5
        if args.w_bounds is None:
            args.w_bounds = 0.0
        if args.seq_len is None:
            args.seq_len = 1
        if args.time_feat is None:
            args.time_feat = "u"
        # Enable pin_memory by default for speed on GPU
        if not args.pin_memory:
            args.pin_memory = True

    elif args.mode == "time_lstm":
        if args.epochs is None:
            args.epochs = 50
        if args.batch is None:
            args.batch = 256
        if args.batch_val is None:
            args.batch_val = 512
        if args.lr is None:
            args.lr = 3e-4
        if args.weight_decay is None:
            args.weight_decay = 1e-5
        if args.w_bounds is None:
            args.w_bounds = 0.1
        if args.seq_len is None:
            args.seq_len = 180
        if args.time_feat is None:
            args.time_feat = "u_u2"
        if not args.pin_memory:
            args.pin_memory = True

    elif args.mode == "time_phase":
        if args.epochs is None:
            args.epochs = 50
        if args.batch is None:
            args.batch = 256
        if args.batch_val is None:
            args.batch_val = 512
        if args.lr is None:
            args.lr = 3e-4
        if args.weight_decay is None:
            args.weight_decay = 1e-5
        if args.w_bounds is None:
            args.w_bounds = 0.1
        if args.seq_len is None:
            args.seq_len = 180
        if args.time_feat is None:
            args.time_feat = "u_u2"
        # Prefer phase one-hot in this mode unless user disables it
        if not args.use_phase_onehot:
            args.use_phase_onehot = True
        if args.n_phases is None:
            args.n_phases = 7
        # Often not worth pinning because sequences can be large; keep as explicit choice
        args.pin_memory = False

    elif args.mode == "feat_lstm":
        if args.epochs is None:
            args.epochs = 50
        if args.batch is None:
            args.batch = 64
        if args.batch_val is None:
            args.batch_val = 64
        if args.lr is None:
            args.lr = 3e-4
        if args.weight_decay is None:
            args.weight_decay = 5e-4
        if args.w_bounds is None:
            args.w_bounds = 0.2
        if args.seq_len is None:
            args.seq_len = 60
        if args.time_feat is None:
            args.time_feat = "u_u2_tnorm"
        # Slightly higher dropout often helps when using visual features
        if args.dropout == 0.2:
            args.dropout = 0.4

    else:
        raise ValueError(args.mode)


# =========================
# CLI arguments
# =========================
def parse_args():
    """
    Parse command-line arguments.

    Note:
      - Many hyperparameters default to None and are later filled by apply_mode_defaults().
      - `--pin_memory`, `--use_phase_onehot`, `--use_elapsed_phase`, `--eval_only` are boolean flags.
      - Early stopping is disabled unless --early_stop_patience is explicitly set.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=MODES)

    ap.add_argument("--label_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)

    ap.add_argument("--feat_dir", type=str, default="", help="required for feat_lstm")

    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--batch_val", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--w_bounds", type=float, default=None)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)

    # sequence and time features
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

    # phase features
    ap.add_argument("--use_phase_onehot", action="store_true")
    ap.add_argument("--use_elapsed_phase", action="store_true")
    ap.add_argument("--n_phases", type=int, default=None)

    # checkpoints / running
    ap.add_argument("--out_dir", type=str, default="./runs_taskA")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--eval_only", action="store_true", help="skip training; evaluate val using provided ckpt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="enable early stopping if set (default: disabled)",
    )

    return ap.parse_args()


# =========================
# Reproducibility
# =========================
def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU/GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Main
# =========================
def main():
    """
    Entry point.

    Workflow:
      1) parse args + fill defaults + set seed
      2) build dataloaders
      3) infer input dims from one batch -> build model
      4) train loop with validation, checkpointing (best/last)
      5) optional early stopping (only when --early_stop_patience is set)
    """
    args = parse_args()
    apply_mode_defaults(args)
    set_seed(args.seed)

    # Sanity check for modes requiring extra inputs
    if args.mode == "feat_lstm" and not args.feat_dir:
        raise ValueError("feat_lstm requires --feat_dir")

    print("Note: test evaluation is in test_taskA.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    run_name = args.run_name if args.run_name else args.mode
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    # Build train/val loaders
    dl_train, dl_val = build_dataloaders(args)

    # Use one batch to infer input dimensions robustly
    sample_batch = next(iter(dl_train))
    model = build_model(args, sample_batch).to(device)

    # Optimizer (+ scheduler for feat_lstm only)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = None
    if args.mode == "feat_lstm":
        # Reduce LR when validation stops improving (helps convergence on feature-based model)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

    start_epoch = 1
    best_total = float("inf")

    # Resume training from a checkpoint if requested
    if args.resume:
        start_epoch, best_total = load_ckpt(Path(args.resume), model, opt, device)
        print(f"=> resumed from {args.resume} start_epoch={start_epoch} best_total={best_total:.3f}")

    # Eval-only: load checkpoint and report validation MAE
    if args.eval_only:
        if not args.resume and not best_ckpt.exists():
            raise RuntimeError("eval_only requires --resume or an existing best.pt in run dir")
        if not args.resume:
            _, best_total = load_ckpt(best_ckpt, model, opt=None, device=device)  # type: ignore
        scores = eval_mae_seconds(model, dl_val, device, args.mode)
        print(f"[VAL] rt={scores[0]:.1f}s | bounds={scores[1]:.1f}s | total={scores[2]:.1f}s")
        return

    # Train loop
    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, dl_train, device, args.mode, opt, w_bounds=args.w_bounds)
        rt_mae, bounds_mae, total_mae = eval_mae_seconds(model, dl_val, device, args.mode)

        # Update scheduler (if used) based on validation total MAE
        if sched is not None:
            sched.step(total_mae)

        dt = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d} | train_loss(log)={train_loss:.4f} | "
            f"val_rt={rt_mae:.1f}s ({rt_mae/60:.2f}m) | "
            f"val_bounds={bounds_mae:.1f}s | val_total={total_mae:.1f}s | "
            f"lr={lr:.2e} | {dt/60:.1f} min"
        )

        # Always save "last" so runs are recoverable even if interrupted
        save_ckpt(last_ckpt, model, opt, epoch, best_total)

        # Save "best" according to validation total MAE
        improved = total_mae < best_total
        if improved:
            best_total = total_mae
            save_ckpt(best_ckpt, model, opt, epoch, best_total)
            print("  saved best (by total MAE):", best_ckpt)

        # Optional early stopping (disabled by default; enabled only if patience is set)
        if args.early_stop_patience is not None:
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.early_stop_patience:
                    print(f"Early stop: no val improvement for {args.early_stop_patience} epochs")
                    break


if __name__ == "__main__":
    main()
