#!/usr/bin/env python3
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

# Example (defaults match original scripts):
#   python train_taskA.py --mode time_mlp --label_dir /path/labels --split_json /path/split.json
#   python train_taskA.py --mode time_lstm --label_dir /path/labels --split_json /path/split.json
#   python train_taskA.py --mode time_phase --label_dir /path/labels --split_json /path/split.json --use_phase_onehot
#   python train_taskA.py --mode feat_lstm --label_dir /path/labels --split_json /path/split.json --feat_dir /path/feats


MODES = ("time_mlp", "time_lstm", "time_phase", "feat_lstm")


# =========================
# Forward adapter (only place where batch/model signatures differ)
# =========================
def forward_adapter(model: nn.Module, batch: Any, device: torch.device, mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      pred_log: (B,15)
      y_log:    (B,15)
      m:        (B,15)
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
# Eval (val/test share this)
# =========================
@torch.no_grad()
def eval_mae_seconds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> Tuple[float, float, float]:
    model.eval()
    rt_abs, bounds_abs, bounds_mask = [], [], []
    total_abs, total_mask = [], []

    for batch in loader:
        pred_log, y_log, m = forward_adapter(model, batch, device, mode)

        pred = torch.expm1(pred_log)
        y = torch.expm1(y_log)
        abs_err = torch.abs(pred - y)

        rt_abs.append(abs_err[:, 0].detach().cpu())

        b_err = abs_err[:, 1:] * m[:, 1:]
        bounds_abs.append(b_err.detach().cpu())
        bounds_mask.append(m[:, 1:].detach().cpu())

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
# Train
# =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    opt: torch.optim.Optimizer,
    w_bounds: float,
) -> float:
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
# Builders
# =========================
def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """
    Return (dl_train, dl_val) for the given mode.
    """
    label_dir = Path(args.label_dir)
    split = json.loads(Path(args.split_json).read_text())

    train_vids = split["train"]
    val_vids = split["val"]

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

    ds_train = ConcatDataset(train_sets)
    ds_val = ConcatDataset(val_sets)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    dl_val = DataLoader(ds_val, batch_size=args.batch_val, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    return dl_train, dl_val


def build_model(args, sample_batch: Any) -> nn.Module:
    """
    Build the correct model for the mode.
    Use sample_batch to infer input dims safely.
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
        x_seq, _, _ = sample_batch
        in_dim = int(x_seq.shape[-1])
        return TimeOnlyLSTM(in_dim=in_dim, hidden=args.hidden, num_layers=args.layers, out_dim=15, dropout=args.dropout)

    if args.mode == "feat_lstm":
        feat_seq, time_seq, _, _ = sample_batch
        feat_dim = int(feat_seq.shape[-1])
        time_dim = int(time_seq.shape[-1])
        return FeatLSTMTime(feat_dim=feat_dim, time_dim=time_dim, hidden=args.hidden, layers=args.layers, out_dim=15, dropout=args.dropout)

    raise ValueError(args.mode)


def save_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, epoch: int, best_total: float):

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "best_total" : best_total
        },
        path,
    )


def load_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, device: torch.device) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt and opt is not None:
        opt.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_total = float(ckpt.get("best_total", float("inf")))

    return start_epoch, best_total


def apply_mode_defaults(args) -> None:
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
        if not args.use_phase_onehot:
            args.use_phase_onehot = True
        if args.n_phases is None:
            args.n_phases = 7
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
        if args.dropout == 0.2:
            args.dropout = 0.4

    else:
        raise ValueError(args.mode)


def parse_args():
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

    # ckpt
    ap.add_argument("--out_dir", type=str, default="./runs_taskA")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--eval_only", action="store_true", help="skip training, just evaluate val using provided ckpt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early_stop_patience", type=int, default=None)

    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    apply_mode_defaults(args)
    set_seed(args.seed)

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

    # loaders
    dl_train, dl_val = build_dataloaders(args)

    # sample batch to infer dims
    sample_batch = next(iter(dl_train))
    model = build_model(args, sample_batch).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = None
    if args.mode == "feat_lstm":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

    start_epoch = 1
    #best_rt = float("inf")
    best_total = float("inf")

    # resume
    if args.resume:
        start_epoch, best_total = load_ckpt(Path(args.resume), model, opt, device)
        print(f"=> resumed from {args.resume} start_epoch={start_epoch} best_total={best_total:.3f}")

    # eval-only
    if args.eval_only:
        if not args.resume and not best_ckpt.exists():
            raise RuntimeError("eval_only requires --resume or an existing best.pt in run dir")
        if not args.resume:
            start_epoch, best_total = load_ckpt(best_ckpt, model, opt=None, device=device)  # type: ignore
        scores = eval_mae_seconds(model, dl_val, device, args.mode)
        print(f"[VAL] rt={scores[0]:.1f}s | bounds={scores[1]:.1f}s | total={scores[2]:.1f}s")

        return

    # train loop
    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, dl_train, device, args.mode, opt, w_bounds=args.w_bounds)
        rt_mae, bounds_mae, total_mae = eval_mae_seconds(model, dl_val, device, args.mode)

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

        # save last
        save_ckpt(last_ckpt, model, opt, epoch, best_total)

        improved = total_mae < best_total
        if improved:
            best_total = total_mae
            save_ckpt(best_ckpt, model, opt, epoch, best_total)
            print("  saved best (by total MAE):", best_ckpt)
        # early stop
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
