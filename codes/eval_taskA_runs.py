#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from datasets_taskA import TaskADataset
from models_taskA import TimeOnlyMLP, TimeOnlyLSTM, FeatLSTMTime


def forward_adapter(model: nn.Module, batch: Any, device: torch.device, mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def build_val_loader(args, mode: str) -> DataLoader:
    label_dir = Path(args.label_dir)
    split = json.loads(Path(args.split_json).read_text())
    val_vids = split["val"]

    if mode == "time_mlp":
        val_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="time_point",
                time_feat=args.time_feat_mlp,
                tnorm_div=args.tnorm_div,
            )
            for vid in val_vids
        ]
    elif mode == "time_lstm":
        val_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="time_seq",
                seq_len=args.seq_len_lstm,
                time_feat=args.time_feat_lstm,
                tnorm_div=args.tnorm_div,
            )
            for vid in val_vids
        ]
    elif mode == "time_phase":
        val_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="timephase_seq",
                seq_len=args.seq_len_phase,
                time_feat=args.time_feat_phase,
                tnorm_div=args.tnorm_div,
                use_phase_onehot=True,
                use_elapsed_phase=args.use_elapsed_phase,
                n_phases=args.n_phases,
            )
            for vid in val_vids
        ]
    elif mode == "feat_lstm":
        feat_dir = Path(args.feat_dir)
        val_sets = [
            TaskADataset(
                label_dir / f"{vid}_labels.npz",
                mode="feat_seq",
                seq_len=args.seq_len_feat,
                time_feat=args.time_feat_feat,
                tnorm_div=args.tnorm_div,
                feat_path=feat_dir / f"{vid}_resnet18.npy",
            )
            for vid in val_vids
        ]
    else:
        raise ValueError(mode)

    ds_val = ConcatDataset(val_sets)
    return DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)


def build_model(mode: str, sample_batch: Any, args) -> nn.Module:
    if mode == "time_mlp":
        x, _, _ = sample_batch
        in_dim = int(x.shape[-1])
        return TimeOnlyMLP(in_dim=in_dim, out_dim=15, hidden=args.hidden)
    if mode == "time_lstm":
        x_seq, _, _ = sample_batch
        in_dim = int(x_seq.shape[-1])
        return TimeOnlyLSTM(in_dim=in_dim, hidden=args.hidden, num_layers=args.layers, out_dim=15, dropout=args.dropout)
    if mode == "time_phase":
        x_seq, _, _ = sample_batch
        in_dim = int(x_seq.shape[-1])
        return TimeOnlyLSTM(in_dim=in_dim, hidden=args.hidden, num_layers=args.layers, out_dim=15, dropout=args.dropout)
    if mode == "feat_lstm":
        feat_seq, time_seq, _, _ = sample_batch
        feat_dim = int(feat_seq.shape[-1])
        time_dim = int(time_seq.shape[-1])
        return FeatLSTMTime(feat_dim=feat_dim, time_dim=time_dim, hidden=args.hidden, layers=args.layers, out_dim=15, dropout=args.dropout)
    raise ValueError(mode)


def eval_one_run(run_dir: Path, mode: str, args, device: torch.device) -> Dict[str, float]:
    ckpt_path = run_dir / args.ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    dl_val = build_val_loader(args, mode)
    sample_batch = next(iter(dl_val))
    model = build_model(mode, sample_batch, args).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    rt_mae, bounds_mae, total_mae = eval_mae_seconds(model, dl_val, device, mode)
    return {"rt_mae": rt_mae, "bounds_mae": bounds_mae, "total_mae": total_mae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--feat_dir", type=str, default="")
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--ckpt_name", type=str, default="best.pt")

    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--tnorm_div", type=float, default=6000.0)
    ap.add_argument("--n_phases", type=int, default=7)
    ap.add_argument("--use_elapsed_phase", action="store_true")

    ap.add_argument("--time_feat_mlp", type=str, default="u")
    ap.add_argument("--time_feat_lstm", type=str, default="u_u2")
    ap.add_argument("--time_feat_phase", type=str, default="u_u2")
    ap.add_argument("--time_feat_feat", type=str, default="u_u2_tnorm")

    ap.add_argument("--seq_len_lstm", type=int, default=180)
    ap.add_argument("--seq_len_phase", type=int, default=180)
    ap.add_argument("--seq_len_feat", type=int, default=120)
    args = ap.parse_args()

    runs = {
        "time_mlp": Path(args.runs_root) / "time_mlp",
        "time_lstm": Path(args.runs_root) / "time_lstm",
        "time_phase": Path(args.runs_root) / "time_phase",
        "feat_lstm": Path(args.runs_root) / "feat_u_u2_tnorm",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    for mode, run_dir in runs.items():
        if mode == "feat_lstm" and not args.feat_dir:
            print(f"[{mode}] skipped (need --feat_dir)")
            continue
        if not run_dir.exists():
            print(f"[{mode}] skipped (missing dir: {run_dir})")
            continue

        scores = eval_one_run(run_dir, mode, args, device)
        print(
            f"[{mode}] rt={scores['rt_mae']:.1f}s | "
            f"bounds={scores['bounds_mae']:.1f}s | total={scores['total_mae']:.1f}s"
        )


if __name__ == "__main__":
    main()
