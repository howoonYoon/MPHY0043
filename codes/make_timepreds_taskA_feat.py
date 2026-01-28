#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets_taskA import TaskADataset
from models_taskA import FeatLSTMTime
from typing import Tuple


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
        pred_log = model(feat_seq, time_seq)  # (B,15)
        pred = torch.expm1(pred_log) if pred_space == "seconds" else pred_log
        preds.append(pred.detach().cpu())

    if len(preds) == 0:
        raise RuntimeError(f"No predictions for {npz_path.name}: seq_len={seq_len}, T={T}")
    pred_tail = torch.cat(preds, dim=0)  # (T-(L-1), 15)

    # --- 수정 ---
    start_t = seq_len - 1
    pred_full = torch.zeros((T, pred_tail.shape[1]), dtype=torch.float32)  # 초반 0
    valid = torch.zeros((T,), dtype=torch.bool)                            # 초반 invalid

    pred_full[start_t:, :] = pred_tail
    valid[start_t:] = True

    return pred_full, valid


def infer_time_dim(sample_npz: Path, sample_feat: Path, seq_len: int, time_feat: str, tnorm_div: float) -> int:
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
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # defaults (맞으면 그대로 사용)
    hidden, layers, dropout = 256, 1, 0.2
    if isinstance(ckpt, dict):
        a = ckpt.get("args", {})
        hidden = int(a.get("hidden", a.get("lstm_hidden", hidden)))
        layers = int(a.get("layers", a.get("num_layers", layers)))
        dropout = float(a.get("dropout", dropout))

    if "lstm.weight_ih_l0" in state:
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
    Your naming: video01_resnet18.npy  -> f"{vid}_{suffix}.npy"
    """
    return feats_dir / f"{vid}_{suffix}.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", type=str, required=True, help="dir with *_labels.npz")
    ap.add_argument("--feats_dir", type=str, required=True, help="dir with {vid}_resnet18.npy")
    ap.add_argument("--feat_suffix", type=str, default="resnet18", help="e.g. resnet18 (file is {vid}_{suffix}.npy)")
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=180)
    ap.add_argument("--time_feat", type=str, default="u_u2_tnorm",
                    choices=[
                        "u", "u_u2", "tnorm", "u_tnorm", "u_u2_tnorm",
                        "u_u2_u3", "u_u2_u3_sin_cos", "u_u2_u3_sin_cos_tnorm", "u_u2_u3_sin_cos_clip"
                    ])
    ap.add_argument("--tnorm_div", type=float, default=6000.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--pred_space", type=str, default="seconds", choices=["log", "seconds"])

    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    feats_dir = Path(args.feats_dir)
    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # infer time_dim
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
        torch.save(
            {"pred": pred_full.cpu(), "valid": valid.cpu()},
            out_path
        )
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
