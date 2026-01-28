import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from models import FeatLSTMTime
from datasets import TaskAFeatSeqDataset
from losses import weighted_masked_smooth_l1


LABEL_DIR = Path("/myriadfs/home/rmaphyo/Scratch/cholec80_data/labels_1fps")
SPLIT_JSON = LABEL_DIR / "split_default_60_10_10.json"

FEAT_DIR = Path("/myriadfs/home/rmaphyo/Scratch/cholec80_data/features_1fps")
CKPT_PATH = LABEL_DIR / "taskA_feat_lstm_best_rt.pt"



@torch.no_grad()
def eval_mae_seconds(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    rt_abs, bounds_abs, bounds_mask = [], [], []
    total_abs, total_mask = [], []

    for feat_seq, time_seq, y_log, m in loader:
        feat_seq = feat_seq.to(device)
        time_seq = time_seq.to(device)
        y_log = y_log.to(device)
        m = m.to(device)

        pred_log = model(feat_seq, time_seq)
        pred = torch.expm1(pred_log)
        y = torch.expm1(y_log)

        abs_err = torch.abs(pred - y)
        rt_abs.append(abs_err[:, 0].cpu())

        b_err = abs_err[:, 1:] * m[:, 1:]
        bounds_abs.append(b_err.cpu())
        bounds_mask.append(m[:, 1:].cpu())

        total_abs.append((abs_err * m).cpu())
        total_mask.append(m.cpu())

    rt_mae = torch.cat(rt_abs).mean().item()
    b_err_all = torch.cat(bounds_abs, dim=0)
    b_m_all = torch.cat(bounds_mask, dim=0)
    bounds_mae = (b_err_all.sum() / b_m_all.sum().clamp_min(1.0)).item()

    t_err_all = torch.cat(total_abs, dim=0)
    t_m_all = torch.cat(total_mask, dim=0)
    total_mae = (t_err_all.sum() / t_m_all.sum().clamp_min(1.0)).item()

    return rt_mae, bounds_mae, total_mae


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    SEQ_LEN = 120
    TIME_DIM = 1        # [u, t_norm]
    EPOCHS = 50
    BATCH = 64

    split = json.loads(SPLIT_JSON.read_text())

    train_sets = []
    for vid in split["train"]:
        npz = LABEL_DIR / f"{vid}_labels.npz"
        feat = FEAT_DIR / f"{vid}_resnet18.npy"
        train_sets.append(TaskAFeatSeqDataset(npz, feat, seq_len=SEQ_LEN, time_feat_dim=TIME_DIM))

    val_sets = []
    for vid in split["val"]:
        npz = LABEL_DIR / f"{vid}_labels.npz"
        feat = FEAT_DIR / f"{vid}_resnet18.npy"
        val_sets.append(TaskAFeatSeqDataset(npz, feat, seq_len=SEQ_LEN, time_feat_dim=TIME_DIM))

    ds_train = ConcatDataset(train_sets)
    ds_val = ConcatDataset(val_sets)

    # after creating a dataset, e.g. first train video
    ds0 = train_sets[0]
    print("T npz:", ds0.T, "F len:", ds0.F.shape[0], "X_time:", ds0.X_time.shape[0])

    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)

    # infer feat dim
    f0, t0, _, _ = ds_train[0]
    feat_dim = int(f0.shape[-1])
    time_dim = int(t0.shape[-1])
    print("feat_dim:", feat_dim, "time_dim:", time_dim)

    model = FeatLSTMTime(feat_dim=feat_dim, time_dim=time_dim, hidden=128, layers=1, out_dim=15, dropout=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=4
    )

    best_rt = float("inf")
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for feat_seq, time_seq, y_log, m in dl_train:
            feat_seq = feat_seq.to(device)
            time_seq = time_seq.to(device)
            y_log = y_log.to(device)
            m = m.to(device)

            pred_log = model(feat_seq, time_seq)
            loss = weighted_masked_smooth_l1(pred_log, y_log, m, w_bounds=0.2)
            #loss = weighted_masked_smooth_l1(pred_log[:, :1], y_log[:, :1], m[:, :1], w_bounds=0.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()

        train_loss = running / max(len(dl_train), 1)
        rt_mae, bounds_mae, total_mae = eval_mae_seconds(model, dl_val, device)

        print(
            f"epoch {epoch:02d} | train_loss(log)={train_loss:.4f} "
            f"| val_rt_MAE={rt_mae:.1f}s ({rt_mae/60:.2f}m) "
            f"| val_bounds_MAE={bounds_mae:.1f}s ({bounds_mae/60:.2f}m) "
            f"| val_total_MAE={total_mae:.1f}s ({total_mae/60:.2f}m)"
        )
        print(f"  lr={opt.param_groups[0]['lr']:.2e}")

        # ✅ 여기!
        sched.step(rt_mae)
        
        if rt_mae < best_rt:
            best_rt = rt_mae
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_PATH)
            print("  saved:", CKPT_PATH)

    print("done. best val rt MAE:", best_rt, "seconds | epoch:", best_epoch)


if __name__ == "__main__":
    main()
