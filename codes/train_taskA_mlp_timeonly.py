import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from datasets import TaskATimeOnlyDataset
from models import TimeOnlyMLP
from losses import masked_smooth_l1


LABEL_DIR = Path("/myriadfs/home/rmaphyo/Scratch/cholec80_data/labels_1fps")
SPLIT_JSON = LABEL_DIR / "split_default_60_10_10.json"
CKPT_PATH = LABEL_DIR / "taskA_mlp_timeonly_best_rt.pt"


@torch.no_grad()
def eval_mae_seconds(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    rt_abs = []
    bounds_abs = []
    total_abs = []
    total_mask = []
    bounds_mask = []

    for x, y_log, m in loader:
        x = x.to(device)
        y_log = y_log.to(device)
        m = m.to(device)

        pred_log = model(x)
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

    split = json.loads(SPLIT_JSON.read_text())
    train_sets = [TaskATimeOnlyDataset(LABEL_DIR / f"{vid}_labels.npz") for vid in split["train"]]
    val_sets = [TaskATimeOnlyDataset(LABEL_DIR / f"{vid}_labels.npz") for vid in split["val"]]

    ds_train = ConcatDataset(train_sets)
    ds_val = ConcatDataset(val_sets)

    dl_train = DataLoader(ds_train, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

    model = TimeOnlyMLP(in_dim=1, out_dim=15, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(1, 21):
        model.train()
        running = 0.0

        for x, y_log, m in dl_train:
            x = x.to(device)
            y_log = y_log.to(device)
            m = m.to(device)

            pred_log = model(x)
            loss = masked_smooth_l1(pred_log, y_log, m)

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

        if rt_mae < best_val:
            best_val = rt_mae
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_PATH)
            print("  saved:", CKPT_PATH)

    print("done. best val rt MAE:", best_val, "seconds | epoch:", best_epoch)

if __name__ == "__main__":
    main()
