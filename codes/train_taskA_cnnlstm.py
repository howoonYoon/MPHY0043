import json
from pathlib import Path
from typing import Tuple
import time
import argparse


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from datasets_img import TaskAImageSeqDataset
from models_img import ResNetLSTMTime
from losses import weighted_masked_smooth_l1


LABEL_DIR = Path("/myriadfs/home/rmaphyo/Scratch/cholec80_data/labels_1fps")
SPLIT_JSON = LABEL_DIR / "split_default_60_10_10.json"

CHOLEC80_DIR = Path("/myriadfs/home/rmaphyo/Scratch/cholec80_data/cholec80")
FRAMES_ROOT = CHOLEC80_DIR / "frames"

OUT_DIR = LABEL_DIR / "taskA_resnetlstm_layer4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = OUT_DIR / "best.pt"
LAST_CKPT = OUT_DIR / "last.pt"


@torch.no_grad()
def eval_mae_seconds(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    rt_abs, bounds_abs, bounds_mask = [], [], []
    total_abs, total_mask = [], []

    for j, (img_seq, time_seq, y_log, m) in enumerate(loader, 1):
        img_seq = img_seq.to(device, non_blocking=True)
        time_seq = time_seq.to(device, non_blocking=True)
        y_log = y_log.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        pred_log = model(img_seq, time_seq)
        pred = torch.expm1(pred_log)
        y = torch.expm1(y_log)

        abs_err = torch.abs(pred - y)

        rt_abs.append(abs_err[:, 0].cpu())

        b_err = abs_err[:, 1:] * m[:, 1:]
        bounds_abs.append(b_err.cpu())
        bounds_mask.append(m[:, 1:].cpu())

        total_abs.append((abs_err * m).cpu())
        total_mask.append(m.cpu())
        if j % 50 == 0:
            print(f"  eval iter {j}/{len(loader)}", flush=True)

    rt_mae = torch.cat(rt_abs).mean().item()
    b_err_all = torch.cat(bounds_abs, dim=0)
    b_m_all = torch.cat(bounds_mask, dim=0)
    bounds_mae = (b_err_all.sum() / b_m_all.sum().clamp_min(1.0)).item()
    t_err_all = torch.cat(total_abs, dim=0)
    t_m_all = torch.cat(total_mask, dim=0)
    total_mae = (t_err_all.sum() / t_m_all.sum().clamp_min(1.0)).item()

    return rt_mae, bounds_mae, total_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--resume", type=str, default=None, help="path to last/best checkpoint to resume")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # hyperparams (start small)
    SEQ_LEN = 12          # 이미지 포함은 먼저 짧게 (30초 컨텍스트)
    IMG_SIZE = 160
    TIME_DIM = 2          # [u, t_norm]
    EPOCHS = args.epochs
    BATCH = 4             # 이미지라 batch 작게 시작

    split = json.loads(SPLIT_JSON.read_text())
    train_sets = [
        TaskAImageSeqDataset(
            LABEL_DIR / f"{vid}_labels.npz",
            FRAMES_ROOT,
            seq_len=SEQ_LEN,
            img_size=IMG_SIZE,
            time_feat_dim=TIME_DIM,
            stride=5,
        )
        for vid in split["train"]
    ]
    val_sets = [
        TaskAImageSeqDataset(
            LABEL_DIR / f"{vid}_labels.npz",
            FRAMES_ROOT,
            seq_len=SEQ_LEN,
            img_size=IMG_SIZE,
            time_feat_dim=TIME_DIM,
            stride=1,   # ✅ 평가는 1fps
        )
        for vid in split["val"]
    ]



    ds_train = ConcatDataset(train_sets)
    ds_val = ConcatDataset(val_sets)
    

    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    print("loading first batch...", flush=True)
    _ = next(iter(dl_train))
    print("first batch ok", flush=True)
    print("len train ds:", len(ds_train), "len train loader:", len(dl_train))

    model = ResNetLSTMTime(
        time_dim=TIME_DIM,
        lstm_hidden=256,
        lstm_layers=1,
        out_dim=15,
        freeze_cnn=True,
        unfreeze_layer4=True,
        dropout=0.2,
    ).to(device)

    # ✅ 파라미터 그룹 분리
    cnn_params = [p for p in model.cnn.parameters() if p.requires_grad]   # layer4만 들어올 것
    other_params = [p for n, p in model.named_parameters() if (p.requires_grad and not n.startswith("cnn."))]

    opt = torch.optim.Adam(
        [
            {"params": other_params, "lr": 3e-4, "weight_decay": 1e-5},
            {"params": cnn_params,   "lr": 1e-5, "weight_decay": 1e-5},  # ✅ layer4는 작게
        ]
    )
    best_rt = float("inf")
    best_epoch = -1
    start_epoch = 1
    scaler = torch.cuda.amp.GradScaler()
    print("lr other:", opt.param_groups[0]["lr"], "lr cnn:", opt.param_groups[1]["lr"])

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_rt = ckpt.get("best_rt", best_rt)
            best_epoch = ckpt.get("best_epoch", best_epoch)
        else:
            model.load_state_dict(ckpt)

        print(f"=> Resumed from {args.resume}")
        print(f"=> start_epoch={start_epoch}, best_rt={best_rt}, best_epoch={best_epoch}")

    print("trainable cnn params:", sum(p.numel() for p in model.cnn.parameters() if p.requires_grad))

    t_data0 = time.time()
    # epoch loop 안쪽
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running = 0.0

        t0 = time.time()
        t_data0 = time.time()   # ✅ epoch 시작에서 초기화

        for it, (img_seq, time_seq, y_log, m) in enumerate(dl_train, 1):
            # ✅ data loading 시간 (이 줄 실행되기 전까지가 dataloader 대기시간)
            t_data = time.time() - t_data0

            # ✅ H2D copy
            img_seq  = img_seq.to(device, non_blocking=True)
            time_seq = time_seq.to(device, non_blocking=True)
            y_log    = y_log.to(device, non_blocking=True)
            m        = m.to(device, non_blocking=True)

            t_gpu0 = time.time()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred_log = model(img_seq, time_seq)
                loss = weighted_masked_smooth_l1(pred_log, y_log, m, w_bounds=0.05)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running += loss.item()

            t_gpu = time.time() - t_gpu0

            if it % 50 == 0:
                dt = time.time() - t0
                print(f"iter {it}/{len(dl_train)} avg {dt/it:.3f}s/iter | data={t_data:.3f}s gpu={t_gpu:.3f}s", flush=True)

            t_data0 = time.time()   # ✅ 다음 iter의 dataloader 시간 측정을 위해 업데이트

                
        train_loss = running / max(len(dl_train), 1)
        rt_mae, bounds_mae, total_mae = eval_mae_seconds(model, dl_val, device)

        print(
            f"epoch {epoch:02d} | train_loss(log)={train_loss:.4f} "
            f"| val_rt_MAE={rt_mae:.1f}s ({rt_mae/60:.2f}m) "
            f"| val_bounds_MAE={bounds_mae:.1f}s ({bounds_mae/60:.2f}m) "
            f"| val_total_MAE={total_mae:.1f}s ({total_mae/60:.2f}m)"
        )
        dt = time.time() - t0
        print(f"epoch {epoch:02d} time: {dt/60:.1f} min", flush=True)

        # best 업데이트 먼저
        if rt_mae < best_rt:
            best_rt = rt_mae
            best_epoch = epoch
            print("  new best")


        # 항상 last 저장
        ckpt = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_rt": best_rt,
            "best_epoch": best_epoch,
            "cfg": {
                "SEQ_LEN": SEQ_LEN, "IMG_SIZE": IMG_SIZE, "TIME_DIM": TIME_DIM,
                "BATCH": BATCH, "stride_train": 5
            },
        }
        torch.save(ckpt, LAST_CKPT)
        print("  saved last:", LAST_CKPT)

        # best 저장
        if epoch == best_epoch:
            ckpt["best_rt"] = best_rt
            ckpt["best_epoch"] = best_epoch
            torch.save(ckpt, BEST_CKPT)
            print("  saved best:", BEST_CKPT)


    print("done. best val rt MAE:", best_rt, "seconds | epoch:", best_epoch)


if __name__ == "__main__":
    main()
