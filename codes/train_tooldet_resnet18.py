#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets_taskB import Cholec80ToolDataset, TOOL_COLS
from models_taskB import ResNet18ToolDet
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_pos_weight(
    labels_dir: Path,
    split_json: Path,
    split: str = "train",
    stride: int = 1,
) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss using the same temporal sampling (stride)
    as the dataset used for training/evaluation.

    pos_weight[c] = (#neg[c]) / (#pos[c])
    """
    split_map = json.loads(split_json.read_text())
    vids = split_map[split]

    ys = []
    for vid in vids:
        npz = np.load(labels_dir / f"{vid}_labels.npz", allow_pickle=False)
        y = npz["tools_1fps"].astype(np.float32)  # (T,7)
        y = y[::stride]  # <-- stride 반영
        ys.append(y)

    Y = np.concatenate(ys, axis=0)  # (N,7)

    pos = Y.sum(axis=0)                 # (7,)
    neg = Y.shape[0] - pos              # (7,)
    pos = np.maximum(pos, 1.0)          # avoid div0
    w = neg / pos
    return torch.tensor(w, dtype=torch.float32)



@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_logits = []
    all_y = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()   # (N,7)
    y = torch.cat(all_y, dim=0).numpy()             # (N,7)
    prob = 1 / (1 + np.exp(-logits))                # sigmoid

    # ----- Threshold metrics (optional but handy)
    pred = (prob >= 0.5).astype(np.float32)
    eps = 1e-8
    tp = (pred * y).sum(axis=0)
    fp = (pred * (1 - y)).sum(axis=0)
    fn = ((1 - pred) * y).sum(axis=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # ----- AUPRC / AP
    # per-class AP (Average Precision = area under PR curve)
    ap_per_tool = {}
    ap_vals = []
    for i, name in enumerate(TOOL_COLS):
        # if a class has no positives in this eval set, average_precision_score is ill-defined
        if y[:, i].sum() == 0:
            ap = float("nan")
        else:
            ap = float(average_precision_score(y[:, i], prob[:, i]))
            ap_vals.append(ap)
        ap_per_tool[name] = ap

    mAP = float(np.nanmean(list(ap_per_tool.values())))  # mean across tools (ignoring NaN)
    micro_auprc = float(average_precision_score(y.ravel(), prob.ravel()))
    macro_auprc = mAP  # same as mean of per-class AP

    # ----- (Optional) AUROC too (often reported, but PR is more relevant for imbalance)
    auroc_per_tool = {}
    auroc_vals = []
    for i, name in enumerate(TOOL_COLS):
        if len(np.unique(y[:, i])) < 2:
            au = float("nan")
        else:
            au = float(roc_auc_score(y[:, i], prob[:, i]))
            auroc_vals.append(au)
        auroc_per_tool[name] = au
    macro_auroc = float(np.nanmean(list(auroc_per_tool.values())))

    return {
        "macro_f1@0.5": float(np.mean(f1)),
        "per_tool_f1@0.5": {TOOL_COLS[i]: float(f1[i]) for i in range(len(TOOL_COLS))},
        "mAP(AUPRC_macro)": mAP,
        "AUPRC_micro": micro_auprc,
        "AP_per_tool": ap_per_tool,
        "AUROC_macro": macro_auroc,
        "AUROC_per_tool": auroc_per_tool,
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cholec80_dir", type=str, required=True)
    ap.add_argument("--labels_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--stride_train", type=int, default=2)  # 1fps는 너무 중복이라 2~5 추천
    ap.add_argument("--stride_val", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None,
                help="path to checkpoint (.pt) to resume from")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cholec80_dir = Path(args.cholec80_dir)
    labels_dir = Path(args.labels_dir)
    split_json = Path(args.split_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    ds_tr = Cholec80ToolDataset(cholec80_dir, labels_dir, split_json, "train", stride=args.stride_train, transform=tfm_train)
    ds_va = Cholec80ToolDataset(cholec80_dir, labels_dir, split_json, "val", stride=args.stride_val, transform=tfm_eval)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # 지금 다 freeze 하고 fc만 학습
    model = ResNet18ToolDet(num_tools=7, pretrained=True).to(device)

    # ---- freeze backbone: only train fc ----
    for p in model.net.parameters():
        p.requires_grad = False
    for p in model.net.fc.parameters():
        p.requires_grad = True
    # ----------------------------------------
    # freeze all
    # for p in model.net.parameters():
    #     p.requires_grad = False
    # # unfreeze layer4 + fc
    # for name, p in model.net.named_parameters():
    #     if name.startswith("layer4") or name.startswith("fc"):
    #         p.requires_grad = True

    pos_weight = compute_pos_weight(labels_dir, split_json, "train", stride=args.stride_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best = -1.0
    start_epoch = 1
    history = []
    
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt.get("epoch", 0) + 1
        best = ckpt.get("best_map", best)

        print(f"=> Resumed from {args.resume}")
        print(f"=> start_epoch={start_epoch}, best_map={best:.4f}")


    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for x, y,_ in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item()) * x.size(0)
            n += x.size(0)

        tr_loss = running / max(n, 1)

        metrics = evaluate(model, dl_va, device)
        score = metrics["mAP(AUPRC_macro)"]

        row = {"epoch": ep, "train_loss": tr_loss, **metrics}
        history.append(row)
        print(row)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": ep,
            "best_map": best,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")

        if score > best:
            best = score
            ckpt["best_map"] = best
            torch.save(ckpt, out_dir / "best.pt")
            (out_dir / "best_metrics.json").write_text(json.dumps(row, indent=2))



    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val mAP(AUPRC_macro)={best:.4f}. Saved to {out_dir/'best.pt'}")



if __name__ == "__main__":
    main()
