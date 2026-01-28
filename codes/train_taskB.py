#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from datasets_taskB import TOOL_COLS, Cholec80ToolDataset
from models_taskB import ResNet18ToolDet
from sklearn.metrics import average_precision_score, roc_auc_score


# -------------------------
# Utils
# -------------------------
def compute_pos_weight(labels_dir: Path, split_json: Path, split: str = "train", stride: int = 1) -> torch.Tensor:
    split_map = json.loads(split_json.read_text())
    vids = split_map[split]

    ys = []
    for vid in vids:
        npz = np.load(labels_dir / f"{vid}_labels.npz", allow_pickle=False)
        y = npz["tools_1fps"].astype(np.float32)[::stride]
        ys.append(y)

    Y = np.concatenate(ys, axis=0)
    pos = Y.sum(axis=0)
    neg = Y.shape[0] - pos
    pos = np.maximum(pos, 1.0)
    w = neg / pos
    return torch.tensor(w, dtype=torch.float32)

def build_oracle_timefeat_from_npz(npz: np.lib.npyio.NpzFile, t: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      tfeat: (15,) float32 = [rt_current, bounds_rem(7x2).flatten()]
      tvalid: scalar bool (always True for oracle; bounds parts are masked using mask_bounds if present)
    """
    # rt_current: (T,1) or (T,)
    rt = npz["rt_current"].astype(np.float32)
    rt_t = float(rt[t] if rt.ndim == 1 else rt[t, 0])

    bounds = npz["bounds_rem"].astype(np.float32)          # (T,7,2) or (T,14)
    if bounds.ndim == 3:
        b = bounds[t].reshape(-1)                          # (14,)
    else:
        b = bounds[t].reshape(-1)

    # optional mask for upcoming phases
    if "mask_bounds" in npz.files:
        mb = npz["mask_bounds"].astype(np.float32)
        mb_t = mb[t].reshape(-1)                           # (14,)
        b = b * mb_t                                       # mask out non-existing phases
    # else: keep as is

    feat = np.concatenate([[rt_t], b], axis=0).astype(np.float32)   # (15,)
    tfeat = torch.from_numpy(feat)
    tvalid = torch.tensor(True)
    return tfeat, tvalid



@torch.no_grad()
def evaluate(model, loader, device, use_time: bool) -> dict:
    model.eval()
    all_logits, all_y = [], []

    for batch in loader:
        if use_time:
            # 1,2트
            #x, y, _, tfeat, _, _ = batch
            # valid 사용으로 교체
            x, y, _, tfeat, _, tvalid = batch
            x = x.to(device, non_blocking=True)
            tfeat = tfeat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tvalid = tvalid.to(device, non_blocking=True).float().unsqueeze(1)  # (B,1)
            tfeat = tfeat * tvalid
            logits = model(x, tfeat)
            #3트
            # x, y, _, tfeat, t_idx, tvalid = batch
            # x = x.to(device, non_blocking=True)
            # tfeat = tfeat.to(device, non_blocking=True)
            # t_idx = t_idx.to(device, non_blocking=True)
            # tvalid = tvalid.to(device, non_blocking=True)  # ✅ 추가
            # y = y.to(device, non_blocking=True)
            # logits = model(x, tfeat, t_idx, tvalid)

        else:
            x, y, _ = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    prob = 1 / (1 + np.exp(-logits))

    pred = (prob >= 0.5).astype(np.float32)
    eps = 1e-8
    tp = (pred * y).sum(axis=0)
    fp = (pred * (1 - y)).sum(axis=0)
    fn = ((1 - pred) * y).sum(axis=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    ap_per_tool = {}
    for i, name in enumerate(TOOL_COLS):
        if y[:, i].sum() == 0:
            ap = float("nan")
        else:
            ap = float(average_precision_score(y[:, i], prob[:, i]))
        ap_per_tool[name] = ap

    mAP = float(np.nanmean(list(ap_per_tool.values())))
    micro_auprc = float(average_precision_score(y.ravel(), prob.ravel()))

    auroc_per_tool = {}
    for i, name in enumerate(TOOL_COLS):
        if len(np.unique(y[:, i])) < 2:
            au = float("nan")
        else:
            au = float(roc_auc_score(y[:, i], prob[:, i]))
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


# -------------------------
# Dataset with precomputed timepred
# -------------------------
@dataclass(frozen=True)
class SampleB:
    vid: str
    t: int
    img_path: Path
    y: np.ndarray
    phase_id: int


def load_timepreds(timepred_root: Path, split_json: Path, split: str) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load time predictions from:
      timepred_root/{split}/{vid}.pt
    """
    split_map = json.loads(Path(split_json).read_text())
    vids = split_map[split]

    split_dir = timepred_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")

    out: dict[str, dict[str, torch.Tensor]] = {}
    for vid in vids:
        p = split_dir / f"{vid}.pt"
        obj = torch.load(p, map_location="cpu")

        if isinstance(obj, dict) and "pred" in obj and "valid" in obj:
            pred = obj["pred"].to(dtype=torch.float32)
            valid = obj["valid"].to(dtype=torch.bool)
        elif torch.is_tensor(obj):
            # 옛날 파일 호환: valid는 전부 True로 처리
            pred = obj.to(dtype=torch.float32)
            valid = torch.ones((pred.shape[0],), dtype=torch.bool)
        else:
            raise RuntimeError(f"{p} must be Tensor or dict(pred, valid)")

        out[vid] = {"pred": pred, "valid": valid}
    return out


class Cholec80ToolDatasetWithTimePred(Dataset):
    def __init__(
        self,
        cholec80_dir: Path,
        labels_dir: Path,
        split_json: Path,
        split: str,
        timepreds: dict[str, dict[str, torch.Tensor]],
        stride: int = 1,
        transform=None,
    ):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.transform = transform
        self.samples: list[SampleB] = []
        self.timepreds = timepreds

        split_map = json.loads(split_json.read_text())
        videos = split_map[split]

        for vid in videos:
            frames_dir = cholec80_dir / "frames" / vid
            npz_path = labels_dir / f"{vid}_labels.npz"
            frames = sorted(frames_dir.glob("*.png"))
            data = np.load(npz_path, allow_pickle=False)

            tools = data["tools_1fps"].astype(np.float32)
            phase_ids = data["phase_ids_1fps"].astype(np.int64)
            T = int(data["T_1fps"][0])

            if len(frames) != T:
                raise RuntimeError(f"{vid}: frames={len(frames)} != T_1fps={T}")
            if vid not in self.timepreds:
                raise RuntimeError(f"Missing timepreds for {vid}")
            if int(self.timepreds[vid]["pred"].shape[0]) != T:
                raise RuntimeError(f"{vid}: timepred_len={self.timepreds[vid]['pred'].shape[0]} != T_1fps={T}")

            for t in range(0, T, stride):
                self.samples.append(
                    SampleB(
                        vid=vid,
                        t=t,
                        img_path=frames[t],
                        y=tools[t],
                        phase_id=int(phase_ids[t]),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = torch.from_numpy(s.y)  # (7,)
        phase_id = torch.tensor(s.phase_id)

        tp = self.timepreds[s.vid]
        tfeat = tp["pred"][s.t].to(dtype=torch.float32)      # (15,)
        tvalid = tp["valid"][s.t]                           # bool (scalar)
        t_idx = torch.tensor(float(s.t), dtype=torch.float32)

        return img, y, phase_id, tfeat, t_idx, tvalid


        # 1) 1,2,트
        # tfeat = self.timepreds[s.vid][s.t].to(dtype=torch.float32)  # (Ft,)
        # return img, y, phase_id, tfeat


class Cholec80ToolDatasetWithOracleTime(Dataset):
    def __init__(
        self,
        cholec80_dir: Path,
        labels_dir: Path,
        split_json: Path,
        split: str,
        stride: int = 1,
        transform=None,
    ):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.transform = transform
        self.samples: list[SampleB] = []
        self.labels_dir = Path(labels_dir)

        split_map = json.loads(Path(split_json).read_text())
        videos = split_map[split]

        # 미리 각 vid의 npz를 keep (IO 줄이기)
        self.npz_map: dict[str, np.lib.npyio.NpzFile] = {}

        for vid in videos:
            frames_dir = Path(cholec80_dir) / "frames" / vid
            npz_path = self.labels_dir / f"{vid}_labels.npz"
            frames = sorted(frames_dir.glob("*.png"))
            npz = np.load(npz_path, allow_pickle=False)
            self.npz_map[vid] = npz

            tools = npz["tools_1fps"].astype(np.float32)
            phase_ids = npz["phase_ids_1fps"].astype(np.int64)
            T = int(npz["T_1fps"][0])

            if len(frames) != T:
                raise RuntimeError(f"{vid}: frames={len(frames)} != T_1fps={T}")

            # oracle feature에 필요한 키 체크
            for k in ["rt_current", "bounds_rem"]:
                if k not in npz.files:
                    raise RuntimeError(f"{vid}: missing '{k}' in labels npz (needed for oracle time)")

            for t in range(0, T, stride):
                self.samples.append(
                    SampleB(
                        vid=vid,
                        t=t,
                        img_path=frames[t],
                        y=tools[t],
                        phase_id=int(phase_ids[t]),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = torch.from_numpy(s.y)  # (7,)
        phase_id = torch.tensor(s.phase_id)

        npz = self.npz_map[s.vid]
        tfeat, tvalid = build_oracle_timefeat_from_npz(npz, s.t)    # (15,), bool
        t_idx = torch.tensor(float(s.t), dtype=torch.float32)

        return img, y, phase_id, tfeat, t_idx, tvalid





# -------------------------
# Model: image + timepred
# -------------------------
# 1) 1트 : 앞에다 fusion하는 방법
class ResNet18ToolDetWithTimePred(nn.Module):
    def __init__(
        self,
        time_feat_dim: int,
        num_tools: int = 7,
        pretrained: bool = True,
        dropout: float = 0.2,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        if pretrained:
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            m = resnet18(weights=None)

        self.backbone = nn.Sequential(*list(m.children())[:-1])  # output (B,512,1,1)
        self.img_dim = m.fc.in_features  # 512

        self.time_emb = nn.Sequential(
            nn.LayerNorm(time_feat_dim),
            nn.Linear(time_feat_dim, time_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(self.img_dim + time_emb_dim, self.img_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.img_dim, num_tools),
        )

    def forward(self, x: torch.Tensor, tfeat: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x).flatten(1)  # (B,512)
        t = self.time_emb(tfeat)         # (B,time_emb_dim)
        z = torch.cat([f, t], dim=1)
        return self.head(z)

# 2트) 뒤에다 fusion 하는 방법
# class ResNet18ToolDetWithTimePred(nn.Module):
#     def __init__(
#         self,
#         time_feat_dim: int,
#         num_tools: int = 7,
#         pretrained: bool = True,
#         dropout: float = 0.2,
#         time_emb_dim: int = 64,
#         time_drop_p: float = 0.0,  # (옵션) 0.7 추천
#     ):
#         super().__init__()
#         if pretrained:
#             m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         else:
#             m = resnet18(weights=None)

#         self.backbone = nn.Sequential(*list(m.children())[:-1])  # (B,512,1,1)
#         self.img_dim = m.fc.in_features  # 512
#         self.time_drop_p = float(time_drop_p)

#         self.head_img = nn.Sequential(
#             nn.Linear(self.img_dim, self.img_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.img_dim, num_tools),
#         )

#         self.head_time = nn.Sequential(
#             nn.LayerNorm(time_feat_dim),
#             nn.Linear(time_feat_dim, time_emb_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(time_emb_dim, num_tools),
#         )
#         self.time_drop_p = float(time_drop_p)   # 0.5~0.8 추천

#     def forward(self, x: torch.Tensor, tfeat: torch.Tensor) -> torch.Tensor:
#         f = self.backbone(x).flatten(1)

#         if self.training and self.time_drop_p > 0:
#             keep = (torch.rand(tfeat.size(0), 1, device=tfeat.device) > self.time_drop_p).float()
#             tfeat = tfeat * keep

#         return self.head_img(f) + self.head_time(tfeat)

# 3트
class ResNet18ToolDetWithTimeSummary(nn.Module):
    """
    Timed model using ONLY 2 scalars from timepred:
      p = progress = t / predicted_total_end
      r = log1p(rt_current / 60)
    Then add per-tool linear corrections: logits += w_p * p + w_r * r
    """
    def __init__(self, num_tools: int = 7, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # (B,512,1,1)
        self.img_dim = m.fc.in_features

        self.head_img = nn.Sequential(
            nn.Linear(self.img_dim, self.img_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.img_dim, num_tools),
        )

        # tool별로 progress/remaining이 logits에 주는 영향 (파라미터 매우 작음)
        self.w_p = nn.Parameter(torch.zeros(num_tools))
        self.w_r = nn.Parameter(torch.zeros(num_tools))

    def forward(self, x: torch.Tensor, tfeat: torch.Tensor, t_idx: torch.Tensor, tvalid: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x).flatten(1)          # (B,512)
        logits = self.head_img(f)                # (B,7)

        mask = tvalid.to(dtype=torch.float32).view(-1)

        rt_cur = torch.clamp(tfeat[:, 0], min=0.0)
        r = torch.log1p(rt_cur / 60.0) * mask

        ends = tfeat[:, 2::2]
        total_rem = ends.max(dim=1).values.clamp(min=1.0)

        p = (t_idx.view(-1) / (t_idx.view(-1) + total_rem)).clamp(0.0, 1.0) * mask

        # tool별 선형 보정
        logits = logits + p.unsqueeze(1) * self.w_p.unsqueeze(0) + r.unsqueeze(1) * self.w_r.unsqueeze(0)
        return logits




# -------------------------
# Train
# -------------------------
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
    ap.add_argument("--stride_train", type=int, default=2)
    ap.add_argument("--stride_val", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None, help="path to checkpoint (.pt) to resume from")

    # ---- timed model options ----
    ap.add_argument(
        "--time_source",
        type=str,
        default="none",
        choices=["none", "pred", "oracle"],
        help="none: baseline, pred: use saved timepreds, oracle: build GT time features from labels npz on-the-fly"
    )
    ap.add_argument("--timepred_root", type=str, default="", help="required if time_source=pred")
    ap.add_argument("--time_emb_dim", type=int, default=64)
    ap.add_argument("--time_drop_p", type=float, default=0.0)

    # ---- backbone freezing options ----
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--unfreeze_layer4", action="store_true")


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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # -------------------------
    # Datasets
    # -------------------------
    use_time = (args.time_source != "none")
    use_pred = (args.time_source == "pred")
    use_oracle = (args.time_source == "oracle")

    if use_pred:
        if not args.timepred_root:
            raise ValueError("--timepred_root is required when time_source=pred")

        timepred_root = Path(args.timepred_root)

        preds_train = load_timepreds(timepred_root, split_json, "train")
        preds_val   = load_timepreds(timepred_root, split_json, "val")

        ds_tr = Cholec80ToolDatasetWithTimePred(
            cholec80_dir, labels_dir, split_json, "train",
            timepreds=preds_train, stride=args.stride_train, transform=tfm_train
        )
        ds_va = Cholec80ToolDatasetWithTimePred(
            cholec80_dir, labels_dir, split_json, "val",
            timepreds=preds_val, stride=args.stride_val, transform=tfm_eval
        )
    elif use_oracle:
        ds_tr = Cholec80ToolDatasetWithOracleTime(
            cholec80_dir, labels_dir, split_json, "train",
            stride=args.stride_train, transform=tfm_train
        )
        ds_va = Cholec80ToolDatasetWithOracleTime(
            cholec80_dir, labels_dir, split_json, "val",
            stride=args.stride_val, transform=tfm_eval
        )
    else:
        ds_tr = Cholec80ToolDataset(cholec80_dir, labels_dir, split_json, "train", stride=args.stride_train, transform=tfm_train)
        ds_va = Cholec80ToolDataset(cholec80_dir, labels_dir, split_json, "val", stride=args.stride_val, transform=tfm_eval)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # -------------------------
    # Model
    # -------------------------
    if use_time:
        sample = ds_tr[0]
        time_feat_dim = int(sample[3].shape[-1])
        # model = ResNet18ToolDetWithTimeSummary(
        #     num_tools=7,
        #     pretrained=True,
        #     dropout=0.2,
        # ).to(device)
        model = ResNet18ToolDetWithTimePred(
            time_feat_dim=time_feat_dim,
            num_tools=7,
            pretrained=True,
            dropout=0.2,
            time_emb_dim=args.time_emb_dim,
        ).to(device)

        if args.freeze_backbone:
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True

        if args.unfreeze_layer4:
            for p in model.backbone[7].parameters():
                p.requires_grad = True

    else:
        model = ResNet18ToolDet(num_tools=7, pretrained=True).to(device)
        # keep your baseline style: train only fc
        for p in model.net.parameters():
            p.requires_grad = False
        for p in model.net.fc.parameters():
            p.requires_grad = True

    # -------------------------
    # Loss / optim
    # -------------------------
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

    # resume
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

    # -------------------------
    # Train loop
    # -------------------------
    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)

            if use_time:
                # 1트
                # x, y, _, tfeat, _, _ = batch
                # valid사용
                x, y, _, tfeat, _, tvalid = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                tfeat = tfeat.to(device, non_blocking=True)
                tvalid = tvalid.to(device, non_blocking=True).float().unsqueeze(1)
                tfeat = tfeat * tvalid



                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(x, tfeat)
                    loss = criterion(logits, y)
                # 3트
                # x, y, _, tfeat, t_idx, tvalid = batch
                # x = x.to(device, non_blocking=True)
                # y = y.to(device, non_blocking=True)
                # tfeat = tfeat.to(device, non_blocking=True)
                # t_idx = t_idx.to(device, non_blocking=True)
                # tvalid = tvalid.to(device, non_blocking=True)

                # with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                #     logits = model(x, tfeat, t_idx, tvalid)
                #     loss = criterion(logits, y)


            else:
                x, y, _ = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(x)
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item()) * x.size(0)
            n += x.size(0)

        tr_loss = running / max(n, 1)
        metrics = evaluate(model, dl_va, device, use_time)
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
