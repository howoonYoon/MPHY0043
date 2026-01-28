#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
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


PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderRetraction",
    "CleaningCoagulation",
    "GallbladderPackaging",
]


def ap_per_class(y: np.ndarray, prob: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(TOOL_COLS):
        if y[:, i].sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(average_precision_score(y[:, i], prob[:, i]))
    return out


# -------------------------
# Oracle time feature
# -------------------------
def build_oracle_timefeat_from_npz(npz: np.lib.npyio.NpzFile, t: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      tfeat: (15,) float32 = [rt_current, bounds_rem(7x2).flatten()]
      tvalid: scalar bool (always True for oracle; bounds parts are masked using mask_bounds if present)
    """
    rt = npz["rt_current"].astype(np.float32)
    rt_t = float(rt[t] if rt.ndim == 1 else rt[t, 0])

    bounds = npz["bounds_rem"].astype(np.float32)  # (T,7,2) or (T,14)
    b = bounds[t].reshape(-1) if bounds.ndim != 3 else bounds[t].reshape(-1)  # (14,)

    if "mask_bounds" in npz.files:
        mb = npz["mask_bounds"].astype(np.float32)
        mb_t = mb[t].reshape(-1)  # (14,)
        b = b * mb_t

    feat = np.concatenate([[rt_t], b], axis=0).astype(np.float32)  # (15,)
    tfeat = torch.from_numpy(feat)
    tvalid = torch.tensor(True)
    return tfeat, tvalid


@dataclass(frozen=True)
class SampleB:
    vid: str
    t: int
    img_path: Path
    y: np.ndarray
    phase_id: int


# -------------------------
# Load timepreds (pred + valid)
# -------------------------
def load_timepreds(timepred_root: Path, split_json: Path, split: str) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load time predictions from:
      timepred_root/{split}/{vid}.pt

    Supports:
      - dict with {"pred": Tensor(T,F), "valid": BoolTensor(T,)}
      - Tensor(T,F)  (then valid is all True)
    """
    split_map = json.loads(Path(split_json).read_text())
    vids = split_map[split]

    split_dir = timepred_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")

    out: dict[str, dict[str, torch.Tensor]] = {}
    for vid in vids:
        p = split_dir / f"{vid}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing timepred file: {p}")

        obj = torch.load(p, map_location="cpu")

        if isinstance(obj, dict) and "pred" in obj and "valid" in obj:
            pred = obj["pred"].to(dtype=torch.float32)
            valid = obj["valid"].to(dtype=torch.bool)
        elif torch.is_tensor(obj):
            pred = obj.to(dtype=torch.float32)
            valid = torch.ones((pred.shape[0],), dtype=torch.bool)
        else:
            raise RuntimeError(f"{p} must be Tensor or dict(pred, valid)")

        out[vid] = {"pred": pred, "valid": valid}
    return out


# -------------------------
# Dataset: predicted time
# -------------------------
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

        y = torch.from_numpy(s.y)
        phase_id = torch.tensor(s.phase_id)

        tp = self.timepreds[s.vid]
        tfeat = tp["pred"][s.t].to(dtype=torch.float32)   # (F,)
        tvalid = tp["valid"][s.t]                         # bool scalar
        t_idx = torch.tensor(float(s.t), dtype=torch.float32)

        return img, y, phase_id, tfeat, t_idx, tvalid


# -------------------------
# Dataset: oracle time (GT)
# -------------------------
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

        # 안정성: npz 객체를 들고 있지 말고 path만 저장
        self.npz_path_map: dict[str, Path] = {}

        for vid in videos:
            frames_dir = Path(cholec80_dir) / "frames" / vid
            npz_path = self.labels_dir / f"{vid}_labels.npz"
            frames = sorted(frames_dir.glob("*.png"))
            data = np.load(npz_path, allow_pickle=False)

            T = int(data["T_1fps"][0])
            if len(frames) != T:
                raise RuntimeError(f"{vid}: frames={len(frames)} != T_1fps={T}")

            for k in ["rt_current", "bounds_rem"]:
                if k not in data.files:
                    raise RuntimeError(f"{vid}: missing '{k}' in labels npz (needed for oracle time)")

            tools = data["tools_1fps"].astype(np.float32)
            phase_ids = data["phase_ids_1fps"].astype(np.int64)

            self.npz_path_map[vid] = npz_path

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

        y = torch.from_numpy(s.y)
        phase_id = torch.tensor(s.phase_id)

        npz = np.load(self.npz_path_map[s.vid], allow_pickle=False)
        tfeat, tvalid = build_oracle_timefeat_from_npz(npz, s.t)
        t_idx = torch.tensor(float(s.t), dtype=torch.float32)

        return img, y, phase_id, tfeat, t_idx, tvalid


# -------------------------
# Model: image + timefeat
# -------------------------
class ResNet18ToolDetWithTimePred(nn.Module):
    def __init__(self, time_feat_dim: int, num_tools: int = 7, pretrained: bool = True, dropout: float = 0.2, time_emb_dim: int = 64):
        super().__init__()
        if pretrained:
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            m = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.img_dim = m.fc.in_features
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
        f = self.backbone(x).flatten(1)
        t = self.time_emb(tfeat)
        z = torch.cat([f, t], dim=1)
        return self.head(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cholec80_dir", type=str, required=True)
    ap.add_argument("--labels_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out_json", type=str, default="eval_taskB.json")

    # unified timed option
    ap.add_argument(
        "--time_source",
        type=str,
        default="none",
        choices=["none", "pred", "oracle"],
        help="none: baseline, pred: use saved timepreds, oracle: build GT time features from labels npz on-the-fly"
    )
    ap.add_argument("--timepred_root", type=str, default="", help="required if time_source=pred")
    ap.add_argument("--time_emb_dim", type=int, default=64)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cholec80_dir = Path(args.cholec80_dir)
    labels_dir = Path(args.labels_dir)
    split_json = Path(args.split_json)

    use_time = (args.time_source != "none")
    use_pred = (args.time_source == "pred")
    use_oracle = (args.time_source == "oracle")

    # dataset
    if use_pred:
        if not args.timepred_root:
            raise ValueError("--timepred_root is required when time_source=pred")
        preds = load_timepreds(Path(args.timepred_root), split_json, args.split)
        ds = Cholec80ToolDatasetWithTimePred(
            cholec80_dir, labels_dir, split_json, args.split,
            timepreds=preds, stride=args.stride, transform=tfm
        )
    elif use_oracle:
        # oracle IO 많음: 기본값 workers=4면 느리거나 문제가 날 수 있음.
        # 필요하면 CLI로 --num_workers 0 줘.
        ds = Cholec80ToolDatasetWithOracleTime(
            cholec80_dir, labels_dir, split_json, args.split,
            stride=args.stride, transform=tfm
        )
    else:
        ds = Cholec80ToolDataset(cholec80_dir, labels_dir, split_json, args.split, stride=args.stride, transform=tfm)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # model
    if use_time:
        sample = ds[0]
        time_feat_dim = int(sample[3].shape[-1])
        model = ResNet18ToolDetWithTimePred(
            time_feat_dim=time_feat_dim, num_tools=7, pretrained=False, dropout=0.2, time_emb_dim=args.time_emb_dim
        )
    else:
        model = ResNet18ToolDet(num_tools=7, pretrained=False)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_logits, all_y, all_phase = [], [], []
    with torch.no_grad():
        for batch in dl:
            if use_time:
                x, y, phase_id, tfeat, _, tvalid = batch
                x = x.to(device, non_blocking=True)
                tfeat = tfeat.to(device, non_blocking=True)
                tvalid = tvalid.to(device, non_blocking=True).float().unsqueeze(1)  # (B,1)
                tfeat = tfeat * tvalid
                logits = model(x, tfeat).detach().cpu()
            else:
                x, y, phase_id = batch
                x = x.to(device, non_blocking=True)
                logits = model(x).detach().cpu()

            all_logits.append(logits)
            all_y.append(y)
            all_phase.append(phase_id)

    logits = torch.cat(all_logits, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    phase = torch.cat(all_phase, dim=0).numpy()

    prob = 1 / (1 + np.exp(-logits))
    pred = (prob >= 0.5).astype(np.float32)

    eps = 1e-8
    tp = (pred * y).sum(axis=0)
    fp = (pred * (1 - y)).sum(axis=0)
    fn = ((1 - pred) * y).sum(axis=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    ap_per_tool = ap_per_class(y, prob)
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

    phasewise = {}
    for p in range(7):
        idx = (phase == p)
        if idx.sum() == 0:
            continue
        y_p = y[idx]
        prob_p = prob[idx]
        AP_p = ap_per_class(y_p, prob_p)
        mAP_p = float(np.nanmean(list(AP_p.values())))
        phasewise[PHASE_NAMES[p]] = {
            "n_samples": int(idx.sum()),
            "mAP(AUPRC_macro)": mAP_p,
            "AP_per_tool": AP_p,
        }

    result = {
        "split": args.split,
        "stride": args.stride,
        "n_samples": int(len(ds)),
        "time_source": args.time_source,
        "mAP(AUPRC_macro)": mAP,
        "AUPRC_micro": micro_auprc,
        "AP_per_tool": ap_per_tool,
        "AUROC_macro": macro_auroc,
        "AUROC_per_tool": auroc_per_tool,
        "macro_f1@0.5": float(np.mean(f1)),
        "per_tool_f1@0.5": {TOOL_COLS[i]: float(f1[i]) for i in range(7)},
        "phasewise": phasewise,
    }

    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
