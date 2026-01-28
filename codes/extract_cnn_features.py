#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


class FrameFolderDataset(Dataset):
    def __init__(self, frames_dir: Path, img_size: int = 224):
        self.frames_dir = frames_dir
        self.paths = sorted(frames_dir.glob("*.png"))
        if len(self.paths) == 0:
            raise RuntimeError(f"No png frames found in: {frames_dir}")

        weights = ResNet18_Weights.DEFAULT
        mean = weights.transforms().mean
        std = weights.transforms().std

        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        return x


class ResNet18Feature(nn.Module):
    """Return 512-d feature after avgpool, before fc."""
    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
            m.avgpool
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) -> (B,512)
        f = self.stem(x)              # (B,512,1,1)
        f = f.flatten(1)              # (B,512)
        return f


@torch.no_grad()
def extract_video(frames_dir: Path, out_path: Path, device: torch.device,
                  batch_size: int = 64, img_size: int = 224, num_workers: int = 0,
                  dtype: str = "float16") -> None:
    ds = FrameFolderDataset(frames_dir, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    model = ResNet18Feature().to(device)
    model.eval()

    feats = []
    for xb in dl:
        xb = xb.to(device, non_blocking=False)
        fb = model(xb)  # (B,512)
        feats.append(fb.cpu())

    F = torch.cat(feats, dim=0).numpy()  # (T,512)
    if dtype == "float16":
        F = F.astype(np.float16)
    else:
        F = F.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, F)
    print(f"saved: {out_path}  shape={F.shape} dtype={F.dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", type=str, required=True,
                    help=".../cholec80/frames (contains videoXX folders)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="where to save features, e.g. .../features_1fps")
    ap.add_argument("--video", type=str, default="all",
                    help="videoXX or 'all'")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    out_dir = Path(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.video != "all":
        vids = [args.video]
    else:
        vids = sorted([p.name for p in frames_root.glob("video*") if p.is_dir()])

    for vid in vids:
        frames_dir = frames_root / vid
        out_path = out_dir / f"{vid}_resnet18.npy"
        if out_path.exists() and not args.overwrite:
            print(f"skip (exists): {out_path}")
            continue
        print(f"[{vid}] extracting from {frames_dir}")
        extract_video(frames_dir, out_path, device,
                      batch_size=args.batch, img_size=args.img_size,
                      num_workers=args.num_workers, dtype=args.dtype)


if __name__ == "__main__":
    main()
