from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import json
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms.functional as TF


TOOL_COLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]

def _sorted_pngs(frames_dir: Path) -> List[Path]:
    return sorted(frames_dir.glob("*.png"))

@dataclass(frozen=True)
class Sample:
    img_path: Path
    y: np.ndarray       # (7,)
    phase_id: int       # scalar

class Cholec80ToolDataset(Dataset):
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
        self.samples: List[Sample] = []

        split_map = json.loads(split_json.read_text())
        videos = split_map[split]

        for vid in videos:
            frames_dir = cholec80_dir / "frames" / vid
            npz_path = labels_dir / f"{vid}_labels.npz"

            frames = _sorted_pngs(frames_dir)
            data = np.load(npz_path, allow_pickle=False)

            tools = data["tools_1fps"].astype(np.float32)          # (T,7)
            phase_ids = data["phase_ids_1fps"].astype(np.int64)    # (T,)
            T = int(data["T_1fps"][0])

            if len(frames) != T:
                raise RuntimeError(f"{vid}: frames={len(frames)} != T_1fps={T}")

            for t in range(0, T, stride):
                self.samples.append(Sample(img_path=frames[t], y=tools[t], phase_id=int(phase_ids[t])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = torch.from_numpy(s.y)                 # (7,)
        phase_id = torch.tensor(s.phase_id)       # ()
        return img, y, phase_id

def default_image_transform(img: Image.Image, size: int = 224) -> torch.Tensor:
    # torchvision 없이 간단히 (PIL->Tensor, resize, normalize)
    img = img.resize((size, size))
    x = torch.from_numpy(np.array(img)).float() / 255.0  # (H,W,3)
    if x.ndim == 2:
        x = x.unsqueeze(-1).repeat(1, 1, 3)
    x = x.permute(2, 0, 1)  # (3,H,W)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    return x

