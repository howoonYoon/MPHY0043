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

def build_oracle_timefeat_from_npz(npz: np.lib.npyio.NpzFile, t: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      tfeat: (15,) float32 = [rt_current, bounds_rem(7x2).flatten()]
      tvalid: scalar bool (always True for oracle; bounds parts are masked using mask_bounds if present)
    """
    # rt_current: (T,1) or (T,)
    rt = npz["rt_current"].astype(np.float32)
    rt_t = float(rt[t] if rt.ndim == 1 else rt[t, 0])

    rt_t = np.log1p(rt_t)
    b = np.log1p(np.maximum(b, 0.0))

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

@dataclass(frozen=True)
class Sample:
    vid: str
    img_path: Path
    y: np.ndarray       # (7,)
    phase_id: int       # scalar

@dataclass(frozen=True)
class SampleB:
    vid: str
    t: int
    img_path: Path
    y: np.ndarray
    phase_id: int


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
                self.samples.append(Sample(vid=vid, img_path=frames[t], y=tools[t], phase_id=int(phase_ids[t])))


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = torch.from_numpy(s.y)                 # (7,)
        phase_id = torch.tensor(s.phase_id)       # ()
        return img, y, phase_id, s.vid





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
        return img, y, phase_id, tfeat, t_idx, tvalid, s.vid



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
        return img, y, phase_id, tfeat, t_idx, tvalid, s.vid



