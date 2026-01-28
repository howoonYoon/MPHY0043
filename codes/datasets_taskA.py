from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Literal, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision.io import read_image
import torchvision.transforms.functional as TF


Mode = Literal["time_point", "time_seq", "timephase_seq", "image_seq", "feat_seq"]


def _log1p_safe(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(y, 0.0)).astype(np.float32)


def _build_targets(d: np.lib.npyio.NpzFile, T: int) -> Tuple[np.ndarray, np.ndarray]:
    rt = d["rt_current"].astype(np.float32).reshape(-1, 1)[:T]
    bounds = d["bounds_rem"].astype(np.float32).reshape(-1, 7, 2)[:T]
    mask_b = d["mask_bounds"].astype(np.float32).reshape(-1, 7, 2)[:T]

    Y = np.concatenate([rt, bounds.reshape(T, 14)], axis=1)  # (T,15)
    M = np.concatenate([np.ones((T, 1), np.float32), mask_b.reshape(T, 14)], axis=1)  # (T,15)
    return _log1p_safe(Y), M.astype(np.float32)


def _time_feats(
    T: int,
    feat_dim: int = 1,
    use_u: bool = True,
    use_u2: bool = False,
    use_u3: bool = False,
    use_sin_cos: bool = False,
    use_u_clip: bool = False,
    use_tnorm: bool = True,
    tnorm_div: float = 6000.0,
) -> np.ndarray:
    """
    Returns (T, Ft)
    - u = t/(T-1)
    - u2 = u^2
    - u3 = u^3
    - sin/cos = sin(pi*u), cos(pi*u)
    - u_clip = clip((u-0.5)*2, -1, 1)
    - t_norm = clip(t/tnorm_div, 0, 2)
    """
    t = np.arange(T, dtype=np.float32)
    cols = []

    if use_u:
        u = t / max(T - 1, 1)
        cols.append(u)
        if use_u2:
            cols.append(u * u)
        if use_u3:
            cols.append(u * u * u)
        if use_sin_cos:
            cols.append(np.sin(np.pi * u))
            cols.append(np.cos(np.pi * u))
        if use_u_clip:
            u_clip = np.clip((u - 0.5) * 2.0, -1.0, 1.0)
            cols.append(u_clip)

    if use_tnorm:
        t_norm = np.clip(t / float(tnorm_div), 0.0, 2.0)
        cols.append(t_norm)

    X = np.stack(cols, axis=1).astype(np.float32)  # (T,Ft)
    return X


class TaskADataset(Dataset):
    """
    One dataset for Task A with different modes.

    Returns by mode:
      - time_point:    x, y_log, m
      - time_seq:      x_seq, y_log, m
      - timephase_seq: x_seq, y_log, m
      - image_seq:     img_seq, time_seq, y_log, m
      - feat_seq:      feat_seq, time_seq, y_log, m

    Default time_feat when not specified:
      - time_point: "u"
      - time_seq / timephase_seq: "u_u2"
      - image_seq: "u_tnorm"
      - feat_seq: "tnorm"
    """

    def __init__(
        self,
        npz_path: Path,
        mode: Mode,
        seq_len: int = 120,

        # ---- time feature config ----
        time_feat: Optional[Literal[
            "u",
            "u_u2",
            "tnorm",
            "u_tnorm",
            "u_u2_tnorm",
            "u_u2_u3",
            "u_u2_u3_sin_cos",
            "u_u2_u3_sin_cos_tnorm",
            "u_u2_u3_sin_cos_clip",
        ]] = None,
        tnorm_div: float = 6000.0,

        # ---- phase config ----
        use_phase_onehot: bool = True,
        use_elapsed_phase: bool = False,
        n_phases: int = 7,

        # ---- image config (for image_seq) ----
        frames_root: Optional[Path] = None,
        img_size: int = 224,
        stride: int = 1,

        # ---- cached feature config (for feat_seq) ----
        feat_path: Optional[Path] = None,
    ):
        super().__init__()
        self.mode = mode
        self.seq_len = int(seq_len)
        self.img_size = int(img_size)
        self.stride = int(stride)

        d = np.load(npz_path)
        self.video = str(d["video"]) if "video" in d.files else None
        T = int(d["T_1fps"][0])
        self.T = T

        # targets
        if mode in ("time_point", "time_seq", "timephase_seq", "feat_seq"):
            rt = d["rt_current"].astype(np.float32).reshape(-1, 1)[:T]
            bounds = d["bounds_rem"].astype(np.float32).reshape(-1, 7, 2)[:T]
            mask_b = d["mask_bounds"].astype(np.float32).reshape(-1, 7, 2)[:T]

            Y = np.concatenate([rt, bounds.reshape(T, 14)], axis=1)  # (T,15)
            M = np.concatenate([np.ones((T, 1), np.float32), mask_b.reshape(T, 14)], axis=1)  # (T,15)

            self.Y_log = np.log1p(Y).astype(np.float32)
            self.M = M.astype(np.float32)
        else:
            self.Y_log, self.M = _build_targets(d, T)  # (T,15), (T,15)

        # ----- base time features -----
        if time_feat is None:
            if mode == "time_point":
                time_feat = "u"
            elif mode in ("time_seq", "timephase_seq"):
                time_feat = "u_u2"
            elif mode == "image_seq":
                time_feat = "u_tnorm"
            else:  # feat_seq
                time_feat = "tnorm"

        if time_feat == "u":
            X_time = _time_feats(T, use_u=True, use_u2=False, use_tnorm=False)
        elif time_feat == "u_u2":
            X_time = _time_feats(T, use_u=True, use_u2=True, use_tnorm=False)
        elif time_feat == "u_u2_u3":
            X_time = _time_feats(T, use_u=True, use_u2=True, use_u3=True, use_tnorm=False)
        elif time_feat == "u_u2_u3_sin_cos":
            X_time = _time_feats(T, use_u=True, use_u2=True, use_u3=True, use_sin_cos=True, use_tnorm=False)
        elif time_feat == "u_u2_u3_sin_cos_tnorm":
            X_time = _time_feats(T, use_u=True, use_u2=True, use_u3=True, use_sin_cos=True, use_tnorm=True, tnorm_div=tnorm_div)
        elif time_feat == "u_u2_u3_sin_cos_clip":
            X_time = _time_feats(T, use_u=True, use_u2=True, use_u3=True, use_sin_cos=True, use_u_clip=True, use_tnorm=False)
        elif time_feat == "tnorm":
            X_time = _time_feats(T, use_u=False, use_u2=False, use_tnorm=True, tnorm_div=tnorm_div)
        elif time_feat == "u_tnorm":
            X_time = _time_feats(T, use_u=True, use_u2=False, use_tnorm=True, tnorm_div=tnorm_div)
        else:  # "u_u2_tnorm"
            X_time = _time_feats(T, use_u=True, use_u2=True, use_tnorm=True, tnorm_div=tnorm_div)

        # add phase info (only used in timephase_seq mode)
        if mode == "timephase_seq":
            X = X_time
            if use_phase_onehot:
                phase_id = d["phase_ids_1fps"].astype(np.int64)[:T]
                phase_id = np.clip(phase_id, 0, n_phases - 1)
                phase_oh = np.eye(n_phases, dtype=np.float32)[phase_id]  # (T,7)
                X = np.concatenate([X, phase_oh], axis=1)
            if use_elapsed_phase:
                ep = d["elapsed_phase"].astype(np.float32)[:T].reshape(-1, 1)
                ep = np.clip(ep / 600.0, 0.0, 10.0)
                X = np.concatenate([X, ep], axis=1)
            self.X = X.astype(np.float32)
        else:
            self.X_time = X_time.astype(np.float32)

        # image paths
        if mode == "image_seq":
            if frames_root is None or self.video is None:
                raise ValueError("image_seq requires frames_root and npz must contain 'video'")
            self.frames_dir = Path(frames_root) / self.video
            self.frame_paths: List[Path] = sorted(self.frames_dir.glob("*.png"))
            if len(self.frame_paths) != T:
                self.T = min(self.T, len(self.frame_paths))
                self.frame_paths = self.frame_paths[: self.T]
                self.Y_log = self.Y_log[: self.T]
                self.M = self.M[: self.T]
                if mode == "timephase_seq":
                    self.X = self.X[: self.T]
                else:
                    self.X_time = self.X_time[: self.T]

        # cached feat
        if mode == "feat_seq":
            if feat_path is None:
                raise ValueError("feat_seq requires feat_path")
            F = np.load(feat_path).astype(np.float32)  # (T,512)
            if F.shape[0] != T:
                self.T = min(self.T, F.shape[0])
                F = F[: self.T]
                self.Y_log = self.Y_log[: self.T]
                self.M = self.M[: self.T]
                self.X_time = self.X_time[: self.T]
            self.F = F

        # indices
        if mode in ("time_point", "time_seq", "timephase_seq"):
            self.indices = np.arange(self.T, dtype=np.int64)
        elif mode == "image_seq":
            self.indices = np.arange(0, self.T, self.stride, dtype=np.int64)
        else:  # feat_seq
            self.indices = np.arange(self.seq_len - 1, self.T, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def _get_window_ids(self, t: int) -> np.ndarray:
        L = self.seq_len
        start = t - L + 1
        if start >= 0:
            return np.arange(start, t + 1, dtype=np.int64)
        pad_n = -start
        ids = np.concatenate([np.zeros(pad_n, dtype=np.int64), np.arange(0, t + 1, dtype=np.int64)])
        return ids[-L:]

    def _load_img(self, idx: int) -> torch.Tensor:
        img = read_image(str(self.frame_paths[idx])).float() / 255.0
        if img.shape[0] == 4:
            img = img[:3]
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = TF.resize(img, [self.img_size, self.img_size], antialias=True)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img - mean) / std

    def __getitem__(self, idx: int):
        t = int(self.indices[idx])

        if self.mode == "time_point":
            # x: (Ft,)
            x = torch.from_numpy(self.X_time[t]).float()
            y_log = torch.from_numpy(self.Y_log[t]).float()
            m = torch.from_numpy(self.M[t]).float()
            return x, y_log, m

        if self.mode == "time_seq":
            ids = self._get_window_ids(t)
            x_seq = torch.from_numpy(self.X_time[ids]).float()  # (L,Ft)
            y_log = torch.from_numpy(self.Y_log[t]).float()
            m = torch.from_numpy(self.M[t]).float()
            return x_seq, y_log, m

        if self.mode == "timephase_seq":
            ids = self._get_window_ids(t)
            x_seq = torch.from_numpy(self.X[ids]).float()       # (L,F)
            y_log = torch.from_numpy(self.Y_log[t]).float()
            m = torch.from_numpy(self.M[t]).float()
            return x_seq, y_log, m

        if self.mode == "image_seq":
            ids = self._get_window_ids(t)
            img_seq = torch.stack([self._load_img(int(i)) for i in ids], dim=0)  # (L,3,H,W)
            time_seq = torch.from_numpy(self.X_time[ids]).float()                # (L,Ft)
            y_log = torch.from_numpy(self.Y_log[t]).float()
            m = torch.from_numpy(self.M[t]).float()
            return img_seq, time_seq, y_log, m

        if self.mode == "feat_seq":
            ids = self._get_window_ids(t)
            feat_seq = torch.from_numpy(self.F[ids]).float()       # (L,512)
            time_seq = torch.from_numpy(self.X_time[ids]).float()  # (L,Ft)
            y_log = torch.from_numpy(self.Y_log[t]).float()
            m = torch.from_numpy(self.M[t]).float()
            return feat_seq, time_seq, y_log, m

        raise RuntimeError(f"Unknown mode: {self.mode}")
