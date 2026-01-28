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


class TaskATimeOnlyDataset(Dataset):
    """
    Input:  x = [t/T] (T,1)
    Target: y = [rt_current, bounds_rem_flat] (T,15) in log1p(seconds)
    Mask:   m = [1, mask_bounds_flat] (T,15)
    """
    def __init__(self, npz_path: Path):
        d = np.load(npz_path)

        T = int(d["T_1fps"][0])
        t = np.arange(T, dtype=np.float32)
        x = (t / max(T - 1, 1)).reshape(-1, 1)

        rt = d["rt_current"].astype(np.float32).reshape(T, 1)
        bounds = d["bounds_rem"].astype(np.float32).reshape(T, 7, 2)
        mask_b = d["mask_bounds"].astype(np.float32).reshape(T, 7, 2)

        y = np.concatenate([rt, bounds.reshape(T, 14)], axis=1)  # (T,15)
        m = np.concatenate([np.ones((T, 1), np.float32), mask_b.reshape(T, 14)], axis=1)

        self.x = torch.from_numpy(x).float()
        self.y_log = torch.from_numpy(np.log1p(y)).float()
        self.m = torch.from_numpy(m).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_log[idx], self.m[idx]


class TaskATimeSeqDataset(Dataset):
    """
    Time-only sequence dataset for Task A.

    Input:
      x_seq: (SEQ_LEN, F) where F=2 by default [u, u^2], u=t/(T-1)
    Target:
      y_log: (15,) log1p seconds at current time t (the last step of the window)
    Mask:
      m:     (15,) [1, mask_bounds_flat] at time t
    """
    def __init__(self, npz_path: Path, seq_len: int = 120, feat_dim: int = 2):
        assert feat_dim in (1, 2), "feat_dim should be 1 (u) or 2 ([u, u^2])"
        d = np.load(npz_path)

        T = int(d["T_1fps"][0])
        t = np.arange(T, dtype=np.float32)
        u = t / max(T - 1, 1)

        if feat_dim == 1:
            X = u.reshape(-1, 1).astype(np.float32)            # (T,1)
        else:
            X = np.stack([u, u * u], axis=1).astype(np.float32) # (T,2)

        rt = d["rt_current"].astype(np.float32).reshape(T, 1)          # (T,1)
        bounds = d["bounds_rem"].astype(np.float32).reshape(T, 7, 2)   # (T,7,2)
        mask_b = d["mask_bounds"].astype(np.float32).reshape(T, 7, 2)  # (T,7,2)

        Y = np.concatenate([rt, bounds.reshape(T, 14)], axis=1)  # (T,15)
        M = np.concatenate([np.ones((T, 1), np.float32), mask_b.reshape(T, 14)], axis=1)  # (T,15)

        self.X = X
        self.Y_log = np.log1p(Y).astype(np.float32)
        self.M = M.astype(np.float32)

        self.seq_len = int(seq_len)
        self.T = T

        # We'll create one sample per time t (0..T-1)
        # Each sample uses a left-padded window ending at t.
        self.indices = np.arange(T, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        L = self.seq_len

        start = t - L + 1
        if start >= 0:
            x_seq = self.X[start:t+1]  # (L,F)
        else:
            # left pad with first frame features
            pad = np.repeat(self.X[0:1], repeats=-start, axis=0)
            x_seq = np.concatenate([pad, self.X[0:t+1]], axis=0)  # (L,F)

        # safety
        if x_seq.shape[0] != L:
            # in case of any mismatch, pad/trim to L
            if x_seq.shape[0] < L:
                pad = np.repeat(x_seq[0:1], repeats=(L - x_seq.shape[0]), axis=0)
                x_seq = np.concatenate([pad, x_seq], axis=0)
            else:
                x_seq = x_seq[-L:]

        y_log = self.Y_log[t]  # (15,)
        m = self.M[t]          # (15,)

        return torch.from_numpy(x_seq).float(), torch.from_numpy(y_log).float(), torch.from_numpy(m).float()



class TaskATimeSeqDatasetWithPhase(Dataset):
    """
    Task A sequence dataset (time features + phase features), no images.

    Returns:
      x_seq: (L, F)
      y_log: (15,)
      m:     (15,)

    Base time features (feat_dim):
      1 -> [u]
      2 -> [u, u^2]
      3 -> [u, u^2, t_norm]

    Phase features:
      - phase one-hot from 'phase_ids_1fps' (T, 7)
      - optionally include 'elapsed_phase' as 1 extra dim
    """

    def __init__(
        self,
        npz_path: Path,
        seq_len: int = 180,
        feat_dim: int = 2,
        use_phase_onehot: bool = True,
        use_elapsed_phase: bool = False,
        n_phases: int = 7,
    ):
        assert feat_dim in (1, 2, 3)
        self.seq_len = int(seq_len)

        d = np.load(npz_path)
        T = int(d["T_1fps"][0])
        self.T = T

        # ----- base time features -----
        t = np.arange(T, dtype=np.float32)
        u = t / max(T - 1, 1)
        u2 = u * u
        t_norm = np.clip(t / 6000.0, 0.0, 2.0)

        if feat_dim == 1:
            X = u.reshape(-1, 1)
        elif feat_dim == 2:
            X = np.stack([u, u2], axis=1)
        else:
            X = np.stack([u, u2, t_norm], axis=1)

        # ----- phase one-hot -----
        if use_phase_onehot:
            phase_id = d["phase_ids_1fps"].astype(np.int64)[:T]  # (T,)
            # safety: clamp to valid range
            phase_id = np.clip(phase_id, 0, n_phases - 1)
            phase_oh = np.eye(n_phases, dtype=np.float32)[phase_id]  # (T,7)
            X = np.concatenate([X, phase_oh], axis=1)

        # ----- elapsed phase (optional) -----
        if use_elapsed_phase:
            ep = d["elapsed_phase"].astype(np.float32)[:T].reshape(-1, 1)  # (T,1)
            # optional normalization (keeps scale reasonable)
            ep = np.clip(ep / 600.0, 0.0, 10.0)  # 10min cap (tune if you want)
            X = np.concatenate([X, ep], axis=1)

        self.X = X.astype(np.float32)

        # ----- targets/masks at 1fps -----
        rt = d["rt_current"].astype(np.float32).reshape(-1, 1)[:T]
        bounds = d["bounds_rem"].astype(np.float32).reshape(-1, 7, 2)[:T]
        mask_b = d["mask_bounds"].astype(np.float32).reshape(-1, 7, 2)[:T]

        Y = np.concatenate([rt, bounds.reshape(T, 14)], axis=1)  # (T,15)
        M = np.concatenate([np.ones((T, 1), np.float32), mask_b.reshape(T, 14)], axis=1)

        self.Y_log = np.log1p(Y).astype(np.float32)
        self.M = M.astype(np.float32)

        self.indices = np.arange(T, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        L = self.seq_len

        start = t - L + 1
        if start >= 0:
            ids = np.arange(start, t + 1, dtype=np.int64)
        else:
            pad_n = -start
            ids = np.concatenate([np.zeros(pad_n, dtype=np.int64), np.arange(0, t + 1, dtype=np.int64)])
            ids = ids[-L:]

        x_seq = torch.from_numpy(self.X[ids]).float()      # (L,F)
        y_log = torch.from_numpy(self.Y_log[t]).float()    # (15,)
        m = torch.from_numpy(self.M[t]).float()            # (15,)
        return x_seq, y_log, m


class TaskAImageSeqDataset(Dataset):
    """
    Uses Cholec80 1fps frames + label npz to build sequence samples.
    Each sample corresponds to time t, returns:
      img_seq: (L,3,H,W)
      time_seq:(L,Ft)  Ft=2 default [u, t_norm]
      y_log:   (15,)
      m:       (15,)
    """
    def __init__(
        self,
        npz_path: Path,
        frames_root: Path,   # .../cholec80/frames
        seq_len: int = 60,
        img_size: int = 224,
        time_feat_dim: int = 2,
        stride: int = 1
    ):
        assert time_feat_dim in (1, 2, 3)
        d = np.load(npz_path)

        self.video = str(d["video"])
        self.T = int(d["T_1fps"][0])
        self.seq_len = int(seq_len)
        self.img_size = int(img_size)
        self.time_feat_dim = int(time_feat_dim)
        self.stride = int(stride)

        # frames directory: frames_root/videoXX
        self.frames_dir = frames_root / self.video
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Missing frames dir: {self.frames_dir}")

        # list png paths sorted
        self.frame_paths: List[Path] = sorted(self.frames_dir.glob("*.png"))
        if len(self.frame_paths) != self.T:
            # not fatal, but should match
            # fallback: clamp indexing by min length
            self.T = min(self.T, len(self.frame_paths))

        # build time features (T, Ft)
        t = np.arange(self.T, dtype=np.float32)
        u = t / max(self.T - 1, 1)
        t_norm = np.clip(t / 6000.0, 0.0, 2.0)

        if time_feat_dim == 1:
            X_time = u.reshape(-1, 1)
        elif time_feat_dim == 2:
            X_time = np.stack([u, t_norm], axis=1)
        else:
            X_time = np.stack([u, u * u, t_norm], axis=1)

        self.X_time = X_time.astype(np.float32)

        # targets/masks at 1fps
        rt = d["rt_current"].astype(np.float32).reshape(-1, 1)[:self.T]
        bounds = d["bounds_rem"].astype(np.float32).reshape(-1, 7, 2)[:self.T]
        mask_b = d["mask_bounds"].astype(np.float32).reshape(-1, 7, 2)[:self.T]

        Y = np.concatenate([rt, bounds.reshape(self.T, 14)], axis=1)  # (T,15)
        M = np.concatenate([np.ones((self.T, 1), np.float32), mask_b.reshape(self.T, 14)], axis=1)

        self.Y_log = np.log1p(Y).astype(np.float32)
        self.M = M.astype(np.float32)

        # TaskAImageSeqDataset.__init__ 맨 아래
        self.indices = np.arange(0, self.T, self.stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def _load_img(self, idx: int) -> torch.Tensor:
        img = read_image(str(self.frame_paths[idx])).float() / 255.0   # (C,H,W), 0~1
        if img.shape[0] == 4:
            img = img[:3]          # alpha drop
        elif img.shape[0] == 1:
            img = img.repeat(3,1,1)
        img = TF.resize(img, [self.img_size, self.img_size], antialias=True)

        # ✅ ImageNet normalization (ResNet18 pretrained 필수)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        L = self.seq_len

        start = t - L + 1
        if start >= 0:
            ids = list(range(start, t + 1))
        else:
            # left pad by repeating first frame/time
            pad_n = -start
            ids = [0] * pad_n + list(range(0, t + 1))
            ids = ids[-L:]

        # load images (L,3,H,W)
        img_seq = torch.stack([self._load_img(i) for i in ids], dim=0)

        # time seq (L,Ft)
        time_seq = torch.from_numpy(self.X_time[ids]).float()

        y_log = torch.from_numpy(self.Y_log[t]).float()
        m = torch.from_numpy(self.M[t]).float()

        return img_seq, time_seq, y_log, m


class TaskAFeatSeqDataset(Dataset):
    """
    Uses cached CNN features + time features to build sequence samples.
    Each sample corresponds to time t, returns:
      feat_seq: (L, D)   D=512 by default
      time_seq: (L, Ft) Ft=2 default [u, t_norm]
      y_log:    (15,)
      m:        (15,)
    """
    def __init__(
        self,
        npz_path: Path,
        feat_path: Path,       # e.g. features_1fps/videoXX_resnet18.npy
        seq_len: int = 30,
        time_feat_dim: int = 2,  # 1:[u], 2:[u,t_norm], 3:[u,u^2,t_norm]
    ):
        assert time_feat_dim == 1
        d = np.load(npz_path)

        self.T = int(d["T_1fps"][0])
        self.seq_len = int(seq_len)
        self.time_feat_dim = int(time_feat_dim)

        F = np.load(feat_path)  # (T,512)
        if F.shape[0] != self.T:
            self.T = min(self.T, F.shape[0])
            F = F[:self.T]
        self.F = F.astype(np.float32)

        # time features (Ft=1)
        t = np.arange(self.T, dtype=np.float32)
        t_norm = np.clip(t / 6000.0, 0.0, 2.0)
        X_time = t_norm.reshape(-1, 1).astype(np.float32)   # (T,1)

        self.X_time = X_time

        # targets/masks
        rt = d["rt_current"].astype(np.float32).reshape(-1, 1)[:self.T]
        bounds = d["bounds_rem"].astype(np.float32).reshape(-1, 7, 2)[:self.T]
        mask_b = d["mask_bounds"].astype(np.float32).reshape(-1, 7, 2)[:self.T]

        Y = np.concatenate([rt, bounds.reshape(self.T, 14)], axis=1)  # (T,15)
        M = np.concatenate([np.ones((self.T, 1), np.float32), mask_b.reshape(self.T, 14)], axis=1)

        Y = np.maximum(Y, 0.0)
        self.Y_log = np.log1p(Y).astype(np.float32)
        self.M = M.astype(np.float32)

        self.indices = np.arange(self.seq_len - 1, self.T, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        L = self.seq_len

        start = t - L + 1
        ids = np.arange(start, t + 1, dtype=np.int64)

        feat_seq = torch.from_numpy(self.F[ids]).float()       # (L,512)
        time_seq = torch.from_numpy(self.X_time[ids]).float()  # (L,Ft)
        y_log = torch.from_numpy(self.Y_log[t]).float()        # (15,)
        m = torch.from_numpy(self.M[t]).float()                # (15,)

        return feat_seq, time_seq, y_log, m
