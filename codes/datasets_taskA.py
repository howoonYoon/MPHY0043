from __future__ import annotations

"""
Task A dataset utilities.

This module defines:
  - helper functions to build (log1p) regression targets and masks
  - time-feature construction utilities
  - TaskADataset: a multi-mode Dataset that can return either:
      * time-only inputs (point/sequence),
      * time+phase sequences,
      * image sequences + time sequences,
      * cached feature sequences + time sequences.

Design notes
------------
- All regression targets are represented in log-space via log1p(y).
  At evaluation time, predictions/labels are converted back with expm1().
- The mask `M` (shape (T,15)) is used to ignore invalid boundary targets
  (e.g., phases already ended) when computing losses/metrics.
- Sequence windows are left-padded by repeating index 0 when t < seq_len-1.
"""
from pathlib import Path
from typing import Tuple, List, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision.io import read_image
import torchvision.transforms.functional as TF

Mode = Literal["time_point", "time_seq", "timephase_seq", "image_seq", "feat_seq"]


def _log1p_safe(y: np.ndarray) -> np.ndarray:
    """
    Apply a safe log1p transform to regression targets.

    Parameters
    ----------
    y:
        Non-negative targets in seconds. Shape arbitrary.

    Returns
    -------
    y_log:
        log1p(max(y, 0)) cast to float32.

    Notes
    -----
    This guards against any small negative values introduced by preprocessing.
    """
    return np.log1p(np.maximum(y, 0.0)).astype(np.float32)


def _build_targets(d: np.lib.npyio.NpzFile, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct per-frame targets and masks from an opened NPZ.

    Expected keys in NPZ
    --------------------
    - rt_current   : (T,) or (T,1) remaining time of current phase
    - bounds_rem   : (T,7,2) remaining time until start/end of each phase
    - mask_bounds  : (T,7,2) mask for bounds_rem (0/1)

    Parameters
    ----------
    d:
        Opened np.load(...) object.
    T:
        Number of 1fps frames to use.

    Returns
    -------
    Y_log:
        Targets in log-space, shape (T, 15).
        Layout: [rt_current, bounds_rem flattened to 14 dims]
    M:
        Target mask, shape (T, 15).
        Layout: [1 for rt_current, mask_bounds flattened to 14 dims]
    """
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
    use_tnorm: bool = True,
    tnorm_div: float = 6000.0,
) -> np.ndarray:
    """
    Build time-progress features for each frame.

    Parameters
    ----------
    T:
        Number of frames (1fps) in the video.
    use_u:
        Include normalized progress u = t/(T-1).
    use_u2:
        Include u^2
    use_tnorm:
        Include normalized absolute time: clip(t/tnorm_div, 0, 2).
    tnorm_div:
        Divisor for t_norm scaling (seconds).

    Returns
    -------
    X_time:
        Time feature matrix of shape (T, Ft), float32.

    Notes
    -----
    - This function assumes 1fps indexing: t is in seconds.
    - `feat_dim` is unused (kept for backward compatibility); feature dimensionality
      is determined by which flags are enabled.
    """
    # Frame index at 1fps
    t = np.arange(T, dtype=np.float32)
    cols = []

    if use_u:
        u = t / max(T - 1, 1)  # normalized progress in [0,1]
        cols.append(u)
        if use_u2:
            cols.append(u * u)

    if use_tnorm:
        t_norm = np.clip(t / float(tnorm_div), 0.0, 2.0)
        cols.append(t_norm)

    X = np.stack(cols, axis=1).astype(np.float32)  # (T,Ft)
    return X


class TaskADataset(Dataset):
    """
    Multi-mode Dataset for Task A (remaining time regression).

    Each NPZ represents a video resampled to 1fps. This dataset can return
    point-wise samples or sequence windows depending on mode.

    Parameters
    ----------
    npz_path:
        Path to {vid}_labels.npz containing targets, masks, and optional phase info.
    mode:
        One of:
          - "time_point"    : time features at time t -> (x, y_log, m)
          - "time_seq"      : time feature sequence window -> (x_seq, y_log, m)
          - "timephase_seq" : time+phase sequence window -> (x_seq, y_log, m)
          - "image_seq"     : image window + time window -> (img_seq, time_seq, y_log, m)
          - "feat_seq"      : cached feature window + time window -> (feat_seq, time_seq, y_log, m)
    seq_len:
        Sequence window length L for sequence modes.

    Time feature configuration
    --------------------------
    time_feat:
        Chooses which time features to include. If None, defaults depend on mode:
          - time_point          : "u"
          - time_seq/timephase  : "u_u2"
          - image_seq           : "u_tnorm"
          - feat_seq            : "tnorm"
    tnorm_div:
        Divisor for t_norm feature (seconds).

    Phase feature configuration (timephase_seq only)
    ------------------------------------------------
    use_phase_onehot:
        If True, append one-hot phase id (size n_phases) to time features.
    use_elapsed_phase:
        If True, append normalized elapsed_phase scalar feature.
    n_phases:
        Number of surgical phases (default 7 for Cholec80).

    Image sequence configuration (image_seq only)
    ---------------------------------------------
    frames_root:
        Root directory containing per-video frame folders.
        Expects frames at: frames_root/{video}/*.png
    img_size:
        Spatial size after resize (square).
    stride:
        Sampling stride for indices when mode="image_seq".
        (Still uses seq_len windows; stride controls which t positions are used.)

    Cached feature configuration (feat_seq only)
    --------------------------------------------
    feat_path:
        Path to a .npy feature array of shape (T, F) (e.g. (T,512)) aligned to 1fps.

    Returns (by mode)
    -----------------
    - time_point:
        x      : (Ft,)
        y_log  : (15,)
        m      : (15,)
    - time_seq:
        x_seq  : (L, Ft)
        y_log  : (15,)
        m      : (15,)
    - timephase_seq:
        x_seq  : (L, F)  where F = Ft + (n_phases if onehot) + (1 if elapsed_phase)
        y_log  : (15,)
        m      : (15,)
    - image_seq:
        img_seq  : (L, 3, H, W) with H=W=img_size, normalized like ImageNet
        time_seq : (L, Ft)
        y_log    : (15,)
        m        : (15,)
    - feat_seq:
        feat_seq : (L, Ffeat) typically (L,512)
        time_seq : (L, Ft)
        y_log    : (15,)
        m        : (15,)

    Sequence window policy
    ----------------------
    For sequence modes, at time t we take indices [t-L+1, ..., t].
    If t < L-1, we left-pad by repeating index 0.
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

        # Optional metadata: used only for image_seq path resolution
        self.video = str(d["video"]) if "video" in d.files else None

        # Number of frames at 1fps
        T = int(d["T_1fps"][0])
        self.T = T

        # -------------------------
        # Targets + mask
        # -------------------------
        # NOTE: you have two implementations here:
        #  - inline version (np.log1p(Y)) for some modes
        #  - _build_targets() for others (with safe clamp)
        #
        # Keeping your original behavior, but documenting it clearly.
        if mode in ("time_point", "time_seq", "timephase_seq", "feat_seq"):
            rt = d["rt_current"].astype(np.float32).reshape(-1, 1)[:T]
            bounds = d["bounds_rem"].astype(np.float32).reshape(-1, 7, 2)[:T]
            mask_b = d["mask_bounds"].astype(np.float32).reshape(-1, 7, 2)[:T]

            Y = np.concatenate([rt, bounds.reshape(T, 14)], axis=1)  # (T,15)
            M = np.concatenate([np.ones((T, 1), np.float32), mask_b.reshape(T, 14)], axis=1)  # (T,15)

            # Targets stored in log1p space
            self.Y_log = np.log1p(Y).astype(np.float32)
            self.M = M.astype(np.float32)
        else:
            self.Y_log, self.M = _build_targets(d, T)  # (T,15), (T,15)

        # -------------------------
        # Base time features (X_time)
        # -------------------------
        if time_feat is None:
            if mode == "time_point":
                time_feat = "u"
            elif mode in ("time_seq", "timephase_seq"):
                time_feat = "u_u2"
            elif mode == "image_seq":
                time_feat = "u_tnorm"
            else:  # feat_seq
                time_feat = "tnorm"

        # Map time_feat string -> actual feature construction
        if time_feat == "u":
            X_time = _time_feats(T, use_u=True, use_u2=False, use_tnorm=False)
        elif time_feat == "u_u2":
            X_time = _time_feats(T, use_u=True, use_u2=True, use_tnorm=False)
        elif time_feat == "tnorm":
            X_time = _time_feats(T, use_u=False, use_u2=False, use_tnorm=True, tnorm_div=tnorm_div)
        elif time_feat == "u_tnorm":
            X_time = _time_feats(T, use_u=True, use_u2=False, use_tnorm=True, tnorm_div=tnorm_div)
        else:  # "u_u2_tnorm"
            X_time = _time_feats(T, use_u=True, use_u2=True, use_tnorm=True, tnorm_div=tnorm_div)

        # -------------------------
        # Add phase features (timephase_seq only)
        # -------------------------
        if mode == "timephase_seq":
            X = X_time

            # Phase one-hot: (T, n_phases)
            if use_phase_onehot:
                phase_id = d["phase_ids_1fps"].astype(np.int64)[:T]
                phase_id = np.clip(phase_id, 0, n_phases - 1)
                phase_oh = np.eye(n_phases, dtype=np.float32)[phase_id]
                X = np.concatenate([X, phase_oh], axis=1)

            # Elapsed phase scalar (normalized): (T,1)
            if use_elapsed_phase:
                ep = d["elapsed_phase"].astype(np.float32)[:T].reshape(-1, 1)
                ep = np.clip(ep / 600.0, 0.0, 10.0)
                X = np.concatenate([X, ep], axis=1)

            self.X = X.astype(np.float32)
        else:
            self.X_time = X_time.astype(np.float32)

        # -------------------------
        # Image sequence setup (image_seq only)
        # -------------------------
        if mode == "image_seq":
            if frames_root is None or self.video is None:
                raise ValueError("image_seq requires frames_root and npz must contain 'video'")

            self.frames_dir = Path(frames_root) / self.video
            self.frame_paths: List[Path] = sorted(self.frames_dir.glob("*.png"))

            # If frames on disk do not match T, shrink to the minimum to stay consistent
            if len(self.frame_paths) != T:
                self.T = min(self.T, len(self.frame_paths))
                self.frame_paths = self.frame_paths[: self.T]
                self.Y_log = self.Y_log[: self.T]
                self.M = self.M[: self.T]
                if mode == "timephase_seq":
                    self.X = self.X[: self.T]
                else:
                    self.X_time = self.X_time[: self.T]

        # -------------------------
        # Cached features setup (feat_seq only)
        # -------------------------
        if mode == "feat_seq":
            if feat_path is None:
                raise ValueError("feat_seq requires feat_path")

            F = np.load(feat_path).astype(np.float32)  # expected (T, Ffeat), e.g., (T,512)

            # If cached feats length differs, shrink to the minimum aligned prefix
            if F.shape[0] != T:
                self.T = min(self.T, F.shape[0])
                F = F[: self.T]
                self.Y_log = self.Y_log[: self.T]
                self.M = self.M[: self.T]
                self.X_time = self.X_time[: self.T]

            self.F = F

        # -------------------------
        # Sampling indices
        # -------------------------
        # Defines which time steps t are valid items in the dataset.
        if mode in ("time_point", "time_seq", "timephase_seq"):
            # Use all frames as individual samples
            self.indices = np.arange(self.T, dtype=np.int64)
        elif mode == "image_seq":
            # Optionally subsample target times by stride (still returns full seq_len window each time)
            self.indices = np.arange(0, self.T, self.stride, dtype=np.int64)
        else:  # feat_seq
            # Typically start at seq_len-1 so first sample has a full (unpadded) window
            self.indices = np.arange(self.seq_len - 1, self.T, dtype=np.int64)

    def __len__(self) -> int:
        """Return number of samples (depends on mode and indexing strategy)."""
        return len(self.indices)

    def _get_window_ids(self, t: int) -> np.ndarray:
        """
        Get a fixed-length window of indices ending at time t.

        Parameters
        ----------
        t:
            End time index (inclusive) in [0, T-1].

        Returns
        -------
        ids:
            Array of shape (L,), where L=seq_len.
            If t < L-1, the window is left-padded with zeros (index 0 repeated).
        """
        L = self.seq_len
        start = t - L + 1
        if start >= 0:
            return np.arange(start, t + 1, dtype=np.int64)

        # Left-pad with 0 for early timesteps
        pad_n = -start
        ids = np.concatenate([np.zeros(pad_n, dtype=np.int64), np.arange(0, t + 1, dtype=np.int64)])
        return ids[-L:]

    def _load_img(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess one RGB frame.

        Returns
        -------
        img:
            Float tensor of shape (3, img_size, img_size), ImageNet-normalized.

        Notes
        -----
        - Handles RGBA by dropping alpha; grayscale by repeating channels.
        - Uses torchvision.io.read_image (returns CHW uint8).
        """
        img = read_image(str(self.frame_paths[idx])).float() / 255.0  # (C,H,W) in [0,1]

        # Handle different channel counts robustly
        if img.shape[0] == 4:
            img = img[:3]
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        # Resize to fixed spatial size
        img = TF.resize(img, [self.img_size, self.img_size], antialias=True)

        # ImageNet normalization (common for ResNet features)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img - mean) / std

    def __getitem__(self, idx: int):
        """
        Fetch one training/evaluation sample.

        The returned tuple depends on `self.mode`. See class docstring for details.

        Parameters
        ----------
        idx:
            Dataset index in [0, len(self)-1]. Internally maps to a time index `t`.

        Returns
        -------
        Mode-dependent tuple:
          - time_point:    (x, y_log, m)
          - time_seq:      (x_seq, y_log, m)
          - timephase_seq: (x_seq, y_log, m)
          - image_seq:     (img_seq, time_seq, y_log, m)
          - feat_seq:      (feat_seq, time_seq, y_log, m)
        """
        t = int(self.indices[idx])

        if self.mode == "time_point":
            # x: (Ft,)
            x = torch.from_numpy(self.X_time[t]).float()
            y_log = torch.from_numpy(self.Y_log[t]).float()  # (15,)
            m = torch.from_numpy(self.M[t]).float()          # (15,)
            return x, y_log, m

        if self.mode == "time_seq":
            # x_seq: (L, Ft)
            ids = self._get_window_ids(t)
            x_seq = torch.from_numpy(self.X_time[ids]).float()
            y_log = torch.from_numpy(self.Y_log[t]).float()
            m = torch.from_numpy(self.M[t]).float()
            return x_seq, y_log, m

        if self.mode == "timephase_seq":
            # x_seq: (L, F) where F includes phase features
            ids = self._get_window_ids(t)
            x_seq = torch.from_numpy(self.X[ids]).float()
            y_log = torch.from_numpy(self.Y_log[t]).float()
            m = torch.from_numpy(self.M[t]).float()
            return x_seq, y_log, m

        if self.mode == "image_seq":
            # img_seq: (L, 3, H, W), time_seq: (L, Ft)
            ids = self._get_window_ids(t)
            img_seq = torch.stack([self._load_img(int(i)) for i in ids], dim=0)
            time_seq = torch.from_numpy(self.X_time[ids]).float()
            y_log = torch.from_numpy(self.Y_log[t]).float()
            m = torch.from_numpy(self.M[t]).float()
            return img_seq, time_seq, y_log, m

        if self.mode == "feat_seq":
            # feat_seq: (L, Ffeat), time_seq: (L, Ft)
            ids = self._get_window_ids(t)
            feat_seq = torch.from_numpy(self.F[ids]).float()
            time_seq = torch.from_numpy(self.X_time[ids]).float()
            y_log = torch.from_numpy(self.Y_log[t]).float()
            m = torch.from_numpy(self.M[t]).float()
            return feat_seq, time_seq, y_log, m

        raise RuntimeError(f"Unknown mode: {self.mode}")
