#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Constants / mappings
# -------------------------
PHASE2ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderRetraction": 4,
    "CleaningCoagulation": 5,
    "GallbladderPackaging": 6,
}
ID2PHASE = {v: k for k, v in PHASE2ID.items()}

TOOL_COLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
NUM_PHASES = 7
FPS_RATIO = 25  # original video 25fps -> released frames 1fps (one frame per second)


def runs_from_labels(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Convert label sequence into runs.
    Returns list of (phase_id, start_idx, end_idx) inclusive.
    """
    labels = np.asarray(labels).astype(int)
    T = len(labels)
    runs = []
    s = 0
    for t in range(1, T):
        if labels[t] != labels[t - 1]:
            runs.append((int(labels[t - 1]), s, t - 1))
            s = t
    runs.append((int(labels[T - 1]), s, T - 1))
    return runs


# -------------------------
# Utilities
# -------------------------
def one_hot(x: int, K: int) -> np.ndarray:
    v = np.zeros((K,), dtype=np.float32)
    if 0 <= x < K:
        v[x] = 1.0
    return v


def count_png_frames(frames_dir: Path) -> int:
    return len(list(frames_dir.glob("*.png")))


def load_phase_25fps(phase_txt: Path) -> np.ndarray:
    """
    Loads per-frame phase at original fps indexing (typically 25fps).
    Returns phase_ids_25 of shape (F_25,), where index corresponds to original frame.
    """
    df = pd.read_csv(phase_txt, sep="\t").sort_values("Frame").reset_index(drop=True)

    # Frame column should be 0..Fmax contiguous in typical Cholec80 exports
    frames = df["Frame"].to_numpy()
    if frames[0] != 0 or not np.all(frames == np.arange(len(frames))):
        # If not contiguous, we still can rebuild full array by reindexing
        Fmax = int(frames.max())
        phase_ids_25 = np.zeros((Fmax + 1,), dtype=np.int64)
        phase_ids_25[:] = -1
        mapped = df["Phase"].map(PHASE2ID).to_numpy(dtype=np.int64)
        phase_ids_25[frames.astype(int)] = mapped
        # forward-fill any gaps (rare)
        last = phase_ids_25[0] if phase_ids_25[0] != -1 else 0
        for i in range(len(phase_ids_25)):
            if phase_ids_25[i] == -1:
                phase_ids_25[i] = last
            else:
                last = phase_ids_25[i]
        return phase_ids_25

    phase_ids_25 = df["Phase"].map(PHASE2ID).to_numpy(dtype=np.int64)
    return phase_ids_25


def load_tools_sparse_25fps(tool_txt: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    tool file often contains rows at frames 0,25,50,... with binary columns.
    Returns:
      frames_sparse: (M,) int
      tools_sparse:  (M,7) float32
    """
    df = pd.read_csv(tool_txt, sep="\t").sort_values("Frame").reset_index(drop=True)
    frames_sparse = df["Frame"].to_numpy(dtype=np.int64)
    tools_sparse = df[TOOL_COLS].to_numpy(dtype=np.float32)
    return frames_sparse, tools_sparse


def expand_sparse_to_dense(frames_sparse: np.ndarray, values_sparse: np.ndarray, Fmax: int) -> np.ndarray:
    """
    Forward-fill sparse annotations to length (Fmax+1).
    frames_sparse: (M,), increasing
    values_sparse: (M,D)
    Returns dense: (Fmax+1, D)
    """
    D = values_sparse.shape[1]
    dense = np.zeros((Fmax + 1, D), dtype=np.float32)
    last = np.zeros((D,), dtype=np.float32)

    j = 0
    for f in range(Fmax + 1):
        if j < len(frames_sparse) and int(frames_sparse[j]) == f:
            last = values_sparse[j]
            j += 1
        dense[f] = last
    return dense


def downsample_25fps_to_1fps(arr_25: np.ndarray, T_1: int, fps_ratio: int = FPS_RATIO, center: bool = False) -> np.ndarray:
    """
    Downsample by picking one sample per second.
    If center=True, picks 25*t + 12 (mid-frame) instead of 25*t.
    Works for 1D (F,) or 2D (F,D).
    """
    if center:
        idx = (np.arange(T_1) * fps_ratio + (fps_ratio // 2)).astype(np.int64)
    else:
        idx = (np.arange(T_1) * fps_ratio).astype(np.int64)
    idx = np.clip(idx, 0, len(arr_25) - 1)
    return arr_25[idx]


# -------------------------
# Task A label generation
# -------------------------
@dataclass(frozen=True)
class PhaseSegment:
    phase: int
    start: int  # inclusive 1fps index
    end: int    # inclusive 1fps index


def extract_phase_segments_1fps(phase_ids_1: np.ndarray, num_phases: int = NUM_PHASES) -> Dict[int, PhaseSegment]:
    phase_ids_1 = np.asarray(phase_ids_1).astype(int)
    segs: Dict[int, PhaseSegment] = {}
    for p in range(num_phases):
        idx = np.where(phase_ids_1 == p)[0]
        if idx.size == 0:
            continue
        segs[p] = PhaseSegment(phase=p, start=int(idx.min()), end=int(idx.max()))
    return segs

def build_taskA_for_video_1fps(
    phase_ids_1: np.ndarray,
    num_phases: int = NUM_PHASES,
    clamp_nonnegative: bool = True,
) -> Dict[str, np.ndarray]:
    """
    All times are in seconds because 1fps index == seconds index.
    Outputs:
      X: (T, 9) = phase onehot(7) + elapsed_surgery + elapsed_phase
      # NOTE:
        # X is stored for optional analysis / ablation.
        # In the main Task A experiments, phase one-hot is NOT used as model input.

      rt_current: (T,1)
      bounds_rem: (T,7,2) where [:,p,0]=start_p - t, [:,p,1]=end_p - t (future run)
      mask_bounds: (T,7,2) valid mask
      elapsed_phase: (T,1)
    """
    phase_ids_1 = np.asarray(phase_ids_1).astype(int)
    T = len(phase_ids_1)
    t_sec = np.arange(T, dtype=np.float32)

    runs = runs_from_labels(phase_ids_1)

    # per-time current run start/end
    cur_start = np.zeros((T,), dtype=np.int64)
    cur_end = np.zeros((T,), dtype=np.int64)
    for p, s, e in runs:
        cur_start[s:e+1] = s
        cur_end[s:e+1] = e

    # inputs
    X = np.zeros((T, num_phases + 2), dtype=np.float32)
    X[:, num_phases:num_phases+1] = t_sec.reshape(-1, 1)  # elapsed_surgery

    # targets
    rt_current = (cur_end.astype(np.float32) - t_sec).reshape(-1, 1)
    elapsed_phase = (t_sec - cur_start.astype(np.float32)).reshape(-1, 1)
    X[:, num_phases+1:num_phases+2] = elapsed_phase

    # phase onehot
    for t in range(T):
        p = int(phase_ids_1[t])
        X[t, :num_phases] = one_hot(p, num_phases)

    # upcoming phase bounds: "the next run of phase p that starts at/after t"
    next_start = np.full((T, num_phases), np.nan, dtype=np.float32)
    next_end = np.full((T, num_phases), np.nan, dtype=np.float32)

    # Fill next occurrence by scanning runs from back to front
    # Maintain latest run (start,end) seen in the future for each phase.
    future_s = np.full((num_phases,), np.nan, dtype=np.float32)
    future_e = np.full((num_phases,), np.nan, dtype=np.float32)

    run_idx = len(runs) - 1
    for t in range(T - 1, -1, -1):
        # update future pointers when passing run boundaries
        while run_idx >= 0 and runs[run_idx][1] == t:
            p, s, e = runs[run_idx]
            future_s[p] = float(s)
            future_e[p] = float(e)
            run_idx -= 1
        next_start[t, :] = future_s
        next_end[t, :] = future_e

    bounds_rem = np.zeros((T, num_phases, 2), dtype=np.float32)
    mask_bounds = np.zeros((T, num_phases, 2), dtype=np.float32)

    # Compute remaining-to-start/end; valid when next_start is not nan AND next_start >= t
    for p in range(num_phases):
        s = next_start[:, p]
        e = next_end[:, p]
        valid = ~np.isnan(s)
        bounds_rem[valid, p, 0] = s[valid] - t_sec[valid]
        bounds_rem[valid, p, 1] = e[valid] - t_sec[valid]
        mask_bounds[valid, p, :] = 1.0

    if clamp_nonnegative:
        rt_current = np.maximum(rt_current, 0.0)
        bounds_rem = np.maximum(bounds_rem, 0.0)

    return {
        "X": X,  # (T,9)
        "phase_ids_1fps": phase_ids_1.astype(np.int64),  # (T,)
        "rt_current": rt_current.astype(np.float32),  # (T,1)
        "bounds_rem": bounds_rem.astype(np.float32),  # (T,7,2)
        "mask_bounds": mask_bounds.astype(np.float32),  # (T,7,2)
        "elapsed_phase": elapsed_phase.astype(np.float32),  # (T,1)
        "cur_start_1fps": cur_start.astype(np.int64),  # optional debug
        "cur_end_1fps": cur_end.astype(np.int64),      # optional debug
    }


# -------------------------
# Main processing
# -------------------------
def process_video(cholec80_dir: Path, video_num: int, out_dir: Path, center_sample: bool = False) -> None:
    """
    video_num: 1..80 corresponds to folder names video01..video80
    """
    vid_name = f"video{video_num:02d}"
    phase_txt = cholec80_dir / "phase_annotations" / f"{vid_name}-phase.txt"
    tool_txt = cholec80_dir / "tool_annotations" / f"{vid_name}-tool.txt"
    frames_dir = cholec80_dir / "frames" / vid_name

    if not phase_txt.exists():
        raise FileNotFoundError(f"Missing phase file: {phase_txt}")
    if not tool_txt.exists():
        raise FileNotFoundError(f"Missing tool file: {tool_txt}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Missing frames dir: {frames_dir}")

    T_1 = count_png_frames(frames_dir)
    if T_1 == 0:
        raise RuntimeError(f"No png frames found in {frames_dir}")

    # Phase: 25fps -> 1fps
    phase_25 = load_phase_25fps(phase_txt)
    phase_1 = downsample_25fps_to_1fps(phase_25, T_1, fps_ratio=FPS_RATIO, center=center_sample)

    # Tools: sparse 25fps rows -> dense -> 1fps
    frames_sparse, tools_sparse = load_tools_sparse_25fps(tool_txt)
    Fmax = int(max(frames_sparse.max(), len(phase_25) - 1))
    tools_25_dense = expand_sparse_to_dense(frames_sparse, tools_sparse, Fmax=Fmax)
    tools_1 = downsample_25fps_to_1fps(tools_25_dense, T_1, fps_ratio=FPS_RATIO, center=center_sample)

    # Task A labels from 1fps phase ids
    taskA = build_taskA_for_video_1fps(phase_1)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{vid_name}_labels.npz",
        video=vid_name,
        T_1fps=np.array([T_1], dtype=np.int64),
        tools_1fps=tools_1.astype(np.float32),  # (T,7)
        **taskA,
    )

    # Lightweight metadata json for debugging
    meta = {
        "video": vid_name,
        "T_1fps": T_1,
        "phase_25_len": int(len(phase_25)),
        "tools_sparse_rows": int(len(frames_sparse)),
        "tools_sparse_last_frame": int(frames_sparse.max()),
        "downsample_center": bool(center_sample),
    }
    (out_dir / f"{vid_name}_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cholec80_dir", type=str, required=True,
                    help="Path to cholec80 directory (contains frames/, phase_annotations/, tool_annotations/)")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write *.npz label files")
    ap.add_argument("--video_start", type=int, default=1, help="Start video number (1..80)")
    ap.add_argument("--video_end", type=int, default=80, help="End video number (1..80)")
    ap.add_argument("--center_sample", action="store_true",
                    help="Use mid-frame (25*t+12) instead of 25*t for downsampling")
    args = ap.parse_args()

    cholec80_dir = Path(args.cholec80_dir)
    out_dir = Path(args.out_dir)

    for v in range(args.video_start, args.video_end + 1):
        print(f"[{v:02d}] processing...")
        process_video(cholec80_dir, v, out_dir, center_sample=args.center_sample)

    # Save a simple default split (60/10/10) based on video number order
    split = {
        "train": [f"video{v:02d}" for v in range(1, 61)],
        "val":   [f"video{v:02d}" for v in range(61, 71)],
        "test":  [f"video{v:02d}" for v in range(71, 81)],
    }
    (out_dir / "split_default_60_10_10.json").write_text(json.dumps(split, indent=2))
    print("Done. Wrote split_default_60_10_10.json")



if __name__ == "__main__":
    main()
