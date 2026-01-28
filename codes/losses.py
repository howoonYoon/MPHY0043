from __future__ import annotations
import torch


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom

import torch


def weighted_masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, w_bounds: float = 0.2, beta: float = 1.0) -> torch.Tensor:
    """
    pred/target/mask: (B,15) in log-space targets.
    Loss = SmoothL1(rt) + w_bounds * SmoothL1(bounds) with masking.
    """
    diff = torch.abs(pred - target)
    elem = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)

    # rt dim 0
    rt = (elem[:, :1] * mask[:, :1]).sum() / mask[:, :1].sum().clamp_min(1.0)
    # bounds dims 1:15
    b = (elem[:, 1:] * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp_min(1.0)
    return rt + w_bounds * b
