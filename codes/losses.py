from __future__ import annotations

import torch


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute masked Smooth L1 loss averaged over valid targets.

    Parameters
    ----------
    pred:
        Predicted values, shape (B, D).
    target:
        Ground-truth values, shape (B, D).
        Typically in log-space for Task A.
    mask:
        Binary or continuous mask indicating valid targets, shape (B, D).
        Elements with mask=0 do not contribute to the loss.
    beta:
        Smooth L1 transition point (same definition as in PyTorch's SmoothL1Loss).

    Returns
    -------
    loss:
        Scalar tensor: mean masked Smooth L1 loss over all valid elements.

    Notes
    -----
    - Loss is computed element-wise, then multiplied by `mask`.
    - The final loss is normalized by the sum of mask values to avoid
      dependence on the number of valid targets.
    - `clamp_min(1.0)` prevents division by zero when all mask values are zero.
    """
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def weighted_masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    w_bounds: float = 0.2,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute weighted masked Smooth L1 loss for Task A targets.

    The target vector is assumed to have the following structure:
      - index 0   : remaining time of the current phase (rt_current)
      - indices 1: remaining-time bounds for all phases (flattened)

    Loss formulation
    ----------------
    loss = SmoothL1(rt_current) + w_bounds * SmoothL1(bounds)

    Both terms are:
      - masked element-wise,
      - normalized by the number of valid elements in their respective groups.

    Parameters
    ----------
    pred:
        Predicted values, shape (B, 15), in log-space.
    target:
        Ground-truth values, shape (B, 15), in log-space.
    mask:
        Mask tensor, shape (B, 15).
        mask[:, 0] corresponds to rt_current (typically all ones).
        mask[:, 1:] corresponds to phase-boundary targets.
    w_bounds:
        Weight applied to the boundary-loss term.
        Higher values emphasize accuracy on phase-boundary predictions.
    beta:
        Smooth L1 transition point.

    Returns
    -------
    loss:
        Scalar tensor: weighted masked Smooth L1 loss.

    Notes
    -----
    - This loss decouples the scale of rt_current and boundary errors.
    - Normalizing rt and bounds separately prevents domination by the
      larger group of boundary targets.
    - If all boundary masks are zero, the bounds term safely contributes zero.
    """
    diff = torch.abs(pred - target)
    elem = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)

    # rt_current (dimension 0)
    rt = (elem[:, :1] * mask[:, :1]).sum() / mask[:, :1].sum().clamp_min(1.0)

    # phase-boundary targets (dimensions 1:)
    b = (elem[:, 1:] * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp_min(1.0)

    return rt + w_bounds * b
