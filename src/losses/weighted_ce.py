"""Plain and class-weighted binary cross-entropy (A0 uses unit weights, A2 uses train weights)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce(
    logits: torch.Tensor, y: torch.Tensor, w_neg: float = 1.0, w_pos: float = 1.0
) -> torch.Tensor:
    """mean_i w_{y_i} * BCE(logit_i, y_i); logits (B,), y (B,) in {0,1}.

    Computed from logits (not probabilities) for numerical stability; no clipping needed.
    """
    per = F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="none")
    w = torch.where(y > 0.5, torch.full_like(per, w_pos), torch.full_like(per, w_neg))
    return (w * per).mean()
