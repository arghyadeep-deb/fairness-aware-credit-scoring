"""Focal loss (A5): alpha from the class prior, gamma tuned in [1, 3]."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor, y: torch.Tensor, gamma: float, alpha_pos: float
) -> torch.Tensor:
    """mean_i -alpha_t (1 - p_t)^gamma log p_t; logits (B,), y (B,).

    alpha_pos weights positives and (1 - alpha_pos) negatives. Taking log p_t from the
    logit-space BCE avoids log(0).
    """
    yf = y.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), yf, reduction="none")  # = -log p_t
    p_t = torch.exp(-bce)
    alpha_t = torch.where(
        yf > 0.5, torch.full_like(bce, alpha_pos), torch.full_like(bce, 1 - alpha_pos)
    )
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()
