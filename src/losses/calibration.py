"""Differentiable soft-binned ECE, computed per gender group (used by G1; reused in Phase 3)."""

from __future__ import annotations

import torch


def soft_group_ece(
    logits: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    bins: int = 10,
    sigma: float = 0.05,
    clip: float = 1e-6,
) -> torch.Tensor:
    """Mean over groups of soft-binned ECE. logits/y/a are (B,); a in {0,1}.

    Each sample is spread over bin centres with a Gaussian kernel, so the loss is smooth in
    the probabilities. Groups absent from the batch are skipped (returns 0 if none present).
    """
    p = torch.sigmoid(logits.float()).clamp(clip, 1 - clip)
    centres = torch.linspace(0.5 / bins, 1 - 0.5 / bins, bins, device=p.device)
    k = torch.exp(-((p[:, None] - centres[None, :]) ** 2) / (2 * sigma**2))  # (B, bins)
    k = k / k.sum(dim=1, keepdim=True).clamp_min(1e-12)
    yf = y.float()
    eces = []
    for g in (0, 1):
        m = a == g
        n = int(m.sum())
        if n == 0:
            continue
        kg, pg, yg = k[m], p[m], yf[m]
        mass = kg.sum(0)  # (bins,)
        avg_p = (kg * pg[:, None]).sum(0) / mass.clamp_min(1e-12)
        avg_y = (kg * yg[:, None]).sum(0) / mass.clamp_min(1e-12)
        eces.append((mass / n * (avg_p - avg_y).abs()).sum())
    return torch.stack(eces).mean() if eces else p.new_zeros(())
