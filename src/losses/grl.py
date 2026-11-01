"""Gradient reversal layer: identity forward, gradient multiplied by -lambda backward."""

from __future__ import annotations

import torch
from torch import nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float, clip: float, weight: float, stats):  # (B, D) -> (B, D)
        ctx.lambd, ctx.clip, ctx.weight, ctx.stats = float(lambd), float(clip), float(weight), stats
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        g = -ctx.lambd * grad_out
        if ctx.stats is not None or ctx.clip > 0:
            # per-row norm of the UNWEIGHTED reversed gradient (loss weight and 1/B divided out)
            rows = grad_out.float() * (grad_out.shape[0] / max(ctx.weight, 1e-12))
            n = rows.norm(dim=1)
            if ctx.stats is not None:
                ctx.stats.append(float(n.median()))
            if ctx.clip > 0:
                g = g * (ctx.clip / (n + 1e-12)).clamp(max=1.0).unsqueeze(1).to(g.dtype)
        return g, None, None, None, None


def grad_reverse(
    x: torch.Tensor,
    lambd: float = 1.0,
    clip: float = 0.0,
    weight: float = 1.0,
    stats: list | None = None,
) -> torch.Tensor:
    """Gradient reversal with coefficient `lambd`.

    Optional (rescue variant b): `clip` > 0 clips the per-row L2 norm of the unweighted reversed
    gradient to `clip` (`weight` = the loss weight that multiplied the term, divided out here);
    `stats` collects the per-batch median row norm. Defaults reproduce the plain GRL exactly.
    """
    return _GradReverse.apply(x, lambd, clip, weight, stats)


class GradientReversal(nn.Module):
    """Module form; `lambd` can be changed between steps (e.g. by a ramp schedule)."""

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = lambd
        self.clip = 0.0  # rescue variant b (off)
        self.weight = 1.0  # loss weight of the adversary term, set by the trainer each step
        self.stats: list | None = None  # per-batch median row norm of the reversed gradient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambd, self.clip, self.weight, self.stats)
