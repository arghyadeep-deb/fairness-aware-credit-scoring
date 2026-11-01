"""Gradient reversal layer: identity forward, gradient multiplied by -lambda backward."""

from __future__ import annotations

import torch
from torch import nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # (B, D) -> (B, D)
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return -ctx.lambd * grad_out, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with coefficient `lambd`."""
    return _GradReverse.apply(x, lambd)


class GradientReversal(nn.Module):
    """Module form; `lambd` can be changed between steps (e.g. by a ramp schedule)."""

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambd)
