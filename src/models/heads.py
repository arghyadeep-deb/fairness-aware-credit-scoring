"""Adversary head (Head B) and the joint-model wrapper. Head B exists at training time only."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.grl import GradientReversal


class AdversaryHead(nn.Module):
    """z -> GRL -> Linear(z, 32) -> ReLU -> Linear(32, 1): predicts gender from z.

    The GRL uses coefficient 1 (pure reversal). The strength lambda_a is applied to the loss
    term in JointLoss, so with lambda_a = 0 this head cannot influence the trunk at all.
    """

    def __init__(self, z_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.grl = GradientReversal(1.0)
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B, z_dim) -> (B,)
        return self.net(self.grl(z)).squeeze(-1)

    def frozen(self, z: torch.Tensor) -> torch.Tensor:
        """Head output with detached parameters: gradients reach z only (confusion loss)."""
        l1, l2 = self.net[0], self.net[2]
        h = F.relu(F.linear(z, l1.weight.detach(), l1.bias.detach()))
        return F.linear(h, l2.weight.detach(), l2.bias.detach()).squeeze(-1)


class JointModel(nn.Module):
    """Backbone (Head A: default probability) plus adversary Head B on the same z."""

    def __init__(self, backbone: nn.Module, adv_hidden: int = 32, mode: str = "grl") -> None:
        super().__init__()
        self.backbone = backbone
        self.adversary = AdversaryHead(backbone.z_dim, adv_hidden)
        self.mode = mode  # "grl" (spec) or "confusion" (bounded deviation, Phase 4 rescue c)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
        """Third output: GRL logit (grl) or (head logit on detached z, frozen-head logit on z)."""
        logit, z = self.backbone(x_cont, x_cat)  # (B,), (B, z_dim)
        if self.mode == "confusion":
            return logit, z, (self.adversary.net(z.detach()).squeeze(-1), self.adversary.frozen(z))
        return logit, z, self.adversary(z)

    def predict(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Served path: Head A only; Head B is never evaluated at inference."""
        return self.backbone(x_cont, x_cat)[0]
