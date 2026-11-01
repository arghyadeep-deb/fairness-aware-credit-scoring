"""Joint objective L = L_wCE + lambda_f L_fair + lambda_c L_cal + lambda_a L_adv (specs/09).

Schedule: `warmup_epochs` of weighted CE only (all lambdas 0), then the lambdas ramp linearly
to their targets over `ramp_epochs`. A term whose effective lambda is 0 is not even computed,
so with all lambdas 0 the value AND the gradients equal the A2 loss exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.calibration import soft_group_ece
from src.losses.fairness import fairness_loss
from src.losses.weighted_ce import weighted_bce


@dataclass
class JointConfig:
    lambda_f: float = 0.0
    lambda_c: float = 0.0
    lambda_a: float = 0.0
    warmup_epochs: float = 3
    ramp_epochs: float = 5
    fairness_type: str = "eo"
    flag_rate: float = 0.10
    temperature: float = 0.1
    cal_bins: int = 10
    cal_sigma: float = 0.05

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **lambdas: float) -> "JointConfig":
        j = cfg["joint"]
        return cls(
            warmup_epochs=j["warmup_epochs"],
            ramp_epochs=j["ramp_epochs"],
            fairness_type=j["fairness"]["type"],
            flag_rate=cfg["evaluation"]["flag_rate"],
            temperature=j["fairness"]["temperature"],
            cal_bins=j["calibration"]["bins"],
            cal_sigma=j["calibration"]["sigma"],
            **lambdas,
        )


class JointLoss(nn.Module):
    def __init__(self, cfg: JointConfig, w_neg: float, w_pos: float) -> None:
        super().__init__()
        self.cfg, self.w_neg, self.w_pos = cfg, w_neg, w_pos
        self.epoch = 0.0

    def set_epoch(self, epoch: float) -> None:
        """Epoch may be fractional (epoch + step / steps_per_epoch) for a smooth ramp."""
        self.epoch = float(epoch)

    def ramp(self) -> float:
        """0 during warm-up, then linear to 1 over ramp_epochs."""
        c = self.cfg
        if self.epoch < c.warmup_epochs:
            return 0.0
        if c.ramp_epochs <= 0:
            return 1.0
        return float(min(1.0, (self.epoch - c.warmup_epochs) / c.ramp_epochs))

    def lambdas(self) -> dict[str, float]:
        r = self.ramp()
        return {"f": self.cfg.lambda_f * r, "c": self.cfg.lambda_c * r, "a": self.cfg.lambda_a * r}

    def forward(
        self,
        logit: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        adv_logit: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """logit/y/a/adv_logit are (B,). Returns (total, parts) with parts = unweighted terms."""
        lam, c = self.lambdas(), self.cfg
        wce = weighted_bce(logit, y, w_neg=self.w_neg, w_pos=self.w_pos)
        total, parts = wce, {"wce": wce.detach()}
        if lam["f"] > 0:
            lf = fairness_loss(logit, y, a, c.fairness_type, c.flag_rate, c.temperature)
            total, parts["fair"] = total + lam["f"] * lf, lf.detach()
        if lam["c"] > 0:
            lc = soft_group_ece(logit, y, a, c.cal_bins, c.cal_sigma)
            total, parts["cal"] = total + lam["c"] * lc, lc.detach()
        if lam["a"] > 0:
            if adv_logit is None:
                raise ValueError("lambda_a > 0 needs the adversary logit (use JointModel)")
            la = F.binary_cross_entropy_with_logits(adv_logit.float(), a.float())
            total, parts["adv"] = total + lam["a"] * la, la.detach()
        return total, parts
