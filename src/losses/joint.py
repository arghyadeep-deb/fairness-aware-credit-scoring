"""Joint objective L = L_wCE + lambda_f L_fair + lambda_c L_cal + lambda_a L_adv (specs/09).

Schedule: `warmup_epochs` of weighted CE only (all lambdas 0), then the lambdas ramp linearly
to their targets over `ramp_epochs`. A term whose effective lambda is 0 is not even computed,
so with all lambdas 0 the value AND the gradients equal the A2 loss exactly.

Normalisation (`normalize="initial"`, Phase 4): each auxiliary term is divided by its own
initial magnitude, the mean of the (detached) term over the last warm-up epoch, frozen when the
ramp starts. The lambdas then weigh terms that all start near 1, so one grid is comparable
across terms. During warm-up the terms are only observed under no_grad: the loss value and
gradients stay identical to A2 there. The reference is fixed afterwards (a running average would
let a term that is being minimised blow up its own gradient scale).
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
    normalize: str = "none"  # "none" | "initial"; configs/train.yaml sets "initial" (Phase 4)
    ref_floor: float = 1e-3  # a reference below this is clamped (avoids dividing by ~0)
    adv_pretrain: bool = False  # DIAGNOSTIC (off): fit Head B on detached z during warm-up

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
            normalize=j.get("normalize", "none"),
            **lambdas,
        )


class JointLoss(nn.Module):
    def __init__(self, cfg: JointConfig, w_neg: float, w_pos: float) -> None:
        super().__init__()
        self.cfg, self.w_neg, self.w_pos = cfg, w_neg, w_pos
        self.epoch = 0.0
        self._acc: dict[str, list[float]] = {
            "fair": [0.0, 0.0],
            "cal": [0.0, 0.0],
            "adv": [0.0, 0.0],
        }
        self.ref: dict[str, float] | None = None  # frozen when the ramp starts

    def _targets(self) -> dict[str, float]:
        return {"fair": self.cfg.lambda_f, "cal": self.cfg.lambda_c, "adv": self.cfg.lambda_a}

    def _raw_terms(self, logit, y, a, adv_logit, names) -> dict[str, torch.Tensor]:
        c, out = self.cfg, {}
        if "fair" in names:
            out["fair"] = fairness_loss(logit, y, a, c.fairness_type, c.flag_rate, c.temperature)
        if "cal" in names:
            out["cal"] = soft_group_ece(logit, y, a, c.cal_bins, c.cal_sigma)
        if "adv" in names:
            if adv_logit is None:
                raise ValueError("lambda_a > 0 needs the adversary logit (use JointModel)")
            out["adv"] = F.binary_cross_entropy_with_logits(adv_logit.float(), a.float())
        return out

    def _observe_warmup(self, logit, y, a, adv_logit) -> None:
        """Accumulate term magnitudes over the last warm-up epoch (no grad, no effect on loss)."""
        c = self.cfg
        if c.normalize != "initial" or self.epoch < c.warmup_epochs - 1:
            return
        names = [
            k for k, v in self._targets().items() if v > 0 and (k != "adv" or adv_logit is not None)
        ]
        with torch.no_grad():
            for k, v in self._raw_terms(logit.detach(), y, a, adv_logit, names).items():
                self._acc[k][0] += float(v)
                self._acc[k][1] += 1

    def _freeze_reference(self, first_batch: dict[str, torch.Tensor]) -> None:
        """Reference = warm-up mean; if there was no warm-up, fall back to the first ramp batch."""
        ref = {}
        for k, (tot, n) in self._acc.items():
            v = tot / n if n else (float(first_batch[k].detach()) if k in first_batch else 1.0)
            ref[k] = max(v, self.cfg.ref_floor)
        self.ref = ref

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
        if self.epoch < c.warmup_epochs:
            self._observe_warmup(logit, y, a, adv_logit)
        key = {"fair": "f", "cal": "c", "adv": "a"}
        active = [k for k, short in key.items() if lam[short] > 0]
        terms = self._raw_terms(logit, y, a, adv_logit, active)
        if c.normalize == "initial" and active and self.ref is None:
            self._freeze_reference(terms)
        for k in active:  # fixed order: fair, cal, adv
            scale = 1.0 / self.ref[k] if (c.normalize == "initial" and self.ref) else 1.0
            total = total + lam[key[k]] * scale * terms[k]
            parts[k] = terms[k].detach()
        return total, parts
