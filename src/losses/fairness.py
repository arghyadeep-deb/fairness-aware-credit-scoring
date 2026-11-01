"""Differentiable fairness surrogates: soft equalized odds (eo) and soft demographic parity (dp).

A hard flag (score above the flag-rate threshold) has no gradient, so each sample gets a soft
flag s_i = sigmoid((logit_i - tau) / (T * std)) where tau is the batch (1 - flag_rate) quantile.
Standardising by the batch std keeps temperature T meaningful whatever the logit scale
(weighted CE inflates logits). tau and std are detached; gradients flow through logit_i only.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def soft_flags(logits: torch.Tensor, flag_rate: float, temperature: float) -> torch.Tensor:
    """Soft indicator of 'in the top flag_rate of this batch'. logits (B,) -> (B,) in (0, 1)."""
    lg = logits.float()
    tau = torch.quantile(lg.detach(), 1.0 - flag_rate)
    scale = lg.detach().std().clamp_min(1e-6)
    return torch.sigmoid((lg - tau) / (temperature * scale))


def _gap(values: torch.Tensor, mask0: torch.Tensor, mask1: torch.Tensor, what: str) -> torch.Tensor:
    """|mean over group 0 - mean over group 1|, or 0 (with a warning) if a group is absent."""
    if not bool(mask0.any()) or not bool(mask1.any()):
        log.warning("fairness term '%s' skipped: a group is absent from the batch", what)
        return values.new_zeros(())
    return (values[mask0].mean() - values[mask1].mean()).abs()


def fairness_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    kind: str = "eo",
    flag_rate: float = 0.10,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Soft fairness gap between genders. logits/y/a are (B,); a in {0,1}.

    eo: |soft TPR gap| + |soft FPR gap|.   dp: |soft flag-rate gap|.
    """
    if kind not in ("eo", "dp"):
        raise ValueError(f"fairness type must be 'eo' or 'dp', got {kind}")
    s = soft_flags(logits, flag_rate, temperature)
    g0, g1 = a == 0, a == 1
    if kind == "dp":
        return _gap(s, g0, g1, "dp")
    pos, neg = y > 0.5, y <= 0.5
    return _gap(s, g0 & pos, g1 & pos, "tpr") + _gap(s, g0 & neg, g1 & neg, "fpr")
