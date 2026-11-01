"""S1: analytic prior correction that undoes class-weight inflation of the logits."""

from __future__ import annotations

import numpy as np


def prior_shift(class_weights: dict[str, float]) -> float:
    """log(w_pos / w_neg): the logit offset that class weighting adds."""
    return float(np.log(class_weights["1"] / class_weights["0"]))


def correct_logits(logits: np.ndarray, class_weights: dict[str, float]) -> np.ndarray:
    """Subtract the weighting offset. Monotone, so rankings and flagged sets are unchanged."""
    return np.asarray(logits, dtype="float64") - prior_shift(class_weights)
