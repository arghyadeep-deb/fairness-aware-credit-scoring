"""The ONE decision-threshold rule shared by every arm (CRITICAL_RULES 8): top x% flagged."""

from __future__ import annotations

import numpy as np


def flag_threshold(val_scores: np.ndarray, flag_rate: float) -> float:
    """Score above which an applicant is flagged, chosen on VALIDATION scores only."""
    if not 0 < flag_rate < 1:
        raise ValueError(f"flag_rate must be in (0, 1), got {flag_rate}")
    return float(np.quantile(val_scores, 1.0 - flag_rate))


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean flags (True = flagged = not approved)."""
    return np.asarray(scores) > threshold
