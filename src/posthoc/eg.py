"""S2-EG helpers (fairlearn ExponentiatedGradient on frozen z; gender-blind at inference).

Phase 6 diagnosis: the Phase 2 S2-EG matched the flag rate of the BASE classifier (about 10%)
before EG, but EG's own mixture flags about 9.6%. The shared 10% rule then appended the missing
rows by S1 rank, which broke the equalized-odds guarantee (EOD 0.009 -> 0.032). The fix matches
the flag rate of the EG mixture itself (`match_eg_weight`), so the shared rule changes almost no
decisions. eps barely matters because EG reaches a zero-gap solution within a few iterations.
"""

from __future__ import annotations

import numpy as np
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression


def fit_eg(ztr, ytr, atr, w_pos: float, eps: float, max_iter: int, lr_iter: int):
    base = LogisticRegression(max_iter=lr_iter, class_weight={0: 1.0, 1: float(w_pos)})
    eg = ExponentiatedGradient(base, constraints=EqualizedOdds(), eps=eps, max_iter=max_iter)
    eg.fit(ztr, ytr, sensitive_features=atr)
    return eg


def pmf(eg, z) -> np.ndarray:
    """P(flag) of the randomized predictor (mixture over the stored predictors)."""
    return eg._pmf_predict(z)[:, 1]


def match_eg_weight(
    ztr, ytr, atr, target: float, eps: float, max_iter: int, lr_iter: int, steps: int = 7
) -> tuple[float, float]:
    """Bisection on log(w_pos) so that the EG mixture flags `target` of TRAIN rows. Returns
    (w_pos, achieved train rate). Uses train data only."""
    lo, hi = 0.0, float(np.log(40.0))
    best = (np.inf, 1.0, 0.0)
    for _ in range(steps):
        mid = (lo + hi) / 2
        w = float(np.exp(mid))
        rate = float(pmf(fit_eg(ztr, ytr, atr, w, eps, max_iter, lr_iter), ztr).mean())
        if abs(rate - target) < best[0]:
            best = (abs(rate - target), w, rate)
        lo, hi = (mid, hi) if rate < target else (lo, mid)
    return best[1], best[2]
