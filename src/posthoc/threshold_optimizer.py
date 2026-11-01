"""S2: sequential fairness patching on top of S1 (needs gender at decision time).

Primary S2 keeps the shared flag rate (specs/09: identical for every arm): it searches one
degree of freedom, the share of each group flagged, so overall flags stay at flag_rate while
the equalized-odds gap max(|dTPR|, |dFPR|) is minimised. fairlearn's ThresholdOptimizer fixes
its own operating point, so it is run separately as S2_fairlearn (not rate-matched).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _group_flags(p: np.ndarray, a: np.ndarray, q_f: float, flag_rate: float) -> np.ndarray | None:
    """Flag the top q_f of group 0 and the share of group 1 that keeps the overall rate."""
    n, n_f = len(p), int((a == 0).sum())
    n_m = n - n_f
    q_m = (flag_rate * n - q_f * n_f) / n_m
    if not 0 <= q_m <= 1:
        return None
    flags = np.zeros(n, dtype=bool)
    for g, q in ((0, q_f), (1, q_m)):
        idx = np.flatnonzero(a == g)
        k = int(round(q * len(idx)))
        if k > 0:
            flags[idx[np.argsort(-p[idx], kind="mergesort")[:k]]] = True
    return flags


def _eod_gap(y: np.ndarray, a: np.ndarray, flags: np.ndarray) -> float:
    gaps = []
    for t in (1, 0):
        r = [flags[(a == g) & (y == t)].mean() for g in (0, 1)]
        gaps.append(abs(r[0] - r[1]))
    return max(gaps)


def fit_group_thresholds(
    y: np.ndarray, p: np.ndarray, a: np.ndarray, flag_rate: float, grid: int = 400
) -> dict[str, float]:
    """Choose per-group flag shares on VALIDATION; returns group score thresholds."""
    best = (np.inf, None)
    n_f = (a == 0).sum()
    q_lo = max(0.0, (flag_rate * len(p) - (a == 1).sum()) / n_f)
    q_hi = min(1.0, flag_rate * len(p) / n_f)
    for q_f in np.linspace(q_lo, q_hi, grid):
        fl = _group_flags(p, a, q_f, flag_rate)
        if fl is None:
            continue
        gap = _eod_gap(y, a, fl)
        if gap < best[0]:
            best = (gap, fl)
    flags = best[1]
    thr = {}
    for g in (0, 1):
        sel = p[(a == g) & flags]
        thr[str(g)] = float(sel.min()) if len(sel) else float("inf")
    return {"threshold_f": thr["0"], "threshold_m": thr["1"], "val_eod": float(best[0])}


def apply_group_thresholds(p: np.ndarray, a: np.ndarray, thr: dict[str, float]) -> np.ndarray:
    """Flags from fitted group thresholds (>= because thresholds are the smallest flagged score)."""
    t = np.where(a == 0, thr["threshold_f"], thr["threshold_m"])
    return p >= t


class _ScoreAsModel(BaseEstimator, ClassifierMixin):
    """Wraps stored scores as a prefit classifier so fairlearn can post-process them."""

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.stack([1 - s, s], axis=1)


def fairlearn_threshold_flags(
    y: np.ndarray, p: np.ndarray, a: np.ndarray, seed: int = 0
) -> np.ndarray:
    """fairlearn ThresholdOptimizer (equalized odds, balanced accuracy), native operating point."""
    from fairlearn.postprocessing import ThresholdOptimizer

    to = ThresholdOptimizer(
        estimator=_ScoreAsModel(),
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    X = p.reshape(-1, 1)
    to.fit(X, y, sensitive_features=a)
    return np.asarray(to.predict(X, sensitive_features=a, random_state=seed)).astype(bool)
