"""Fixed metric battery (specs/09), identical for every arm.

Pure numpy functions on (y_true, y_prob, group, flags). Group a: 0 = female, 1 = male.
Favourable outcome = approved (not flagged).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CLIP = 1e-6
METRIC_NAMES = [
    "auc_roc",
    "auc_pr",
    "ks",
    "brier",
    "ece",
    "ece_f",
    "ece_m",
    "worst_group_ece",
    "cal_slope",
    "cal_intercept",
    "dpd",
    "eod",
    "dir",
    "tpr_f",
    "tpr_m",
    "fpr_f",
    "fpr_m",
    "recall",
    "precision",
    "flag_rate",
]


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Max gap between the score CDFs of defaulters and non-defaulters.

    The gap is read only at the end of each block of equal scores, so tied scores (bf16, isotonic)
    neither inflate KS nor make it depend on row order (before 2026-11-03: read after every row).
    """
    order = np.argsort(p, kind="mergesort")
    ys, ps = y[order], p[order]
    n1, n0 = ys.sum(), len(ys) - ys.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    gap = np.abs(np.cumsum(ys) / n1 - np.cumsum(1 - ys) / n0)
    block_end = np.r_[ps[1:] != ps[:-1], True]
    return float(np.max(gap[block_end]))


def ece_equal_freq(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-frequency bins; empty bins are skipped."""
    p = np.clip(p, CLIP, 1 - CLIP)
    order = np.argsort(p, kind="mergesort")
    n = len(p)
    total = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            total += len(idx) / n * abs(y[idx].mean() - p[idx].mean())
    return float(total)


def calibration_slope_intercept(
    y: np.ndarray, p: np.ndarray, iters: int = 25
) -> tuple[float, float]:
    """Logistic recalibration y ~ sigmoid(intercept + slope * logit(p)) via Newton's method."""
    p = np.clip(p, CLIP, 1 - CLIP)
    x = np.log(p / (1 - p))
    X = np.stack([np.ones_like(x), x], axis=1)
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        grad = X.T @ (y - mu)
        hess = (X * w[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[1]), float(beta[0])


def _rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def compute_metrics(
    y: np.ndarray, p: np.ndarray, a: np.ndarray, flags: np.ndarray, ece_bins: int = 15
) -> dict[str, float]:
    """All battery metrics. `flags` come from the shared threshold rule (or S2's group rule)."""
    y, a, flags = (
        np.asarray(y).astype(int),
        np.asarray(a).astype(int),
        np.asarray(flags).astype(bool),
    )
    p = np.clip(np.asarray(p, dtype="float64"), CLIP, 1 - CLIP)
    out: dict[str, float] = {}
    has_both = 0 < y.sum() < len(y)
    out["auc_roc"] = float(roc_auc_score(y, p)) if has_both else float("nan")
    out["auc_pr"] = float(average_precision_score(y, p)) if has_both else float("nan")
    out["ks"] = ks_statistic(y, p)
    out["brier"] = float(np.mean((p - y) ** 2))
    out["ece"] = ece_equal_freq(y, p, ece_bins)
    for g, key in ((0, "ece_f"), (1, "ece_m")):
        m = a == g
        out[key] = ece_equal_freq(y[m], p[m], ece_bins) if m.any() else float("nan")
    out["worst_group_ece"] = float(np.nanmax([out["ece_f"], out["ece_m"]]))
    out["cal_slope"], out["cal_intercept"] = calibration_slope_intercept(y, p)
    approve = {g: 1.0 - flags[a == g].mean() if (a == g).any() else float("nan") for g in (0, 1)}
    out["dpd"] = abs(approve[1] - approve[0])
    hi, lo = max(approve.values()), min(approve.values())
    out["dir"] = float(lo / hi) if hi > 0 else float("nan")  # NaN, never coerced, if undefined
    for g, s in ((0, "f"), (1, "m")):
        m = a == g
        out[f"tpr_{s}"] = _rate(flags[m & (y == 1)].sum(), (m & (y == 1)).sum())
        out[f"fpr_{s}"] = _rate(flags[m & (y == 0)].sum(), (m & (y == 0)).sum())
    out["eod"] = float(max(abs(out["tpr_m"] - out["tpr_f"]), abs(out["fpr_m"] - out["fpr_f"])))
    out["recall"] = _rate(flags[y == 1].sum(), (y == 1).sum())
    out["precision"] = _rate(flags[y == 1].sum(), flags.sum())
    out["flag_rate"] = float(flags.mean())
    return out
