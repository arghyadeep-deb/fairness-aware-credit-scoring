"""Phase 5 helpers (validation only): per-group Platt/isotonic (analysis only, uses gender),
matched-fairness operating points for the comparators that have a fairness knob.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.eval.metrics import compute_metrics
from src.eval.threshold import apply_threshold, flag_threshold


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype="float64")))


def pergroup_calibrate(
    y: np.ndarray, logits: np.ndarray, a: np.ndarray, kind: str, seed: int
) -> np.ndarray:
    """Per-gender-group Platt ('platt') or isotonic ('isotonic') recalibration of `logits`.

    2-fold stratified (TARGET x gender) cross-fitting inside validation: each half is scored by
    calibrators fitted on the other half, so the calibrator never scores the rows it saw.
    Uses gender at inference: ANALYSIS ONLY.
    """
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), dtype=int)
    for t in (0, 1):
        for g in (0, 1):
            ix = np.flatnonzero((y == t) & (a == g))
            fold[ix[rng.permutation(len(ix))[: len(ix) // 2]]] = 1
    out = np.zeros(len(y))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        for g in (0, 1):
            fit, app = tr & (a == g), te & (a == g)
            x, t_ = np.asarray(logits)[fit], y[fit]
            if kind == "platt":
                m = LogisticRegression(C=1e6, max_iter=500).fit(x.reshape(-1, 1), t_)
                out[app] = m.predict_proba(np.asarray(logits)[app].reshape(-1, 1))[:, 1]
            elif kind == "isotonic":
                m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, t_)
                out[app] = m.predict(np.asarray(logits)[app])
            else:
                raise ValueError(kind)
    return out + 1e-9 * _sig(logits)  # tie-break so isotonic plateaus do not break the flag rate


def flags_at_rate(score: np.ndarray, rate: float) -> np.ndarray:
    return apply_threshold(score, flag_threshold(score, rate))


def metrics_row(y, p, a, flags, ece_bins: int = 15) -> dict:
    m = compute_metrics(y, p, a, flags, ece_bins)
    m["flag_rate"] = float(np.mean(flags))
    tp = float((flags & (y == 1)).sum())
    m["precision"] = tp / max(1.0, float(flags.sum()))
    return m


def gt_fraction_flags(prob, a, thr: dict, shared_thr: float, t: float) -> np.ndarray:
    """S2-GT thresholds moved a fraction t from the shared threshold to the fitted ones."""
    tf = shared_thr + t * (thr["threshold_f"] - shared_thr)
    tm = shared_thr + t * (thr["threshold_m"] - shared_thr)
    return prob >= np.where(a == 0, tf, tm)


def nearest_eod_fraction(y, prob, a, thr, shared_thr, target, ece_bins=15, grid: int = 101):
    """Fraction t in [0, 1] whose val EOD is closest to `target` (analysis-only comparator)."""
    best = (np.inf, 0.0)
    for t in np.linspace(0, 1, grid):
        fl = gt_fraction_flags(prob, a, thr, shared_thr, t)
        e = compute_metrics(y, prob, a, fl, ece_bins)["eod"]
        if abs(e - target) < best[0]:
            best = (abs(e - target), float(t))
    return best[1]


def pergroup_fit(y, logits, a, kind: str):
    """Fit per-gender-group Platt or isotonic calibrators on (logits, y); returns apply(logits, a)."""
    ms = {}
    for g in (0, 1):
        ix = a == g
        if kind == "platt":
            ms[g] = LogisticRegression(C=1e6, max_iter=500).fit(
                np.asarray(logits)[ix].reshape(-1, 1), y[ix]
            )
        else:
            ms[g] = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                np.asarray(logits)[ix], y[ix]
            )

    def apply(lg, aa):
        lg = np.asarray(lg, dtype="float64")
        out = np.zeros(len(lg))
        for g in (0, 1):
            ix = aa == g
            if kind == "platt":
                out[ix] = ms[g].predict_proba(lg[ix].reshape(-1, 1))[:, 1]
            else:
                out[ix] = ms[g].predict(lg[ix])
        return out + 1e-9 * _sig(lg)

    return apply


def fairlearn_fit_predict(y_val, p_val, a_val, p_test, a_test, seed: int = 0) -> np.ndarray:
    """fairlearn ThresholdOptimizer fitted on VALIDATION scores, applied to test (analysis only)."""
    from fairlearn.postprocessing import ThresholdOptimizer

    from src.posthoc.threshold_optimizer import _ScoreAsModel

    to = ThresholdOptimizer(
        estimator=_ScoreAsModel(),
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    to.fit(np.asarray(p_val).reshape(-1, 1), y_val, sensitive_features=a_val)
    pred = to.predict(np.asarray(p_test).reshape(-1, 1), sensitive_features=a_test, random_state=seed)
    return np.asarray(pred).astype(bool)
