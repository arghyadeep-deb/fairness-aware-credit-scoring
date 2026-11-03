"""Metric battery, shared threshold rule, bootstrap and post-hoc arms."""

from __future__ import annotations

import numpy as np
import pytest
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)

from src.eval.bootstrap import bootstrap_metrics, cell_index, pooled_ci, resample
from src.eval.metrics import (
    METRIC_NAMES,
    calibration_slope_intercept,
    compute_metrics,
    ece_equal_freq,
    ks_statistic,
)
from src.eval.threshold import apply_threshold, flag_threshold
from src.posthoc.prior_correction import correct_logits, prior_shift
from src.posthoc.threshold_optimizer import (
    apply_group_thresholds,
    fairlearn_threshold_flags,
    fit_group_thresholds,
)


def synth(n=20000, seed=0, bias=0.0):
    r = np.random.default_rng(seed)
    a = r.integers(0, 2, n)
    y = r.binomial(1, 0.08 + 0.03 * a, n)
    s = r.normal(size=n) + 1.2 * y + bias * a
    return y, 1 / (1 + np.exp(-(s - 2.6))), a


def test_ece_zero_for_perfectly_calibrated_and_large_for_overconfident():
    r = np.random.default_rng(0)
    p = r.uniform(0.01, 0.99, 200000)
    y = r.binomial(1, p)
    assert ece_equal_freq(y, p) < 0.01
    assert ece_equal_freq(y, np.clip(p * 0.2, 1e-6, 1)) > 0.2


def test_ks_ignores_order_inside_tied_scores():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.full(4, 0.3)) == 0.0  # identical score distributions
    rng = np.random.default_rng(0)
    p = rng.integers(0, 5, 400) / 4.0  # heavy ties
    yy = (rng.random(400) < p * 0.5).astype(int)
    vals = {ks_statistic(yy[o], p[o]) for o in (rng.permutation(400) for _ in range(20))}
    assert len(vals) == 1


def test_ks_and_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    m = compute_metrics(
        y, np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])
    )
    assert (
        m["auc_roc"] == 1.0 and m["auc_pr"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0
    )


def test_calibration_slope_intercept_recovers_known_miscalibration():
    r = np.random.default_rng(1)
    lg = r.normal(-2, 1.5, 100000)
    p_true = 1 / (1 + np.exp(-(0.3 + 0.7 * lg)))
    y = r.binomial(1, p_true)
    slope, inter = calibration_slope_intercept(y, 1 / (1 + np.exp(-lg)))
    assert slope == pytest.approx(0.7, abs=0.03) and inter == pytest.approx(0.3, abs=0.06)


def test_threshold_rule_flag_rate_and_identical_across_arms():
    r = np.random.default_rng(2)
    for _ in range(3):  # different arms, different score distributions, same flag rate
        s = r.normal(size=5000) * r.uniform(0.5, 3) + r.uniform(-2, 2)
        flags = apply_threshold(s, flag_threshold(s, 0.10))
        assert flags.mean() == pytest.approx(0.10, abs=0.002)
    with pytest.raises(ValueError):
        flag_threshold(s, 1.5)


def test_fairness_metrics_match_fairlearn():
    y, p, a = synth(bias=0.6)
    flags = apply_threshold(p, flag_threshold(p, 0.10))
    m = compute_metrics(y, p, a, flags)
    assert m["dpd"] == pytest.approx(
        demographic_parity_difference(y, flags, sensitive_features=a), abs=1e-9
    )
    assert m["eod"] == pytest.approx(
        equalized_odds_difference(y, flags, sensitive_features=a), abs=1e-9
    )


def test_dir_is_one_when_groups_treated_alike_and_nan_when_undefined():
    y, p, a = synth()
    same = np.zeros(len(y), bool)
    same[::10] = True
    assert compute_metrics(y, p, a, same)["dir"] == pytest.approx(1.0, abs=0.01)
    allflag = np.ones(len(y), bool)  # approval 0 for both groups -> undefined, not coerced to 1
    assert np.isnan(compute_metrics(y, p, a, allflag)["dir"])


def test_probabilities_clipped_so_extreme_scores_do_not_nan():
    y = np.array([0, 1, 0, 1])
    a = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.0, 1.0, 0.0, 1.0]), a, np.array([0, 1, 0, 1]))
    assert all(np.isfinite(m[k]) for k in ("brier", "ece", "cal_slope"))


def test_bootstrap_is_stratified_deterministic_and_chunk_independent():
    y, p, a = synth(4000)
    cells = cell_index(y, a)
    ix = resample(cells, 0, 5)
    assert len(ix) == len(y)
    for c in cells:  # each cell keeps its own size
        assert np.isin(ix, c).sum() == len(c)
    assert (resample(cells, 0, 5) == ix).all() and not (resample(cells, 0, 6) == ix).all()
    flags = apply_threshold(p, flag_threshold(p, 0.1))
    b1 = bootstrap_metrics(y, p, a, flags, 12, 0, n_jobs=1)
    b2 = bootstrap_metrics(y, p, a, flags, 12, 0, n_jobs=2)
    np.testing.assert_allclose(b1, b2)
    assert b1.shape == (12, len(METRIC_NAMES))


def test_bootstrap_ci_covers_point_estimate():
    y, p, a = synth(6000)
    flags = apply_threshold(p, flag_threshold(p, 0.1))
    b = bootstrap_metrics(y, p, a, flags, 100, 0, n_jobs=2)
    lo, hi = pooled_ci([b])
    pt = compute_metrics(y, p, a, flags)
    i = METRIC_NAMES.index("auc_roc")
    assert lo[i] <= pt["auc_roc"] <= hi[i]


def test_prior_correction_restores_calibration_of_weighted_model():
    r = np.random.default_rng(3)
    true_logit = r.normal(-2.5, 1.2, 300000)
    y = r.binomial(1, 1 / (1 + np.exp(-true_logit)))
    w = {"0": 0.54, "1": 6.2}
    inflated = true_logit + prior_shift(w)  # what a class-weighted model learns
    raw = ece_equal_freq(y, 1 / (1 + np.exp(-inflated)))
    fixed = ece_equal_freq(y, 1 / (1 + np.exp(-correct_logits(inflated, w))))
    assert raw > 0.1 and fixed < 0.01
    ranks = np.argsort(inflated)
    assert (np.argsort(correct_logits(inflated, w)) == ranks).all()  # monotone


def test_group_thresholds_keep_flag_rate_and_shrink_equalized_odds_gap():
    y, p, a = synth(30000, bias=0.8)
    base = apply_threshold(p, flag_threshold(p, 0.10))
    thr = fit_group_thresholds(y, p, a, 0.10)
    flags = apply_group_thresholds(p, a, thr)
    assert flags.mean() == pytest.approx(0.10, abs=0.002)
    gap = lambda f: equalized_odds_difference(y, f, sensitive_features=a)  # noqa: E731
    assert gap(flags) < gap(base)


def test_fairlearn_threshold_optimizer_runs():
    y, p, a = synth(5000, bias=0.5)
    fl = fairlearn_threshold_flags(y, p, a, seed=0)
    assert fl.shape == y.shape and fl.dtype == bool
