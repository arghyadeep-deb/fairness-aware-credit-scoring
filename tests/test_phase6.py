"""Phase 6 helpers: matched pairs, delta statistics, new-proxy check, EG weight matching,
per-group calibrator fit/apply, and the test-split guard."""

import numpy as np
import pytest

from src.data.tensors import Arrays, load_arrays
from src.eval.final import delta_stats, matched_pairs, new_proxy_check
from src.posthoc.eg import fit_eg, match_eg_weight, pmf
from src.posthoc.phase5 import pergroup_fit
from src.utils import load_config


def _arr(n=400, seed=0):
    r = np.random.default_rng(seed)
    cont = r.normal(size=(n, 5)).astype("float32")
    cat = r.integers(0, 3, size=(n, 2)).astype("int64")
    a = r.integers(0, 2, n)
    return Arrays(cont, cat, r.integers(0, 2, n), a, list("abcde"), ["c1", "c2"], [])


def test_matched_pairs_are_other_gender_and_nearest():
    t = _arr()
    q, m = matched_pairs(t, n_query=100, k_cand=50, seed=0)
    assert np.all(t.a[q] != t.a[m])
    # brute-force check for a few queries
    for i in range(5):
        cand = np.flatnonzero(t.a != t.a[q[i]])
        d = np.linalg.norm(t.cont[cand] - t.cont[q[i]], axis=1) + 0.5 * (
            t.cat[cand] != t.cat[q[i]]
        ).sum(1)
        assert d[np.argmin(np.abs(cand - m[i]))] <= np.sort(d)[2] + 1e-6  # within the best few


def test_delta_stats_values():
    p = np.array([0.1, 0.2, 0.5, 0.6])
    s = delta_stats(p, np.array([0, 1]), np.array([2, 3]))
    assert abs(s["mean"] - 0.4) < 1e-9 and s["max"] > 0.39


def test_new_proxy_check_flags_new_features():
    names = ["f0", "f1", "f2", "f3"]
    a = np.array([[5.0, 4, 0.1, 0.0]] * 10)
    b = np.array([[5.0, 0.1, 4, 0.0]] * 10)
    r = new_proxy_check(a, b, names, {"f2": 0.7, "f1": 0.5}, top=2)
    assert r["new_in_B"] == ["f2"] and r["new_proxy_flags"]["f2"] is True


def test_eg_weight_matching_hits_target_rate():
    r = np.random.default_rng(0)
    n = 3000
    a = r.integers(0, 2, n)
    z = r.normal(size=(n, 4))
    y = (r.random(n) < 1 / (1 + np.exp(-(z[:, 0] * 1.5 - 2.2 + 0.4 * a)))).astype(int)
    w, rate = match_eg_weight(z, y, a, 0.15, 0.01, 30, 200, steps=5)
    assert abs(rate - 0.15) < 0.04
    eg = fit_eg(z, y, a, w, 0.01, 30, 200)
    assert 0 <= pmf(eg, z).min() and pmf(eg, z).max() <= 1


def test_pergroup_fit_apply_matches_cross_fit_shape():
    r = np.random.default_rng(1)
    n = 2000
    a = r.integers(0, 2, n)
    lg = r.normal(-2, 1, n)
    y = (r.random(n) < 1 / (1 + np.exp(-(lg - 0.5)))).astype(int)
    for kind in ("platt", "isotonic"):
        p = pergroup_fit(y, lg, a, kind)(lg, a)
        assert p.shape == (n,) and p.min() >= 0 and abs(p.mean() - y.mean()) < 0.02


def test_test_split_is_locked_without_allow_test(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(PermissionError):
        load_arrays(load_config("configs/train.yaml"), "test")
