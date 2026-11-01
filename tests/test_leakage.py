"""Stage 2 tests: the probe finds planted signal, gives ~0.5 on noise, and never sees TARGET."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.eval.leakage import auc, fit_lgbm_probe, fit_logistic_probe
from src.utils import load_config, resolve

CFG = load_config("configs/data.yaml")
FAST = {**CFG["leakage"]["lightgbm"], "n_estimators": 100, "num_leaves": 15}


def _frame(n=6000, signal=True, seed=0):
    r = np.random.default_rng(seed)
    g = r.integers(0, 2, n)
    x = pd.DataFrame(
        {
            "a": r.normal(size=n) + (1.5 * g if signal else 0),
            "b": r.normal(size=n),
            "c": r.integers(0, 3, n),
        }
    )
    return x, g


def test_probe_detects_planted_signal():
    x, g = _frame(signal=True)
    m = fit_lgbm_probe(x[:3000], g[:3000], x[3000:4500], g[3000:4500], ["c"], FAST, 0)
    assert auc(m, x[4500:], g[4500:]) > 0.8
    lm = fit_logistic_probe(x[:3000], g[:3000], ["c"], CFG["leakage"]["logistic"], 0)
    assert auc(lm, x[4500:], g[4500:]) > 0.8


def test_probe_is_chance_on_noise_and_shuffled_labels():
    x, g = _frame(signal=False)
    m = fit_lgbm_probe(x[:3000], g[:3000], x[3000:4500], g[3000:4500], ["c"], FAST, 0)
    assert abs(auc(m, x[4500:], g[4500:]) - 0.5) < 0.05
    x2, g2 = _frame(signal=True, seed=1)
    perm = np.random.default_rng(3).permutation(g2)
    m2 = fit_lgbm_probe(x2[:3000], perm[:3000], x2[3000:4500], perm[3000:4500], ["c"], FAST, 0)
    assert abs(auc(m2, x2[4500:], perm[4500:]) - 0.5) < 0.05


def test_real_before_json_contract():
    path = resolve(CFG["paths"]["leakage_artifacts"]) / "before.json"
    assert path.exists(), "run: python -m src.eval.leakage --stage before"
    r = json.loads(path.read_text())
    assert r["uses_target"] is False and r["smoke"] is False
    assert "TARGET" not in r["config"]["features"] and not any(
        "GENDER" in f.upper() for f in r["config"]["features"]
    )
    assert abs(r["shuffled_control"]["test_auc"] - 0.5) <= CFG["leakage"]["shuffle_tolerance"]
    assert len(r["top_features"]) == 15 and r["seed"] == 42
    assert r["lightgbm"]["test_auc"] > 0.5 and r["logistic"]["test_auc"] > 0.5
    assert {"timestamp_utc", "git_hash", "config"} <= set(r)
    assert (path.parent / "top15_proxy_features_before.png").stat().st_size > 1000


def test_guard_log_has_only_gender_only_probe_entries():
    log = resolve(CFG["paths"]["guard_log"])
    assert log.exists()
    for line in log.read_text().splitlines():
        e = json.loads(line)
        assert e["purpose"] == "leakage_probe_gender_only" and e["with_target"] is False
