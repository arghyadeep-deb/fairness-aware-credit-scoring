"""Regression tests for the Phase 2 review fixes (A1b, REF, S2-EG, selection rule)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest

from src.data.tensors import smote_nc
from src.run_arms import arm_cfg, match_flag_weight
from src.utils import ROOT, load_config

CFG = load_config("configs/train.yaml")


def test_arm_configs_for_review_arms():
    a1b = arm_cfg("a1b")
    assert a1b["sampling"] == "smote_nc" and a1b["smote_ratio"] == 1.0 and a1b["tuned"]
    assert arm_cfg("a1").get("smote_ratio") is None  # A1 keeps the global ratio 0.5
    assert (
        arm_cfg("s2_gt")["type"] == "posthoc_groupthr"
        and "not deployable" in arm_cfg("s2_gt")["description"]
    )
    assert (
        arm_cfg("s2_eg")["type"] == "posthoc_eg"
        and "No gender at inference" in arm_cfg("s2_eg")["description"]
    )


def test_smote_ratio_one_balances_classes():
    from tests.test_models_losses import small_arrays

    out = smote_nc(small_arrays(), 1.0, 5, 0)
    assert out.y.sum() == (out.y == 0).sum()


def test_optuna_uses_tpe_after_four_startup_trials():
    assert CFG["optuna"]["n_startup_trials"] == 4 and CFG["optuna"]["trials"] > 4


def test_ref_early_stopping_ignores_logloss_with_scale_pos_weight():
    """Regression: tracking logloss under scale_pos_weight stopped REF at round 1."""
    r = np.random.default_rng(0)
    n = 60000
    X = r.normal(size=(n, 8))
    lg = -2.5 + 0.5 * X[:, 0] - 0.4 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3] + 0.3 * np.sin(2 * X[:, 4])
    y = (r.random(n) < 1 / (1 + np.exp(-lg))).astype(int)
    tr, va = slice(0, 45000), slice(45000, n)
    common = dict(
        n_estimators=1500,
        learning_rate=0.05,
        verbose=-1,
        random_state=0,
        n_jobs=1,
        num_leaves=15,
        min_child_samples=100,
        scale_pos_weight=11.0,
    )
    bad = lgb.LGBMClassifier(**common)  # default logloss is tracked too -> stops immediately
    bad.fit(
        X[tr],
        y[tr],
        eval_set=[(X[va], y[va])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    good = lgb.LGBMClassifier(**common, metric="auc")
    good.fit(
        X[tr],
        y[tr],
        eval_set=[(X[va], y[va])],
        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
    )
    assert bad.best_iteration_ <= 5 and good.best_iteration_ >= 100
    ref = CFG["reference"]["lightgbm"]
    assert ref["metric"] == "auc" and "scale_pos_weight" not in ref
    assert ref["early_stopping_rounds"] >= 100


def test_match_flag_weight_hits_target_flag_rate():
    r = np.random.default_rng(1)
    z = r.normal(size=(6000, 8))
    y = (r.random(6000) < 1 / (1 + np.exp(-(-2.4 + z[:, 0])))).astype(int)
    w = match_flag_weight(z, y, 0.10, 200)
    from sklearn.linear_model import LogisticRegression

    rate = LogisticRegression(max_iter=200, class_weight={0: 1.0, 1: w}).fit(z, y).predict(z).mean()
    assert rate == pytest.approx(0.10, abs=0.03) and w > 1


def test_selection_rule_file_matches_spec_and_is_tracked_before_sweeps():
    import subprocess

    import yaml

    rule = yaml.safe_load((ROOT / "configs" / "selection_rule.yaml").read_text())
    c = rule["constraints"]
    assert c["eod_max"] == 0.04 and c["dir_min"] == 0.80 and c["auc_roc_margin"] == 0.02
    assert c["worst_group_ece"] == {"multiplier": 1.5, "floor": 0.02}
    assert rule["objective"] == "auc_pr" and rule["data_split"] == "validation"
    assert [s["to"] for s in rule["relaxation_order"]] == [2.0, 0.03]
    tracked = subprocess.run(
        ["git", "ls-files", "configs/selection_rule.yaml"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    assert tracked == "configs/selection_rule.yaml"
    assert not list((ROOT / "artifacts" / "runs").glob("*sweep*"))  # no sweep exists yet
