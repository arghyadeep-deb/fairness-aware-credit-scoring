"""Stage 2 proxy-leakage audit ("before"): can gender be predicted from the features X alone?

TARGET is never used here. Test AUCs are reported because specs/08 requires them; the test
split is read gender-only through the guarded loader and nothing is tuned on it.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from src.data.io import load_split  # noqa: E402
from src.utils import git_hash, load_config, resolve, seed_everything  # noqa: E402

log = logging.getLogger(__name__)


def fit_lgbm_probe(
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xva: pd.DataFrame,
    yva: np.ndarray,
    cat_cols: list[str],
    params: dict[str, Any],
    seed: int,
) -> lgb.LGBMClassifier:
    """LightGBM gender probe; early stopping on validation AUC only."""
    p = dict(params)
    stop = p.pop("early_stopping_rounds")
    model = lgb.LGBMClassifier(**p, objective="binary", random_state=seed, n_jobs=-1, verbose=-1)
    model.fit(
        xtr,
        ytr,
        eval_set=[(xva, yva)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stop, verbose=False)],
    )
    return model


def fit_logistic_probe(
    xtr: pd.DataFrame, ytr: np.ndarray, cat_cols: list[str], params: dict[str, Any], seed: int
):
    """Logistic reference probe; categoricals one-hot (index labels are not ordinal)."""
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough"
    )
    clf = LogisticRegression(max_iter=params["max_iter"], C=params["C"], random_state=seed)
    return make_pipeline(ct, clf).fit(xtr, ytr)


def auc(model, x: pd.DataFrame, y: np.ndarray) -> float:
    return float(roc_auc_score(y, model.predict_proba(x)[:, 1]))


def plot_top_features(top: list[dict[str, Any]], path: Path, test_auc: float) -> None:
    names = [t["feature"] for t in top][::-1]
    vals = [t["gain_share"] for t in top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, vals, color="#3b6ea5")
    ax.set_xlabel("share of total gain")
    ax.set_title(f"Top-{len(top)} features predicting gender (LightGBM test AUC {test_auc:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_leakage_before(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    """Run probes, shuffled control, top-k features; write before.json + PNG; return result."""
    lc = cfg["leakage"]
    seed = lc["seed"]
    seed_everything(seed)
    proc = resolve(cfg["paths"]["smoke_root"]) / "processed" if smoke else None
    out_dir = (
        resolve(cfg["paths"]["smoke_root"]) / "leakage"
        if smoke
        else resolve(cfg["paths"]["leakage_artifacts"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    purpose = "leakage_probe_gender_only"
    xtr, atr = load_split(cfg, "train", with_target=False, processed=proc)
    xva, ava = load_split(cfg, "val", with_target=False, processed=proc)
    xte, ate = load_split(cfg, "test", with_target=False, purpose=purpose, processed=proc)
    assert cfg["target_column"] not in xtr.columns
    cat_cols = [c for c in xtr.columns if c in set(cfg["categorical"])]
    ytr, yva, yte = atr.to_numpy(), ava.to_numpy(), ate.to_numpy()

    gbm = fit_lgbm_probe(xtr, ytr, xva, yva, cat_cols, lc["lightgbm"], seed)
    logit = fit_logistic_probe(xtr, ytr, cat_cols, lc["logistic"], seed)
    rng = np.random.default_rng(seed)  # shuffle labels independently inside each split
    ytr_s, yva_s, yte_s = rng.permutation(ytr), rng.permutation(yva), rng.permutation(yte)
    sh = fit_lgbm_probe(xtr, ytr_s, xva, yva_s, cat_cols, lc["lightgbm"], seed)
    gain = gbm.booster_.feature_importance(importance_type="gain")
    share = gain / gain.sum()
    order = np.argsort(share)[::-1][: lc["top_k"]]
    top = [{"feature": str(xtr.columns[i]), "gain_share": float(share[i])} for i in order]

    result = {
        "stage": "leakage_before",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "seed": seed,
        "smoke": smoke,
        "config": {"leakage": lc, "n_features": int(xtr.shape[1]), "features": list(xtr.columns)},
        "uses_target": False,
        "lightgbm": {
            "val_auc": auc(gbm, xva, yva),
            "test_auc": auc(gbm, xte, yte),
            "best_iteration": int(gbm.best_iteration_ or 0),
        },
        "logistic": {"val_auc": auc(logit, xva, yva), "test_auc": auc(logit, xte, yte)},
        "shuffled_control": {"val_auc": auc(sh, xva, yva_s), "test_auc": auc(sh, xte, yte_s)},
        "top_features": top,
    }
    result["shuffled_control"]["within_tolerance"] = bool(
        abs(result["shuffled_control"]["test_auc"] - 0.5) <= lc["shuffle_tolerance"]
    )
    headline = result["lightgbm"]["test_auc"]
    result["gender_from_features_auc"] = headline
    result["soft_flag_premise_weak"] = bool(headline < lc["premise_auc_floor"])
    if result["soft_flag_premise_weak"]:
        log.warning(
            "PREMISE WEAK: gender-from-features AUC %.3f < %.2f", headline, lc["premise_auc_floor"]
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Timestamped file is immutable (rule 13); before.json is the latest-pointer copy.
    for name in (f"before_{stamp}.json", "before.json"):
        (out_dir / name).write_text(json.dumps(result, indent=2))
    (out_dir / "top15_before.json").write_text(json.dumps(top, indent=2))
    plot_top_features(top, out_dir / "top15_proxy_features_before.png", headline)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--stage", default="before", choices=["before"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = run_leakage_before(load_config(args.config), smoke=args.smoke)
    print(f"LightGBM  val {r['lightgbm']['val_auc']:.4f}  test {r['lightgbm']['test_auc']:.4f}")
    print(f"Logistic  val {r['logistic']['val_auc']:.4f}  test {r['logistic']['test_auc']:.4f}")
    print(
        f"Shuffled  test {r['shuffled_control']['test_auc']:.4f}  "
        f"ok={r['shuffled_control']['within_tolerance']}"
    )
    print("Top-5:", [(t["feature"], round(t["gain_share"], 3)) for t in r["top_features"][:5]])


if __name__ == "__main__":
    main()
