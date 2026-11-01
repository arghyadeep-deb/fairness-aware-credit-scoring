"""Evaluate every arm in a run folder on VALIDATION: point metrics, bootstrap CIs, results CSV.

python -m src.eval.report --run-dir artifacts/runs/<ts>_phase2
Mean over seeds with a 95% percentile CI from paired stratified bootstrap replicates.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.tensors import load_arrays
from src.eval.bootstrap import bootstrap_metrics, pooled_ci
from src.eval.metrics import METRIC_NAMES, compute_metrics
from src.utils import load_config

log = logging.getLogger(__name__)
ORDER = [
    "a0",
    "a1",
    "a1b",
    "a2",
    "a3",
    "a4",
    "a5",
    "s1",
    "s2_gt",
    "s2_eg",
    "s2_fairlearn",
    "g1",
    "ref",
    "gi",
]
LABEL = {
    "s2_gt": "S2-GT (analysis only: uses gender, not deployable, val-optimistic)",
    "s2_eg": "S2-EG (fairlearn ExponentiatedGradient, deployable)",
    "a1b": "A1b (SMOTE-NC ratio 1.0)",
    "gi": "GI (gender included, NOT deployable)",
    "ref": "REF (LightGBM)",
    "s2_fairlearn": "S2 fairlearn native (not rate-matched)",
}


def evaluate_run(
    run_dir: Path, cfg: dict[str, Any], smoke: bool = False, n_boot: int | None = None
) -> pd.DataFrame:
    val, _ = load_arrays(cfg, "val", smoke=smoke)
    ev = cfg["evaluation"]
    n_boot = n_boot or ev["bootstrap"]["n"]
    rows, long = [], {}
    arms = [a for a in ORDER if (run_dir / a).is_dir()]
    for arm in arms:
        seeds = sorted(
            int(p.name.split("_")[1])
            for p in (run_dir / arm).glob("seed_*")
            if (p / "val_prob.npy").exists()
        )
        point, boots, extra = [], [], []
        for s in seeds:
            d = run_dir / arm / f"seed_{s}"
            prob, flags = np.load(d / "val_prob.npy"), np.load(d / "val_flags.npy")
            m = compute_metrics(val.y, prob, val.a, flags, ev["ece_bins"])
            # diagnostic only: behaviour at the naive 0.5 cut-off (not the shared decision rule)
            m_naive = compute_metrics(val.y, prob, val.a, prob > 0.5, ev["ece_bins"])
            extra.append(
                {"recall_at_0.5": m_naive["recall"], "flag_rate_at_0.5": m_naive["flag_rate"]}
            )
            point.append([m[k] for k in METRIC_NAMES])
            boots.append(
                bootstrap_metrics(
                    val.y, prob, val.a, flags, n_boot, ev["bootstrap"]["seed"], ev["ece_bins"]
                )
            )
        mean = np.nanmean(np.array(point), axis=0)
        lo, hi = pooled_ci(boots)
        row = {"arm": LABEL.get(arm, arm.upper()), "key": arm, "n_seeds": len(seeds)}
        for i, k in enumerate(METRIC_NAMES):
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = mean[i], lo[i], hi[i]
            row[f"{k}_seed_sd"] = float(np.nanstd(np.array(point)[:, i]))
        for k in ("recall_at_0.5", "flag_rate_at_0.5"):
            row[k] = float(np.mean([e[k] for e in extra]))
        rows.append(row)
        long[arm] = {"seeds": seeds, "per_seed": point}
        log.info("evaluated %s (%d seeds)", arm, len(seeds))
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "results.csv", index=False)
    (run_dir / "results_per_seed.json").write_text(
        json.dumps({"metrics": METRIC_NAMES, "arms": long})
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--boot", type=int, default=None)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = evaluate_run(Path(a.run_dir), load_config(a.config), a.smoke, a.boot)
    cols = ["key", "n_seeds", "auc_roc", "auc_pr", "ece", "worst_group_ece", "dir", "eod", "recall"]
    print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
