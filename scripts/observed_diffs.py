"""Observed paired differences beside the saved bootstrap-mean differences (review fix, 2026-11-03).

The original difference tables used the MEAN of the bootstrap differences as the point estimate.
For nonlinear metrics (EOD, DIR, ECE) that is not the observed difference. This script writes,
beside each original table, a copy with `diff_observed` = (seed-averaged metric of A) minus
(seed-averaged metric of B), keeping the bootstrap-mean column and the percentile CI unchanged.
Originals are not overwritten.

Reads saved files only: Phase 6 per-seed metrics.json (no test loader), saved validation tables,
and saved validation probabilities/flags plus the validation split for the three-way references.

python scripts/observed_diffs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T = ROOT / "artifacts" / "tables"
SEEDS = [0, 1, 2]


def test_table(p6: Path) -> pd.DataFrame:
    per = {}
    for d in (p6 / "test").iterdir():
        if (d / "done.json").exists():
            arm = json.loads((d / "done.json").read_text())["arm"]
            ss = sorted(int(x.name.split("_")[1]) for x in d.glob("seed_*"))
            per[arm] = [
                json.loads((d / f"seed_{s}" / "metrics.json").read_text())["metrics"] for s in ss
            ]
    df = pd.read_csv(T / "paired_diffs_test.csv")
    obs = []
    for _, r in df.iterrows():
        n = min(len(per[r.A]), len(per[r.B]))
        a = np.nanmean([m[r.metric] for m in per[r.A][:n]])
        b = np.nanmean([m[r.metric] for m in per[r.B][:n]])
        obs.append(float(a - b))
    df = df.rename(columns={"diff_A_minus_B": "diff_bootstrap_mean"})
    df.insert(3, "diff_observed", obs)
    return df


def diag_table() -> pd.DataFrame:
    tab = pd.read_csv(T / "diag_validation_table.csv").set_index("arm")
    df = pd.read_csv(T / "diag_paired_diffs.csv").rename(columns={"diff": "diff_bootstrap_mean"})
    df.insert(
        3,
        "diff_observed",
        [float(tab.at[r.arm, r.metric] - tab.at[r.minus, r.metric]) for _, r in df.iterrows()],
    )
    return df


def three_way_table(p6: Path) -> pd.DataFrame:
    from src.data.tensors import load_arrays
    from src.eval.metrics import compute_metrics
    from src.run_arms import flags_from_shared_rule
    from src.utils import load_config

    cfg = load_config("configs/train.yaml")
    fr, eb = cfg["evaluation"]["flag_rate"], cfg["evaluation"]["ece_bins"]
    val, _ = load_arrays(cfg, "val")
    ref = {}
    pj = {s: np.load(p6 / "jta_ctl" / "f0p05_c2_a0" / f"seed_{s}" / "val_prob.npy") for s in SEEDS}
    ms = [
        compute_metrics(val.y, pj[s], val.a, flags_from_shared_rule(pj[s], fr)[0], eb)
        for s in SEEDS
    ]
    ref["JT chosen"] = {k: np.nanmean([m[k] for m in ms]) for k in ms[0]}
    d = p6 / "s2_eg_v2"
    ms = [
        compute_metrics(
            val.y,
            np.load(d / f"seed_{s}" / "val_prob.npy"),
            val.a,
            np.load(d / f"seed_{s}" / "val_flags.npy").astype(bool),
            eb,
        )
        for s in SEEDS
    ]
    ref["S2-EG"] = {k: np.nanmean([m[k] for m in ms]) for k in ms[0]}
    tab = pd.read_csv(T / "three_way_table.csv").set_index(
        ["grid", "lambda_f", "lambda_c", "lambda_a"]
    )
    df = pd.read_csv(T / "three_way_diffs.csv").rename(columns={"diff": "diff_bootstrap_mean"})
    obs = [
        float(
            tab.at[(r.grid, r.lambda_f, r.lambda_c, r.lambda_a), r.metric + "_mean"]
            - ref[r.minus][r.metric]
        )
        for _, r in df.iterrows()
    ]
    df.insert(6, "diff_observed", obs)
    return df


def main() -> None:
    p6 = ROOT / Path((ROOT / "artifacts" / "phase6_rundir.txt").read_text().strip())
    for name, df in (
        ("paired_diffs_test", test_table(p6)),
        ("diag_paired_diffs", diag_table()),
        ("three_way_diffs", three_way_table(p6)),
    ):
        lo = "ci_lo" if "ci_lo" in df else "lo"
        hi = "ci_hi" if "ci_hi" in df else "hi"
        df["observed_outside_ci"] = (df["diff_observed"] < df[lo]) | (df["diff_observed"] > df[hi])
        df.to_csv(T / f"{name}_observed.csv", index=False)
        gap = (df["diff_observed"] - df["diff_bootstrap_mean"]).abs()
        print(
            f"{name}: {len(df)} rows, max |observed - bootstrap mean| {gap.max():.4f}, "
            f"observed outside its CI: {int(df['observed_outside_ci'].sum())}"
        )


if __name__ == "__main__":
    main()
