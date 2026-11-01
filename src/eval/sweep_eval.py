"""Sweep aggregation, Pareto frontier and the committed operating-point rule (validation only).

The rule is read from configs/selection_rule.yaml and applied exactly as written; this module
never edits it. Reference values (A2 AUC-ROC, S1 worst-group ECE) come from the Phase 2 run's
results.csv at selection time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.eval.bootstrap import bootstrap_metrics, pooled_ci
from src.eval.metrics import METRIC_NAMES, compute_metrics
from src.utils import ROOT

M = ["auc_roc", "auc_pr", "ks", "brier", "ece", "worst_group_ece", "dpd", "eod", "dir",
     "recall", "cal_slope", "cal_intercept"]  # fmt: skip


def load_points(run_dir: Path, names: tuple[str, ...] = ("jt", "jt_conf")) -> pd.DataFrame:
    """One row per (name, point, seed) from the saved per-run summaries."""
    rows = []
    for name in names:
        for f in sorted((run_dir / name).glob("*/seed_*/summary.json")):
            s = json.loads(f.read_text())
            lam = s["lambdas"]
            rows.append(
                {
                    "name": name,
                    "tag": f.parent.parent.name,
                    "lambda_f": lam["f"],
                    "lambda_c": lam["c"],
                    "lambda_a": lam["a"],
                    "seed": s["seed"],
                    "below_auc_floor": s["below_auc_floor"],
                    "best_epoch": s["best_epoch"],
                    **{k: s[f"val_{k}"] for k in M},
                }
            )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per point, plus the across-seed SD of the key metrics."""
    keys = ["name", "tag", "lambda_f", "lambda_c", "lambda_a"]
    g = df.groupby(keys)
    out = g[M].mean()
    for k in ("auc_roc", "auc_pr", "eod", "worst_group_ece"):
        out[f"{k}_sd"] = g[k].std(ddof=1)
    out["n_seeds"] = g["seed"].nunique()
    out["below_auc_floor"] = g["below_auc_floor"].any()
    return out.reset_index()


def pareto_mask(agg: pd.DataFrame, up=("auc_roc",), down=("eod", "worst_group_ece")) -> np.ndarray:
    """True for non-dominated points: no other point is >= on `up`, <= on `down`, better on one."""
    v = np.column_stack([agg[list(up)].to_numpy(), -agg[list(down)].to_numpy()])
    keep = np.ones(len(v), dtype=bool)
    for i in range(len(v)):
        dom = np.all(v >= v[i], axis=1) & np.any(v > v[i], axis=1)
        keep[i] = not dom.any()
    return keep


def load_rule() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs" / "selection_rule.yaml").read_text())


def references(phase2_dir: Path, rule: dict[str, Any]) -> dict[str, float]:
    res = pd.read_csv(phase2_dir / "results.csv").set_index("key")
    b = rule["baselines"]
    return {
        "a2_auc_roc": float(res.loc[b["auc_roc_reference_arm"], "auc_roc"]),
        "s1_worst_group_ece": float(res.loc[b["ece_reference_arm"], "worst_group_ece"]),
        "a2_eod": float(res.loc["a2", "eod"]),
    }


def feasible(agg: pd.DataFrame, rule: dict[str, Any], ref: dict[str, float], mult: float,
             margin: float) -> pd.Series:  # fmt: skip
    c = rule["constraints"]
    ece_bound = max(mult * ref["s1_worst_group_ece"], c["worst_group_ece"]["floor"])
    return (
        (agg["eod"] <= c["eod_max"])
        & (agg["dir"] >= c["dir_min"])
        & (agg["auc_roc"] >= ref["a2_auc_roc"] - margin)
        & (agg["worst_group_ece"] <= ece_bound)
        & (agg["auc_roc"] >= rule["min_auc_roc_for_frontier"])
    )


def apply_rule(agg: pd.DataFrame, rule: dict[str, Any], ref: dict[str, float]) -> dict[str, Any]:
    """Apply the rule exactly as committed: base constraints, then each relaxation in order."""
    c = rule["constraints"]
    mult, margin = c["worst_group_ece"]["multiplier"], c["auc_roc_margin"]
    steps = [("base rule", mult, margin)]
    for r in rule["relaxation_order"]:
        if r["constraint"] == "worst_group_ece.multiplier":
            mult = r["to"]
        elif r["constraint"] == "auc_roc_margin":
            margin = r["to"]
        steps.append((f"relax {r['constraint']} {r['from']} -> {r['to']}", mult, margin))
    for label, m, g in steps:
        ok = feasible(agg, rule, ref, m, g)
        if ok.any():
            best = agg[ok].sort_values(rule["objective"], ascending=False).iloc[0]
            return {"step": label, "n_feasible": int(ok.sum()), "chosen": best,
                    "ece_bound": max(m * ref["s1_worst_group_ece"], c["worst_group_ece"]["floor"]),
                    "auc_floor": ref["a2_auc_roc"] - g, "soft_flag": False}  # fmt: skip
    fb = agg[(agg["dir"] >= rule["fallback"]["dir_min"]) & (agg["auc_roc"] >= 0.60)]
    best = fb.sort_values(rule["objective"], ascending=False).iloc[0]
    return {"step": "fallback (soft flag)", "n_feasible": 0, "chosen": best, "ece_bound": None,
            "auc_floor": None, "soft_flag": True}  # fmt: skip


def constraint_table(
    agg: pd.DataFrame, rule: dict[str, Any], ref: dict[str, float]
) -> pd.DataFrame:
    """Per-point pass/fail of each base constraint (for the notebook)."""
    c = rule["constraints"]
    ece_bound = max(c["worst_group_ece"]["multiplier"] * ref["s1_worst_group_ece"],
                    c["worst_group_ece"]["floor"])  # fmt: skip
    t = agg[["name", "tag", "auc_roc", "auc_pr", "eod", "worst_group_ece", "dir"]].copy()
    t["ok_eod"] = agg["eod"] <= c["eod_max"]
    t["ok_auc"] = agg["auc_roc"] >= ref["a2_auc_roc"] - c["auc_roc_margin"]
    t["ok_ece"] = agg["worst_group_ece"] <= ece_bound
    t["ok_dir"] = agg["dir"] >= c["dir_min"]
    t["all_ok"] = t[["ok_eod", "ok_auc", "ok_ece", "ok_dir"]].all(axis=1)
    return t


def eval_with_ci(
    dirs: dict[str, tuple[Path, list[int]]], val, ece_bins: int = 15, n_boot: int = 1000,
    boot_seed: int = 0,
) -> pd.DataFrame:  # fmt: skip
    """Metrics + paired-bootstrap 95% CI of the across-seed mean for saved (prob, flags) folders."""
    rows = []
    for label, (base, seeds) in dirs.items():
        point, boots = [], []
        for s in seeds:
            d = base / f"seed_{s}"
            prob, flags = np.load(d / "val_prob.npy"), np.load(d / "val_flags.npy")
            m = compute_metrics(val.y, prob, val.a, flags, ece_bins)
            point.append([m[k] for k in METRIC_NAMES])
            boots.append(bootstrap_metrics(val.y, prob, val.a, flags, n_boot, boot_seed, ece_bins))
        mean, (lo, hi) = np.nanmean(np.array(point), axis=0), pooled_ci(boots)
        row = {"arm": label, "n_seeds": len(seeds)}
        for i, k in enumerate(METRIC_NAMES):
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = mean[i], lo[i], hi[i]
        rows.append(row)
    return pd.DataFrame(rows)
