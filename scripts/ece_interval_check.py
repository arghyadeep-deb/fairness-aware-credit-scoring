"""Synthetic check of bootstrap intervals for paired worst-group ECE differences (2026-11-03).

Declared after an external review: percentile intervals for worst-group ECE differences sometimes
exclude the observed difference. This script measures, on SYNTHETIC data with a known truth, how
often three interval methods cover the true difference and how often they make a wrong directional
call. It uses no project data and changes no reported result; it informs wording, and no method
was chosen for preserving existing conclusions.

Methods (all from the same paired bootstrap, rows resampled jointly for both models):
  percentile  [q2.5, q97.5] of bootstrap differences (the method used in the reports)
  basic       [2*obs - q97.5, 2*obs - q2.5]
  shifted     percentile interval moved by -(mean bootstrap difference - observed difference)

python scripts/ece_interval_check.py   ->  artifacts/tables/ece_interval_coverage.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval.metrics import ece_equal_freq  # noqa: E402

BINS, N, REPS, BOOT, N_TRUE = 15, 15_000, 150, 150, 2_000_000


def ece(y, p):
    """Same as src.eval.metrics.ece_equal_freq (equal-frequency bins), vectorised."""
    order = np.argsort(p, kind="mergesort")
    sizes = np.array([len(c) for c in np.array_split(np.arange(len(p)), BINS)])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    ys, ps = y[order], p[order]
    gy, gp = np.add.reduceat(ys, starts), np.add.reduceat(ps, starts)
    return float(np.sum(np.abs(gy - gp)) / len(p))


def worst(y, p, a):
    return max(ece(y[a == 0], p[a == 0]), ece(y[a == 1], p[a == 1]))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def draw(rng, n, scen):
    a = (rng.random(n) < 0.65).astype(int)
    z = -2.6 + 0.9 * rng.standard_normal(n) + 0.15 * a
    y = (rng.random(n) < sigmoid(z)).astype(float)
    za = z + 0.25 * rng.standard_normal(n)
    zb = z + 0.25 * rng.standard_normal(n)
    if scen == "null: both calibrated":
        pass
    elif scen == "small gap: B over-confident in group 0":
        zb = np.where(a == 0, -2.6 + 1.10 * (zb + 2.6), zb)
    elif scen == "large gap: B shifted in group 0":
        zb = np.where(a == 0, zb + 0.35, zb)
    return y, sigmoid(za), sigmoid(zb), a


SCENARIOS = [
    "null: both calibrated",
    "small gap: B over-confident in group 0",
    "large gap: B shifted in group 0",
]


def main() -> None:
    rng = np.random.default_rng(20261103)
    y, p, _, a = draw(rng, 20_000, SCENARIOS[0])
    assert abs(ece(y, p) - ece_equal_freq(y, p, BINS)) < 1e-9
    rows = []
    for scen in SCENARIOS:
        yt, pa, pb, at = draw(rng, N_TRUE, scen)
        truth = worst(yt, pb, at) - worst(yt, pa, at)  # B minus A at very large n
        cover = {m: 0 for m in ("percentile", "basic", "shifted")}
        wrong = {m: 0 for m in cover}
        excl_obs = 0
        bias = []
        for _ in range(REPS):
            y, pa, pb, a = draw(rng, N, scen)
            obs = worst(y, pb, a) - worst(y, pa, a)
            d = np.empty(BOOT)
            for b in range(BOOT):
                ix = rng.integers(0, N, N)
                d[b] = worst(y[ix], pb[ix], a[ix]) - worst(y[ix], pa[ix], a[ix])
            lo, hi = np.percentile(d, [2.5, 97.5])
            bias.append(d.mean() - obs)
            ivs = {
                "percentile": (lo, hi),
                "basic": (2 * obs - hi, 2 * obs - lo),
                "shifted": (lo - (d.mean() - obs), hi - (d.mean() - obs)),
            }
            excl_obs += not (lo <= obs <= hi)
            for m, (l_, h_) in ivs.items():
                cover[m] += l_ <= truth <= h_
                # wrong directional call: interval excludes 0 on the side opposite to the truth,
                # or excludes 0 when the truth is (close to) zero
                if abs(truth) < 1e-3:
                    wrong[m] += l_ > 0 or h_ < 0
                else:
                    wrong[m] += (l_ > 0 and truth < 0) or (h_ < 0 and truth > 0)
        for m in cover:
            rows.append(
                {
                    "scenario": scen,
                    "true_diff": truth,
                    "method": m,
                    "coverage": cover[m] / REPS,
                    "wrong_direction_or_false_positive": wrong[m] / REPS,
                    "percentile_excludes_observed": excl_obs / REPS,
                    "mean_bootstrap_bias": float(np.mean(bias)),
                    "n": N,
                    "reps": REPS,
                    "boot": BOOT,
                }
            )
        print(
            scen,
            f"truth {truth:+.4f}",
            {m: round(cover[m] / REPS, 3) for m in cover},
            "wrong",
            {m: round(wrong[m] / REPS, 3) for m in wrong},
            f"bias {np.mean(bias):+.4f}",
        )
    out = ROOT / "artifacts" / "tables" / "ece_interval_coverage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("written", out)


if __name__ == "__main__":
    main()
