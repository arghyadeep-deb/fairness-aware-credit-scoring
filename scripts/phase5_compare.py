"""Phase 5 post-hoc comparison (validation only; test set untouched), as declared in RUN_LOG (commit 2ef4769).

python scripts/phase5_compare.py <phase2_dir> <phase4_dir> <phase4b_dir>
Writes <phase4b_dir>/phase5/: table_10pct.csv, paired_diffs.csv, paired_diffs_matched.csv,
table_matched.csv, eg_eps_curve.csv, flagrate_curve.csv, s1_note.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.tensors import load_arrays  # noqa: E402
from src.eval.bootstrap import bootstrap_metrics  # noqa: E402
from src.eval.metrics import METRIC_NAMES  # noqa: E402
from src.eval.sweep_eval import eval_with_ci  # noqa: E402
from src.posthoc.phase5 import (  # noqa: E402
    flags_at_rate,
    gt_fraction_flags,
    metrics_row,
    nearest_eod_fraction,
    pergroup_calibrate,
)
from src.posthoc.prior_correction import prior_shift  # noqa: E402
from src.run_arms import extract_z, match_flag_weight, save_predictions  # noqa: E402
from src.run_joint import load_setup  # noqa: E402
from src.utils import load_config  # noqa: E402

S = [0, 1, 2]
KEYS = ["auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]
EPS_GRID = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
JTL, TUN = "JT chosen = JT-v1 (0.05,2,0)", "JT tuned (0.05,2,0)"


def main() -> None:
    p2, p4, p4b = (Path(x) for x in sys.argv[1:4])
    eg_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None  # corrected S2-EG (Phase 6 step 1)
    out = p4b / (sys.argv[5] if len(sys.argv) > 5 else "phase5")
    out.mkdir(exist_ok=True)
    cfg = load_config("configs/train.yaml")
    fr, eb = cfg["evaluation"]["flag_rate"], cfg["evaluation"]["ece_bins"]
    val, _ = load_arrays(cfg, "val")
    jt = p4 / "jt" / "f0p05_c2_a0"

    # ---- analysis-only per-group Platt / isotonic (cross-fitted inside validation) ----
    for kind in ("platt", "isotonic"):
        for s in S:
            d = out / kind / f"seed_{s}"
            if (d / "val_prob.npy").exists():
                continue
            lg = np.load(p2 / "a2" / f"seed_{s}" / "val_logits.npy")
            prob = pergroup_calibrate(val.y, lg, val.a, kind, seed=s)
            save_predictions(d, prob, flags_at_rate(prob, fr), {"deployable": False, "kind": kind})

    arms = {  # label -> (dir, deployable)
        "A2": (p2 / "a2", True),
        "S1": (p2 / "s1", True),
        "S2-EG": (eg_dir or p2 / "s2_eg", True),
        "S2-GT": (p2 / "s2_gt", False),
        "per-group Platt": (out / "platt", False),
        "per-group isotonic": (out / "isotonic", False),
        "fairlearn ThresholdOptimizer": (p2 / "s2_fairlearn", False),
        JTL: (jt, True),
        TUN: (p4b / "jt_tuned" / "f0p05_c2_a0", True),
        "H1 (f=0.05)+S1": (p4b / "h1", True),
        "H2 (f=0.10)+S1": (p4b / "h2", True),
        "H3 (f=0.25)+S1": (p4b / "h3", True),
    }
    dep = {k: v[1] for k, v in arms.items()}

    # ---- (1) shared 10% flag rate ----
    t10 = eval_with_ci({k: (v[0], S) for k, v in arms.items()}, val)
    t10.insert(1, "deployable", [dep[a] for a in t10.arm])
    t10.to_csv(out / "table_10pct.csv", index=False)
    cols = ["arm", "deployable", "auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]
    print(t10[cols].round(4).to_string(), flush=True)

    # ---- paired bootstrap differences ----
    def boots(base: Path) -> np.ndarray:
        bs = []
        for s in S:
            d = base / f"seed_{s}"
            bs.append(
                bootstrap_metrics(
                    val.y, np.load(d / "val_prob.npy"), val.a, np.load(d / "val_flags.npy"),
                    1000, 0, eb,
                )
            )  # fmt: skip
        return np.nanmean(np.stack(bs), axis=0)

    def diff_rows(pairs, B, tag):
        rows = []
        for a, b in pairs:
            for k in KEYS:
                i = METRIC_NAMES.index(k)
                d = B[a][:, i] - B[b][:, i]
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append(
                    {"setting": tag, "A": a, "B": b, "metric": k, "diff_A_minus_B": d.mean(),
                     "ci_lo": lo, "ci_hi": hi, "significant": bool(lo > 0 or hi < 0)}
                )  # fmt: skip
        return rows

    B = {k: boots(v[0]) for k, v in arms.items()}
    pairs = [(JTL, c) for c in arms if c != JTL] + [(TUN, "S2-EG"), (TUN, JTL)]
    pairs += [(h, "S2-EG") for h in arms if h.startswith("H")]
    pairs = list(dict.fromkeys(pairs))
    pd.DataFrame(diff_rows(pairs, B, "10pct")).to_csv(out / "paired_diffs.csv", index=False)

    # ---- (2) matched fairness: target = EOD of JT chosen ----
    target = float(t10.loc[t10.arm == JTL, "eod"].iloc[0])
    print("matched-fairness target (JT chosen mean val EOD):", round(target, 4), flush=True)
    from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from src.models.nets import build_model

    device, train, _v, cards, weights, _ = load_setup(cfg)
    hp = json.loads((p2 / "a2" / "tuning.json").read_text())["best_params"]
    ec = cfg["exponentiated_gradient"]
    eg_curve = []
    for s in S if eg_dir is None else []:
        model = build_model("tabtransformer", train.cont.shape[1], cards, cfg["model_defaults"], hp)
        model.load_state_dict(torch.load(p2 / "a2" / f"seed_{s}" / "best.pt"))
        model.to(device)
        sc = StandardScaler().fit(extract_z(model, train, device))
        ztr = sc.transform(extract_z(model, train, device))
        zva = sc.transform(extract_z(model, val, device))
        w_pos = match_flag_weight(ztr, train.y, fr, ec["logistic_max_iter"])
        s1p = np.load(p2 / "s1" / f"seed_{s}" / "val_prob.npy")
        for eps in EPS_GRID:
            d = out / f"s2_eg_eps{eps}" / f"seed_{s}"
            if not (d / "val_flags.npy").exists():
                t0 = time.time()
                base = LogisticRegression(
                    max_iter=ec["logistic_max_iter"], class_weight={0: 1.0, 1: w_pos}
                )
                eg = ExponentiatedGradient(
                    base, constraints=EqualizedOdds(), eps=eps, max_iter=ec["max_iter"]
                )
                eg.fit(ztr, train.y, sensitive_features=train.a)
                score = eg._pmf_predict(zva)[:, 1] + 1e-6 * s1p
                save_predictions(d, s1p, flags_at_rate(score, fr), {"eps": eps, "deployable": True})
                print(f"EG eps {eps} seed {s} ({time.time() - t0:.0f}s)", flush=True)
            m = metrics_row(val.y, s1p, val.a, np.load(d / "val_flags.npy"), eb)
            eg_curve.append({"eps": eps, "seed": s, **m})
    if eg_dir is None:
        egc = pd.DataFrame(eg_curve).groupby("eps").mean(numeric_only=True).reset_index()
        egc["abs_gap"] = (egc.eod - target).abs()
        eps_star = float(egc.loc[egc.abs_gap.idxmin(), "eps"])
    else:  # corrected EG: eps curve comes from the step-1 diagnosis (fixed version, shared 10% rule)
        dg = pd.read_csv(eg_dir.parent / "eg_diagnosis.csv")
        dg = dg[dg.version.str.startswith("fixed")]
        egc = dg.groupby("eps").mean(numeric_only=True).reset_index()
        egc = egc.rename(
            columns={
                "shared10_eod": "eod",
                "shared10_recall": "recall",
                "shared10_flag_rate": "flag_rate",
            }
        )
        egc["abs_gap"] = (egc.eod - target).abs()
        eps_star = 0.01
    egc.to_csv(out / "eg_eps_curve.csv", index=False)
    eg_star_dir = eg_dir or out / f"s2_eg_eps{eps_star}"

    gt_dir = out / "s2_gt_matched"
    rate_rows = []
    rates = np.round(np.arange(0.05, 0.2501, 0.01), 2)
    for s in S:
        s1p = np.load(p2 / "s1" / f"seed_{s}" / "val_prob.npy")
        meta = json.loads((p2 / "s2_gt" / f"seed_{s}" / "pred_meta.json").read_text())
        shared = json.loads((p2 / "s1" / f"seed_{s}" / "pred_meta.json").read_text())["threshold"]
        t = nearest_eod_fraction(val.y, s1p, val.a, meta, shared, target, eb)
        save_predictions(
            gt_dir / f"seed_{s}", s1p, gt_fraction_flags(s1p, val.a, meta, shared, t), {"t": t}
        )
        for r in rates:
            m = metrics_row(val.y, s1p, val.a, flags_at_rate(s1p, float(r)), eb)
            rate_rows.append({"seed": s, "rate": float(r), **m})
    rc = pd.DataFrame(rate_rows).groupby("rate").mean(numeric_only=True).reset_index()
    rc["abs_gap"] = (rc.eod - target).abs()
    rc.to_csv(out / "flagrate_curve.csv", index=False)
    rate_star = float(rc.loc[rc.abs_gap.idxmin(), "rate"])
    for s in S:
        for nm in ("a2", "s1"):
            p = np.load(p2 / nm / f"seed_{s}" / "val_prob.npy")
            fl = flags_at_rate(p, rate_star)
            save_predictions(out / f"{nm}_rate{rate_star}" / f"seed_{s}", p, fl, {})

    def mean_row(base: Path, label: str, knob: str, depl: bool) -> dict:
        rs = []
        for s in S:
            d = base / f"seed_{s}"
            rs.append(
                metrics_row(
                    val.y, np.load(d / "val_prob.npy"), val.a, np.load(d / "val_flags.npy"), eb
                )
            )  # fmt: skip
        m = pd.DataFrame(rs).mean()
        return {
            "arm": label, "deployable": depl, "knob": knob,
            **{k: float(m[k]) for k in KEYS + ["flag_rate", "precision"]},
            "eod_gap_to_target": float(m["eod"] - target),
            "matched(|gap|<=0.005)": bool(abs(m["eod"] - target) <= 0.005),
        }  # fmt: skip

    mt = [
        mean_row(p2 / "a2", "A2", "none (10% flag rate)", True),
        mean_row(p2 / "s1", "S1", "none (10% flag rate)", True),
        mean_row(out / f"a2_rate{rate_star}", f"A2 @ flag rate {rate_star:.0%}", "flag rate", True),
        mean_row(out / f"s1_rate{rate_star}", f"S1 @ flag rate {rate_star:.0%}", "flag rate", True),
        mean_row(eg_star_dir, f"S2-EG eps={eps_star}", "EG eps", True),
        mean_row(p2 / "s2_eg", "S2-EG Phase 2 (superseded: base-matched weight)", "EG eps", True),
        mean_row(gt_dir, "S2-GT (threshold fraction t)", "group-threshold fraction", False),
        mean_row(p2 / "s2_gt", "S2-GT (full, EOD minimised)", "none", False),
    ]
    for lab in ("per-group Platt", "per-group isotonic", "fairlearn ThresholdOptimizer"):
        mt.append(mean_row(arms[lab][0], lab, "none (as fitted)", False))
    for lab in [JTL, TUN, "H1 (f=0.05)+S1", "H2 (f=0.10)+S1", "H3 (f=0.25)+S1"]:
        mt.append(mean_row(arms[lab][0], lab, "none (as trained)", True))
    mtab = pd.DataFrame(mt)
    mtab.to_csv(out / "table_matched.csv", index=False)
    print(mtab.round(4).to_string(), flush=True)

    BM = {
        "JT chosen": B[JTL],
        "JT tuned": B[TUN],
        "S2-EG matched": boots(eg_star_dir),
        "S2-GT matched": boots(gt_dir),
        f"S1 @ {rate_star:.0%}": boots(out / f"s1_rate{rate_star}"),
    }
    comp = list(BM)[2:]
    pr = [("JT chosen", c) for c in comp] + [("JT tuned", c) for c in comp]
    pd.DataFrame(diff_rows(pr, BM, "matched")).to_csv(out / "paired_diffs_matched.csv", index=False)

    # ---- (3) why S1 repairs calibration exactly, the learned term only approximately ----
    sp = pd.read_csv(p4 / "sweep_points.csv")
    cal = sp[(sp.name == "jt") & (sp.lambda_f == 0)]
    cal = cal[["lambda_c", "worst_group_ece", "ece", "cal_slope", "cal_intercept"]]
    g = lambda a: float(t10.loc[t10.arm == a, "worst_group_ece"].iloc[0])  # noqa: E731
    note = {
        "prior_shift_logit": prior_shift(weights),
        "class_weights": weights,
        "A2_worst_group_ece": g("A2"),
        "S1_worst_group_ece": g("S1"),
        "JT_worst_group_ece": g(JTL),
        "JT_tuned_worst_group_ece": g(TUN),
        "H_worst_group_ece": {a: g(a) for a in t10.arm if a.startswith("H")},
        "lambda_c_curve_at_lambda_f_0": cal.round(4).to_dict("records"),
        "targets": {"eod_target": target, "eps_star": eps_star, "rate_star": rate_star},
    }
    (out / "s1_note.json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=1)[:1800])


if __name__ == "__main__":
    main()
