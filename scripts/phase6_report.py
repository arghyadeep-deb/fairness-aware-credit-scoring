"""Phase 6 report tables from the saved per-arm test outputs (NO test read here).

python scripts/phase6_report.py <p2> <p6>
Writes <p6>/report/: test_table.csv, paired_diffs_test.csv, leakage_after.csv, shap_proxy.json,
matched_pair.csv, guard_audit.json, retrain_check.csv, pareto_deployable.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.tensors import load_arrays  # noqa: E402
from src.eval.bootstrap import pooled_ci  # noqa: E402
from src.eval.final import new_proxy_check  # noqa: E402
from src.eval.metrics import METRIC_NAMES  # noqa: E402
from src.utils import load_config  # noqa: E402

KEYS = ["auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]
JTA = [f"JT-A k{k} la{la}" for k in (3, 5) for la in (0.1, 0.25, 0.5)]


def arm_dirs(p6: Path) -> dict[str, Path]:
    return {d.name.replace("_", " "): d for d in sorted((p6 / "test").iterdir()) if (d / "done.json").exists()}  # fmt: skip


def main() -> None:
    p2, p6 = Path(sys.argv[1]), Path(sys.argv[2])
    out = p6 / "report"
    out.mkdir(exist_ok=True)
    dirs = arm_dirs(p6)
    # names with spaces were stored with underscores; map back through done.json
    dirs = {json.loads((d / "done.json").read_text())["arm"]: d for d in dirs.values()}
    dep = {a: json.loads((d / "done.json").read_text())["deployable"] for a, d in dirs.items()}

    def seeds_of(d):
        return sorted(int(x.name.split("_")[1]) for x in d.glob("seed_*"))

    rows, B, M = [], {}, {}
    for a, d in dirs.items():
        ss = seeds_of(d)
        ms = [json.loads((d / f"seed_{s}" / "metrics.json").read_text()) for s in ss]
        boots = [np.load(d / f"seed_{s}" / "boot.npy") for s in ss]
        lo, hi = pooled_ci(boots)
        B[a], M[a] = boots, ms
        row = {"arm": a, "deployable": dep[a], "n_seeds": len(ss)}
        for i, k in enumerate(METRIC_NAMES):
            vals = [m["metrics"][k] for m in ms]
            row[k], row[k + "_lo"], row[k + "_hi"] = (
                float(np.nanmean(vals)),
                float(lo[i]),
                float(hi[i]),
            )
            row[k + "_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
        row["test_flag_rate"] = float(np.mean([m["test_flag_rate"] for m in ms]))
        rows.append(row)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "test_table.csv", index=False)

    # ---- paired differences on test (first min(n) seeds, shared bootstrap indices) ----
    def pooled(a, n):
        return np.nanmean(np.stack(B[a][:n]), axis=0)

    def diff(a, b):
        n = min(len(B[a]), len(B[b]))
        pa, pb = pooled(a, n), pooled(b, n)
        res = []
        for k in KEYS:
            i = METRIC_NAMES.index(k)
            d = pa[:, i] - pb[:, i]
            lo, hi = np.percentile(d, [2.5, 97.5])
            res.append({"A": a, "B": b, "metric": k, "diff_A_minus_B": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "significant": bool(lo > 0 or hi < 0)})  # fmt: skip
        return res

    pairs = [("JT chosen", b) for b in dirs if b != "JT chosen"]
    pairs += [(h, "S2-EG") for h in ("H1", "H2", "H3") if h in dirs]
    pairs += [("JT tuned", "S2-EG")]
    pairs += [(j, "JT chosen") for j in JTA if j in dirs]
    pd.DataFrame([r for a, b in dict.fromkeys(pairs) if a in dirs and b in dirs for r in diff(a, b)]).to_csv(out / "paired_diffs_test.csv", index=False)  # fmt: skip

    # ---- leakage after ----
    before = json.loads((ROOT / "artifacts/leakage/before.json").read_text())[
        "gender_from_features_auc"
    ]
    lk = []
    for a in ["A2", "JT chosen", "JT tuned"] + JTA:
        if a in dirs:
            z = [m["z_probe_after"] for m in M[a]]
            lk.append({"arm": a, "n_seeds": len(z),
                       "probe_lgbm_test_auc": np.mean([x["probe_lgbm_test_auc"] for x in z]),
                       "probe_lgbm_test_sd": np.std([x["probe_lgbm_test_auc"] for x in z], ddof=1) if len(z) > 1 else np.nan,
                       "probe_logistic_test_auc": np.mean([x["probe_logistic_test_auc"] for x in z]),
                       "probe_lgbm_val_auc": np.mean([x["probe_lgbm_val_auc"] for x in z]),
                       "shap_probe_test_auc_seed0": M[a][0].get("shap_probe_test_auc"),
                       "shap_method": M[a][0].get("shap_method"),
                       "before_gender_from_X_test_auc": before})  # fmt: skip
    pd.DataFrame(lk).to_csv(out / "leakage_after.csv", index=False)

    # ---- matched pair ----
    mp = [{"arm": a, **M[a][0]["matched_pair"]} for a in dirs if "matched_pair" in M[a][0]]
    pd.DataFrame(mp).to_csv(out / "matched_pair.csv", index=False)

    # ---- new-proxy check (A2 vs JT chosen / JT tuned) on stored SHAP vectors; features from train/val only ----
    cfg = load_config("configs/train.yaml")
    train, _ = load_arrays(cfg, "train")
    val, _ = load_arrays(cfg, "val")
    names = list(train.cont_names) + list(train.cat_names)
    g_auc = {}
    for j, n in enumerate(train.cont_names):
        au = roc_auc_score(val.a, val.cont[:, j]) if len(np.unique(val.cont[:, j])) > 1 else 0.5
        g_auc[n] = float(max(au, 1 - au))
    for j, n in enumerate(train.cat_names):
        enc = pd.Series(train.a).groupby(train.cat[:, j]).mean()
        v = pd.Series(val.cat[:, j]).map(enc).fillna(float(train.a.mean())).to_numpy()
        g_auc[n] = float(max(roc_auc_score(val.a, v), 1 - roc_auc_score(val.a, v)))
    top15_before = [
        t["feature"] for t in json.loads((ROOT / "artifacts/leakage/top15_before.json").read_text())
    ]
    sh = {"before_top15_proxy_features": top15_before}
    sv = lambda a: np.load(dirs[a] / "seed_0" / "shap_test.npy")  # noqa: E731
    for b in ["JT chosen", "JT tuned"] + [j for j in JTA if j in dirs]:
        if b in dirs and "A2" in dirs:
            r = new_proxy_check(sv("A2"), sv(b), names, g_auc)
            r["top_B_in_before_top15"] = [n for n in r["top_B"] if n in top15_before]
            sh[b] = r
    (out / "shap_proxy.json").write_text(json.dumps(sh, indent=1))

    # ---- guard audit ----
    lines = [
        json.loads(x)
        for x in (ROOT / "artifacts/guard/test_access.jsonl").read_text().strip().splitlines()
    ]
    purposes = [x["purpose"] for x in lines]
    p6arms = [p[len("phase6:") :] for p in purposes if p.startswith("phase6:")]
    audit = {"n_entries": len(lines), "phase1_whitelisted": purposes.count("leakage_probe_gender_only"),
             "phase6_entries": len(p6arms), "arms_evaluated": len(dirs),
             "one_entry_per_arm": sorted(p6arms) == sorted(dirs) and len(set(p6arms)) == len(p6arms),
             "missing": sorted(set(dirs) - set(p6arms)), "extra": sorted(set(p6arms) - set(dirs))}  # fmt: skip
    (out / "guard_audit.json").write_text(json.dumps(audit, indent=1))

    # ---- retrain check for REF and GI (validation numbers vs Phase 2) ----
    rc = [{"arm": a, "seed": s, "max_abs_val_prob_diff": M[a][i].get("retrain_max_abs_val_diff")} for a in ("REF", "GI") if a in dirs for i, s in enumerate(seeds_of(dirs[a]))]  # fmt: skip
    pd.DataFrame(rc).to_csv(out / "retrain_check.csv", index=False)

    if "--skip-figures" not in sys.argv:
        from src.eval.phase6_figures import plot_deployable

        plot_deployable(tbl, out)
    print(
        tbl[
            [
                "arm",
                "deployable",
                "n_seeds",
                "auc_roc",
                "auc_pr",
                "worst_group_ece",
                "eod",
                "dir",
                "recall",
                "test_flag_rate",
            ]
        ]
        .round(4)
        .to_string()
    )
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
