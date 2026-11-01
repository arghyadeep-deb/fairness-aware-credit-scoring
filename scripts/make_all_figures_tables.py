"""Phase 8: regenerate EVERY figure and table from the frozen bundle and saved artifacts.

python scripts/make_all_figures_tables.py            (no arguments; no test-set access)
Inputs : artifacts/bundle/v1/json/*.json (fairness, calibration, proxy audit, trade-off, comparison),
         the Phase 6 run folder's report/ CSVs (saved test tables), taiwan/ CSVs, JT-A summary,
         artifacts/leakage/before.json, the Phase 4/5 validation tables.
Outputs: artifacts/figures/*.png, artifacts/tables/*.csv and *.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BUNDLE = ROOT / "artifacts" / "bundle" / "v1"
FIG, TAB = ROOT / "artifacts" / "figures", ROOT / "artifacts" / "tables"
COL = {"jt": "#b03a2e", "post": "#3b6ea5", "base": "#777777", "nn": "#2e7d32"}


def p6_dir() -> Path:
    return ROOT / (ROOT / "artifacts/phase6_rundir.txt").read_text().strip()


def jload(name: str):
    return json.loads((BUNDLE / "json" / f"{name}.json").read_text(encoding="utf-8"))


def md_table(df: pd.DataFrame, cols: list[str], fmt: str = "{:.4f}") -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")  # fmt: skip
    return head + "\n".join(rows) + "\n"


def color_of(arm: str) -> str:
    if arm.startswith("JT") or arm.startswith("H"):
        return COL["jt"]
    if arm.startswith("S") or arm in (
        "per-group Platt",
        "per-group isotonic",
        "ThresholdOptimizer",
    ):
        return COL["post"]
    if arm.startswith("A") or arm in ("G1", "REF", "GI"):
        return COL["base"]
    return COL["nn"]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    p6 = p6_dir()
    cmp_ = pd.DataFrame(jload("comparison")["rows"])
    cmp_.to_csv(TAB / "test_results_all_arms.csv", index=False)
    (TAB / "test_results_all_arms.md").write_text(
        md_table(cmp_.assign(use=np.where(cmp_.deployable, "deployable", "analysis only")),
                 ["arm", "use", "auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]), encoding="utf-8")  # fmt: skip
    # ---- Pareto: deployable arms, EOD vs AUC-PR and EOD vs worst-group ECE (test, 95% CI) ----
    dep = cmp_[cmp_.deployable & ~cmp_.arm.str.startswith("JT-A")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, (y, yl, lg) in zip(ax, [("auc_pr", "AUC-PR (higher better)", False), ("worst_group_ece", "worst-group ECE (lower better)", True)]):  # fmt: skip
        for _, r in dep.iterrows():
            a.errorbar(r.eod, r[y], xerr=[[r.eod - r.eod_lo], [r.eod_hi - r.eod]], yerr=[[r[y] - r[y + "_lo"]], [r[y + "_hi"] - r[y]]],
                       fmt="o", ms=7 if r.arm in ("JT chosen", "S2-EG", "S1", "A2") else 4, color=color_of(r.arm), lw=0.6, alpha=0.9)  # fmt: skip
            a.annotate(r.arm, (r.eod, r[y]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.set_xlabel("EOD (lower better)")
        a.set_ylabel(yl)
        if lg:
            a.set_yscale("log")
    ax[0].set_title(
        "Deployable arms on TEST (95% CI); red = joint/hybrid, blue = post-hoc, grey = baselines",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(FIG / "pareto_deployable_test.png", dpi=160)
    plt.close()
    # ---- reliability diagrams per gender ----
    cal = jload("calibration")
    fig, ax = plt.subplots(figsize=(5, 5))
    mx = (
        max(max(b["mean_pred"], b["frac_pos"]) for g in cal["groups"].values() for b in g["bins"])
        * 1.1
    )
    for g, c in (("M", "#1f5fbf"), ("F", "#a8326f")):
        bins = cal["groups"][g]["bins"]
        ax.plot([b["mean_pred"] for b in bins], [b["frac_pos"] for b in bins], "o-", color=c, label=f"{g} (ECE {cal['groups'][g]['ece']:.4f})")  # fmt: skip
    ax.plot([0, mx], [0, mx], "--", color="#999")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed default rate")
    ax.set_title("Served model (JT chosen): reliability by gender, test")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "reliability_by_gender.png", dpi=160)
    plt.close()
    # ---- leakage before / after ----
    px = jload("proxy_audit")
    la = pd.read_csv(p6 / "report" / "leakage_after.csv")
    la.to_csv(TAB / "leakage_after.csv", index=False)
    labels = ["features (before)", "z: A2 (after)", "z: JT chosen (after)"] + [
        a.replace("JT-A ", "z: ") for a in la.arm if a.startswith("JT-A")
    ]
    vals = [px["before_auc_features"], px["after_auc_z"]["A2"], px["after_auc_z"]["JT"]] + [float(v) for a, v in zip(la.arm, la.probe_lgbm_test_auc) if a.startswith("JT-A")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        range(len(vals)),
        vals,
        color=["#444"] + [COL["base"], COL["jt"]] + ["#d98b84"] * (len(vals) - 3),
    )
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(vals)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("gender-probe ROC-AUC (test)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Gender stays recoverable: before (features) vs after (frozen z)")
    plt.tight_layout()
    plt.savefig(FIG / "leakage_before_after.png", dpi=160)
    plt.close()
    # ---- trade-off sweep ----
    tr = jload("tradeoff")
    pts = pd.DataFrame(tr["points"])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sc = ax.scatter(pts.eod, pts.auc_pr, c=pts.lambda_c, cmap="viridis", s=30)
    fr = pd.DataFrame(tr["frontier"]).sort_values("eod")
    ax.plot(fr.eod, fr.auc_pr, "-", color="#333", lw=0.8, label="non-dominated")
    s = tr["selected"]
    ax.scatter(
        [s["eod"]],
        [s["auc_pr"]],
        marker="D",
        s=90,
        color="#d62728",
        label="rule-selected (0.05, 2, 0)",
    )
    plt.colorbar(sc, label="lambda_c")
    ax.set_xlabel("validation EOD")
    ax.set_ylabel("validation AUC-PR")
    ax.legend()
    ax.set_title("Joint-loss sweep (validation)")
    plt.tight_layout()
    plt.savefig(FIG / "tradeoff_sweep_validation.png", dpi=160)
    plt.close()
    # ---- fairness bars ----
    f = jload("fairness")
    fig, ax = plt.subplots(figsize=(5, 4))
    w = 0.35
    ax.bar(
        np.arange(2) - w / 2,
        [f["per_group"]["M"]["tpr"], f["per_group"]["M"]["fpr"]],
        w,
        label="male",
        color="#1f5fbf",
    )
    ax.bar(
        np.arange(2) + w / 2,
        [f["per_group"]["F"]["tpr"], f["per_group"]["F"]["fpr"]],
        w,
        label="female",
        color="#a8326f",
    )
    ax.set_xticks([0, 1], ["TPR", "FPR"])
    ax.legend()
    ax.set_title(f"Served model, test: EOD {f['eod']:.4f}, DIR {f['dir']:.3f}")
    plt.tight_layout()
    plt.savefig(FIG / "fairness_groups.png", dpi=160)
    plt.close()
    # ---- saved report tables copied for the paper ----
    for name in (
        "val_vs_test.csv",
        "paired_diffs_test.csv",
        "matched_pair.csv",
        "guard_audit.json",
        "shap_proxy.json",
    ):
        src = p6 / "report" / name
        if src.exists():
            (TAB / name).write_bytes(src.read_bytes())
    for name in ("jta_summary.csv",):
        if (p6 / name).exists():
            (TAB / name).write_bytes((p6 / name).read_bytes())
    tw = p6 / "taiwan" / "taiwan_results.csv"
    if tw.exists():
        t = pd.read_csv(tw)
        t.to_csv(TAB / "taiwan_results.csv", index=False)
        tt = t[t.split == "test"]
        (TAB / "taiwan_test.md").write_text(md_table(tt.assign(use=np.where(tt.deployable, "deployable", "analysis only")),
                                                       ["arm", "use", "rate", "auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]), encoding="utf-8")  # fmt: skip
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for a, rn in zip(ax, ("10% flag rate", "train default rate")):
            d = tt[(tt.rate == rn) & tt.deployable]
            for _, r in d.iterrows():
                a.errorbar(r.eod, r.auc_pr, xerr=[[r.eod - r.eod_lo], [r.eod_hi - r.eod]], yerr=[[r.auc_pr - r.auc_pr_lo], [r.auc_pr_hi - r.auc_pr]], fmt="o", color=color_of(r.arm), lw=0.6)  # fmt: skip
                a.annotate(
                    r.arm, (r.eod, r.auc_pr), fontsize=7, xytext=(3, 3), textcoords="offset points"
                )
            a.set_xlabel("EOD")
            a.set_ylabel("AUC-PR")
            a.set_title(f"UCI Taiwan, test, {rn}", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIG / "taiwan_pareto.png", dpi=160)
        plt.close()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    print("tables:", sorted(p.name for p in TAB.glob("*")))


if __name__ == "__main__":
    main()
