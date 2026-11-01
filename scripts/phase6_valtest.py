"""Validation vs test comparison for every Phase 6 arm (reads saved tables only; no test access).

python scripts/phase6_valtest.py <p2> <p4b> <p6>
Writes <p6>/report/val_vs_test.csv and val_vs_test.md ("optimism" = validation minus test for
AUC-ROC/AUC-PR (higher is better), test minus validation for EOD/worst-group ECE (lower is better),
so a positive optimism always means validation looked better than test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]
LOWER_BETTER = {"worst_group_ece", "eod"}


def val_tables(p2: Path, p4b: Path, p6: Path) -> dict[str, pd.Series]:
    r2 = pd.read_csv(p2 / "results.csv").set_index("arm")
    r5 = pd.read_csv(p4b / "phase5_eg_v2" / "table_10pct.csv").set_index("arm")
    jta = pd.read_csv(p6 / "jta_summary.csv").set_index("variant")
    out: dict[str, pd.Series] = {}

    def first(df, prefix):
        hit = [i for i in df.index if str(i).startswith(prefix)]
        return df.loc[hit[0]] if hit else None

    for a in ("A0", "A1", "A2", "A3", "A4", "A5", "G1", "S1"):
        s = r2.loc[a] if a in r2.index else first(r2, a)
        if s is not None:
            out[a] = s
    out["A1b"] = first(r2, "A1b")
    out["REF"] = first(r2, "REF")
    out["GI"] = first(r2, "GI")
    out["S2-EG"] = r5.loc["S2-EG"]
    out["S2-GT"] = r5.loc["S2-GT"]
    out["per-group Platt"] = r5.loc["per-group Platt"]
    out["per-group isotonic"] = r5.loc["per-group isotonic"]
    out["ThresholdOptimizer"] = r5.loc["fairlearn ThresholdOptimizer"]
    out["JT chosen"] = first(r5, "JT chosen")
    out["JT tuned"] = first(r5, "JT tuned")
    out["H1"], out["H2"], out["H3"] = first(r5, "H1"), first(r5, "H2"), first(r5, "H3")
    for k in (3, 5):
        for la in (0.1, 0.25, 0.5):
            row = jta.loc[f"jta_k{k} f0p05_c2_a{str(la).replace('.', 'p')}"]
            out[f"JT-A k{k} la{la}"] = row
    return {k: v for k, v in out.items() if v is not None}


def main() -> None:
    p2, p4b, p6 = (Path(x) for x in sys.argv[1:4])
    test = pd.read_csv(p6 / "report" / "test_table.csv").set_index("arm")
    val = val_tables(p2, p4b, p6)
    rows = []
    for arm in test.index:
        if arm not in val:
            rows.append(
                {
                    "arm": arm,
                    "deployable": bool(test.loc[arm, "deployable"]),
                    "note": "no validation row",
                }
            )
            continue
        v = val[arm]
        row = {"arm": arm, "deployable": bool(test.loc[arm, "deployable"])}
        for k in KEYS:
            vv = float(v[k]) if k in v.index else np.nan
            tt = float(test.loc[arm, k])
            row[f"val_{k}"], row[f"test_{k}"] = vv, tt
            row[f"optimism_{k}"] = (tt - vv) if k in LOWER_BETTER else (vv - tt)
            if k == "dir":
                row[f"optimism_{k}"] = vv - tt
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(p6 / "report" / "val_vs_test.csv", index=False)
    pd.set_option("display.width", 250)
    show = df[
        [
            "arm",
            "val_auc_pr",
            "test_auc_pr",
            "val_eod",
            "test_eod",
            "val_worst_group_ece",
            "test_worst_group_ece",
        ]
    ]
    print(show.round(4).to_string())
    lines = [
        "| arm | val EOD | test EOD | val AUC-PR | test AUC-PR | val wECE | test wECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        if "val_eod" in r and not pd.isna(r.get("val_eod")):
            lines.append(f"| {r.arm} | {r.val_eod:.4f} | {r.test_eod:.4f} | {r.val_auc_pr:.4f} | {r.test_auc_pr:.4f} | {r.val_worst_group_ece:.4f} | {r.test_worst_group_ece:.4f} |")  # fmt: skip
    (p6 / "report" / "val_vs_test.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
