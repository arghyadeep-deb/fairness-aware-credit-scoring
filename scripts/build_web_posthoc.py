"""Write web/data/posthoc_arms.json (REF-EG row for the dashboard) from the saved post-hoc test table.

The frozen bundle's comparison.json does not contain REF-EG and must not change, so the frontend reads this
separate, labelled file. No test access: reads artifacts/tables/posthoc_ref_test_table.csv only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["auc_roc", "auc_pr", "worst_group_ece", "eod"]


def main() -> None:
    t = pd.read_csv(ROOT / "artifacts" / "tables" / "posthoc_ref_test_table.csv").set_index("arm")
    r = t.loc["REF-EG"]
    row = {"arm": "REF-EG", "deployable": True, "posthoc": True, "n_seeds": 3, "dir": float(r["dir"]), "recall": float(r["recall"])}
    for k in KEYS:
        row |= {k: float(r[k]), k + "_lo": float(r[k + "_lo"]), k + "_hi": float(r[k + "_hi"])}
    out = ROOT / "web" / "data" / "posthoc_arms.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"label": "added after the original test run, post-hoc", "source": "artifacts/tables/posthoc_ref_test_table.csv", "rows": [row]}, indent=1), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
