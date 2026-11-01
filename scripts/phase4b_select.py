"""Phase 4b-0: SECOND application of the committed rule to the union of all JT points (validation only).

python scripts/phase4b_select.py <phase4_run_dir> <phase2_run_dir> <out_dir>
Never edits configs/selection_rule.yaml. Writes sweep_points_union.csv, selection_union.json to out_dir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.eval.sweep_eval import (  # noqa: E402
    aggregate,
    apply_rule,
    constraint_table,
    load_points,
    load_rule,
    pareto_mask,
    references,
)
from src.utils import git_hash  # noqa: E402


def main() -> None:
    run, p2, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    rule, df = load_rule(), load_points(run, ("jt", "jt_conf", "jt_fine"))
    ref = references(p2, rule)
    agg = aggregate(df)
    assert (agg["n_seeds"] == 3).all(), "every point needs 3 seeds"
    agg["pareto"] = pareto_mask(agg)
    agg = agg.merge(
        constraint_table(agg, rule, ref)[
            ["name", "tag", "ok_eod", "ok_auc", "ok_ece", "ok_dir", "all_ok"]
        ]
    )
    agg.to_csv(out_dir / "sweep_points_union.csv", index=False)
    res = apply_rule(agg, rule, ref)
    ch = res["chosen"]
    out = {
        "git_hash": git_hash(),
        "rule_file": "configs/selection_rule.yaml (unchanged since commit f8189b2)",
        "application": "second (union of main 20 + confusion 6 + fine 8 JT points)",
        "references": ref,
        "n_points": int(len(agg)),
        "step_used": res["step"],
        "n_feasible": res["n_feasible"],
        "ece_bound": res["ece_bound"],
        "auc_floor": res["auc_floor"],
        "soft_flag": res["soft_flag"],
        "chosen": {k: (v.item() if hasattr(v, "item") else v) for k, v in ch.items()},
    }
    (out_dir / "selection_union.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "chosen"}, indent=1, default=str))
    print("CHOSEN", out["chosen"]["name"], out["chosen"]["tag"])
    ok = agg[agg["all_ok"]]
    pd.set_option("display.width", 200)
    print(ok[["name", "tag", "auc_roc", "auc_pr", "eod", "worst_group_ece", "dir"]].round(4))
    fine = agg[agg.name == "jt_fine"]
    print(
        fine[
            [
                "tag",
                "auc_roc",
                "auc_pr",
                "eod",
                "worst_group_ece",
                "worst_group_ece_sd",
                "dir",
                "all_ok",
            ]
        ].round(4)
    )


if __name__ == "__main__":
    main()
