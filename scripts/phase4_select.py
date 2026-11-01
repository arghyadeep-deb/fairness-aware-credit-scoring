"""Apply the committed selection rule ONCE to the combined sweep set (validation only).

python scripts/phase4_select.py <phase4_run_dir> <phase2_run_dir>
Writes sweep_points.csv, selection.json in the run folder. Never edits selection_rule.yaml.
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
    run, p2 = Path(sys.argv[1]), Path(sys.argv[2])
    rule, df = load_rule(), load_points(run, ("jt", "jt_conf"))
    ref = references(p2, rule)
    agg = aggregate(df)
    assert (agg["n_seeds"] == 3).all(), "every sweep point needs 3 seeds"
    agg["pareto"] = pareto_mask(agg)
    agg = agg.merge(
        constraint_table(agg, rule, ref)[
            ["name", "tag", "ok_eod", "ok_auc", "ok_ece", "ok_dir", "all_ok"]
        ]
    )
    agg.to_csv(run / "sweep_points.csv", index=False)
    res = apply_rule(agg, rule, ref)
    ch = res["chosen"]
    out = {
        "git_hash": git_hash(),
        "rule_file": "configs/selection_rule.yaml (unchanged since commit f8189b2)",
        "references": ref,
        "n_points": int(len(agg)),
        "step_used": res["step"],
        "n_feasible": res["n_feasible"],
        "ece_bound": res["ece_bound"],
        "auc_floor": res["auc_floor"],
        "soft_flag": res["soft_flag"],
        "chosen": {k: (v.item() if hasattr(v, "item") else v) for k, v in ch.items()},
    }
    (run / "selection.json").write_text(json.dumps(out, indent=2, default=str))
    pd.set_option("display.width", 200)
    print(json.dumps({k: v for k, v in out.items() if k != "chosen"}, indent=1, default=str))
    print(out["chosen"]["name"], out["chosen"]["tag"])
    base_ok = agg[agg["all_ok"]]
    print("points passing the BASE rule:", base_ok["tag"].tolist(), base_ok["name"].tolist())


if __name__ == "__main__":
    main()
