"""Export the resolved experiment settings to reproduce/resolved_settings.json (public, 2026-11-03).

The run folders under artifacts/ are not in git, so the tuned hyperparameters and chosen settings
are copied here from the saved files, with their source paths. Reads saved JSON only.

python scripts/export_resolved_settings.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P2 = "artifacts/runs/20261101T055729Z_phase2"
P4B = "artifacts/runs/20261101T092911Z_phase4b"
PT = "artifacts/runs/20261102T161051Z_posthoc_tuning"


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> None:
    arms = {}
    for d in sorted((ROOT / P2).iterdir()):
        if (d / "tuning.json").exists():
            t = load(f"{P2}/{d.name}/tuning.json")
            arms[d.name] = {
                "best_params": t["best_params"],
                "best_val_auc_pr": t["best_value"],
                "n_trials": len(t["trials"]) if isinstance(t["trials"], list) else t["trials"],
                "source": f"{P2}/{d.name}/tuning.json",
            }
    spec = load("artifacts/bundle/v1/model/model_spec.json")
    man = load("artifacts/bundle/v1/manifest.json")
    jt_t = load(f"{P4B}/jt_tuning.json")
    ref_t = load(f"{PT}/tuning.json")
    out = {
        "note": "Copied from saved run files (paths given). Phase 2 run used: the second run "
        "(20261101T055729Z); the first run 20261101T052529Z is kept but not used.",
        "split": "stratified 70/15/15 on TARGET, seed 42 (configs/data.yaml)",
        "seeds": {"model": [0, 1, 2], "jt_a_extra": [3, 4], "served": man["served_seed"]},
        "tuned_arms_phase2": arms,
        "served_model": {
            "arm": "JT chosen (rule-selected, untuned)",
            "lambdas_f_c_a": spec["lambdas"],
            "hyperparameters": spec["hp"],
            "model_defaults": spec["model_defaults"],
            "risk_bands": man["risk_bands"],
            "selection_rule": "configs/selection_rule.yaml (committed before the sweep)",
        },
        "jt_tuned": {
            "source": f"{P4B}/jt_tuning.json",
            "best_params": jt_t.get("best_params", jt_t),
        },
        "ref_t_posthoc": {
            "source": f"{PT}/tuning.json",
            "best_params": ref_t.get("best_params", ref_t),
        },
        "decision_rule": "flag top 10% (validation 90th-percentile threshold), same for all arms",
    }
    dst = ROOT / "reproduce" / "resolved_settings.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("written", dst, "| arms:", ", ".join(arms))


if __name__ == "__main__":
    main()
