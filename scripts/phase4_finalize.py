"""Phase 4 final tables (validation only): JT vs baselines with CIs, ablations with z-probes.

python scripts/phase4_finalize.py <phase4_run_dir> <phase2_run_dir>
Writes compare.csv, ablation.csv (metrics, CIs, fresh LightGBM/logistic probe on frozen z).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.tensors import load_arrays  # noqa: E402
from src.eval.sweep_eval import eval_with_ci  # noqa: E402
from src.eval.zprobe import probe_z  # noqa: E402
from src.run_joint import load_setup  # noqa: E402
from src.utils import load_config  # noqa: E402

SEEDS = [0, 1, 2]
ABLATION = {  # label -> (folder name, tag)
    "control (0,0,0) = A2 + stratified sampler": ("jt", "f0_c0_a0"),
    "fairness only (0.05,0,0)": ("jt", "f0p05_c0_a0"),
    "calibration only (0,2,0)": ("jt", "f0_c2_a0"),
    "fairness + calibration (0.05,2,0) = CHOSEN": ("jt", "f0p05_c2_a0"),
    "adversary only, confusion (0,0,1)": ("abl_conf", "f0_c0_a1"),
    "fairness + adversary (0.05,0,1)": ("abl_conf", "f0p05_c0_a1"),
    "calibration + adversary (0,2,1)": ("jt_conf", "f0_c2_a1"),
    "all three (0.05,2,1)": ("jt_conf", "f0p05_c2_a1"),
}


def main() -> None:
    run, p2 = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = load_config("configs/train.yaml")
    val, _ = load_arrays(cfg, "val")
    chosen = json.loads((run / "selection.json").read_text())["chosen"]
    ch_dir = run / chosen["name"] / chosen["tag"]
    comp = {
        "A2 (weighted CE, raw probabilities)": (p2 / "a2", SEEDS),
        "S1 (A2 + prior correction)": (p2 / "s1", SEEDS),
        "S2-EG (deployable)": (p2 / "s2_eg", SEEDS),
        "S2-GT (analysis only, uses gender)": (p2 / "s2_gt", SEEDS),
        f"JT chosen {chosen['tag']}": (ch_dir, SEEDS),
    }
    comp_df = eval_with_ci(comp, val)
    comp_df.to_csv(run / "compare.csv", index=False)
    print(comp_df[["arm", "auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]].round(4))

    abl_df = eval_with_ci({k: (run / n / t, SEEDS) for k, (n, t) in ABLATION.items()}, val)
    device, train, val2, cards, weights, hp = load_setup(cfg)
    probes = []
    for label, (n, t) in ABLATION.items():
        pr = []
        for s in SEEDS:
            state = torch.load(run / n / t / f"seed_{s}" / "best.pt")
            pr.append(probe_z(cfg, hp, state, train, val2, cards, device, seed=s))
        probes.append(
            {
                "arm": label,
                "z_probe_lgbm_auc": float(np.mean([x["z_probe_lgbm_auc"] for x in pr])),
                "z_probe_lgbm_sd": float(np.std([x["z_probe_lgbm_auc"] for x in pr], ddof=1)),
                "z_probe_logistic_auc": float(np.mean([x["z_probe_logistic_auc"] for x in pr])),
            }
        )
        print(probes[-1], flush=True)
    abl_df = abl_df.merge(pd.DataFrame(probes), on="arm")
    abl_df.to_csv(run / "ablation.csv", index=False)
    print(
        abl_df[["arm", "auc_roc", "auc_pr", "worst_group_ece", "eod", "z_probe_lgbm_auc"]].round(4)
    )


if __name__ == "__main__":
    main()
