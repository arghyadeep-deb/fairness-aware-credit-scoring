"""JT-A secondary analysis (declared in RUN_LOG commit f6d3e3d; validation only; NOT a rule application).

lambda_f = 0.05, lambda_c = 2, spec GRL with adv_steps in {3, 5} and lambda_a in {0.1, 0.25, 0.5};
5 seeds; controls JT (0,0,0) and JT chosen (0.05,2,0) with the same seeds. Each run gets a fresh
LightGBM/logistic z-probe (train z -> val z, Phase 1 settings).

python scripts/phase6b_jta.py <run_dir>     (resumable)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.eval.zprobe import probe_z  # noqa: E402
from src.run_arms import save_json  # noqa: E402
from src.run_joint import load_setup, run_point, tag  # noqa: E402
from src.utils import load_config  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
VARIANTS = [(k, la) for k in (3, 5) for la in (0.1, 0.25, 0.5)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = Path(sys.argv[1])
    cfg = load_config("configs/train.yaml")
    setup = load_setup(cfg)
    device, train, val, cards, weights, hp = setup
    jobs = [("jta_ctl", (0.0, 0.0, 0.0), None), ("jta_ctl", (0.05, 2.0, 0.0), None)]
    jobs += [(f"jta_k{k}", (0.05, 2.0, la), {"adv_steps": k}) for k, la in VARIANTS]
    for name, lam, sched in jobs:
        for s in SEEDS:
            run_point(cfg, out, lam, s, setup, name=name, schedule=sched)
            d = out / name / tag(lam) / f"seed_{s}"
            if not (d / "probe.json").exists():
                state = torch.load(d / "best.pt")
                pr = probe_z(cfg, hp, state, train, val, cards, device, seed=s)
                save_json(d / "probe.json", pr)
                logging.info("%s %s s%d probe %.4f", name, tag(lam), s, pr["z_probe_lgbm_auc"])
    print("done", json.dumps([j[0] + tag(j[1]) for j in jobs]))


if __name__ == "__main__":
    main()
