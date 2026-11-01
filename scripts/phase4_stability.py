"""Phase 4 stability diagnostics: full 3+5 schedule, per-epoch gender-head / z-probe / AUC trends.

python scripts/phase4_stability.py <name> f,c,a [f,c,a ...] [--adv-pretrain]   (val only, seed 0)
Writes artifacts/smoke/joint_stability/<name>.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_arms import save_json  # noqa: E402
from src.run_joint import load_setup  # noqa: E402
from src.train_joint import fit_joint  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    name = sys.argv[1]
    pre = "--adv-pretrain" in sys.argv
    lams = [tuple(float(x) for x in s.split(",")) for s in sys.argv[2:] if not s.startswith("--")]
    cfg = load_config("configs/train.yaml")
    device, train, val, cards, weights, hp = load_setup(cfg)
    out = {}
    for lam in lams:
        r = fit_joint(
            cfg,
            hp,
            train,
            val,
            cards,
            weights,
            0,
            device,
            lam,
            max_epochs=14,
            probe=True,
            schedule={"adv_pretrain": pre} if pre else None,
        )
        out[str(lam)] = {"history": r["history"], "norm_reference": r["norm_reference"]}
        print(f"--- lambdas (f,c,a) = {lam}; norm ref {r['norm_reference']}")
        print("ep ramp  AUC-ROC AUC-PR   EOD   wECE  advAUC zProbe")
        for h in r["history"]:
            print(
                f"{h['epoch']:2d} {h['ramp']:.2f}  {h['val_auc_roc']:.4f} {h['val_auc_pr']:.4f} "
                f"{h['val_eod']:.4f} {h['val_worst_group_ece']:.4f} {h['adv_head_auc']:.3f} "
                f"{h['z_probe_auc']:.3f}"
            )
    d = resolve("artifacts/smoke/joint_stability")
    save_json(d / f"{name}.json", out)


if __name__ == "__main__":
    main()
