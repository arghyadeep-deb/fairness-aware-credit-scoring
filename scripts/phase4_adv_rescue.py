"""Phase 4 adversary rescue (time-boxed, validation only, seed 0, fairness/calibration terms OFF).

Success criterion, fixed BEFORE the first attempt: a fresh LightGBM probe on frozen z (Phase 1
protocol) drops by >= 0.03 AUC below the control's, AND default AUC-ROC loses <= 0.01 vs control.
Stages in order, stop after the first stage that has a working setting (all of a stage's settings
are still run and reported):
  a  GRL with small lambda_a            0.02, 0.05, 0.1, 0.2
  b  GRL + Head-B lr x10 (slower trunk-side reversal relative to the head)   lambda_a 0.1, 0.5, 1.0
  c  bounded confusion loss (deviation from the spec GRL)                    lambda_a 0.1, 0.5, 1.0
python scripts/phase4_adv_rescue.py <out_dir>
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.eval.zprobe import probe_z  # noqa: E402
from src.run_arms import save_json, sigmoid  # noqa: E402
from src.run_joint import load_setup  # noqa: E402
from src.train_joint import fit_joint  # noqa: E402
from src.utils import load_config  # noqa: E402

PROBE_DROP, AUC_LOSS = 0.03, 0.01
STAGES = {
    "a_grl_small": [(0.02, {}), (0.05, {}), (0.1, {}), (0.2, {})],
    "b_grl_fast_head": [(0.1, {"adv_lr_mult": 10.0}), (0.5, {"adv_lr_mult": 10.0}),
                        (1.0, {"adv_lr_mult": 10.0})],
    "c_confusion": [(0.1, {"adv_mode": "confusion"}), (0.5, {"adv_mode": "confusion"}),
                    (1.0, {"adv_mode": "confusion"})],
}  # fmt: skip


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/train.yaml")
    device, train, val, cards, weights, hp = load_setup(cfg)
    rows, t_start = [], time.time()

    def attempt(stage: str, lam_a: float, extra: dict) -> dict:
        t0 = time.time()
        r = fit_joint(
            cfg, hp, train, val, cards, weights, 0, device, (0.0, 0.0, lam_a), schedule=extra
        )
        p = sigmoid(r["val_logits"])
        pr = probe_z(cfg, hp, r["state"], train, val, cards, device)
        h = r["history"]
        row = {
            "stage": stage, "lambda_a": lam_a, **extra,
            "auc_roc": float(roc_auc_score(val.y, p)),
            "auc_pr": float(average_precision_score(val.y, p)),
            "best_epoch": r["best_epoch"], "epochs": len(h),
            "adv_head_auc_best": h[r["best_epoch"]]["adv_head_auc"],
            "max_train_adv_loss": max((x.get("train_adv", 0.0) for x in h), default=0.0),
            **pr, "seconds": time.time() - t0,
        }  # fmt: skip
        rows.append(row)
        save_json(out / "attempts.json", rows)
        print(
            f"{stage:16s} la={lam_a:<5} {extra} AUC-ROC {row['auc_roc']:.4f} "
            f"probe lgbm {row['z_probe_lgbm_auc']:.4f} logit {row['z_probe_logistic_auc']:.4f} "
            f"headAUC {row['adv_head_auc_best']:.3f} maxAdvLoss {row['max_train_adv_loss']:.1f} "
            f"({row['seconds']:.0f}s)", flush=True,
        )  # fmt: skip
        return row

    ctrl = attempt("control", 0.0, {})
    for stage, settings in STAGES.items():
        res = [attempt(stage, la, ex) for la, ex in settings]
        ok = [
            r for r in res
            if ctrl["z_probe_lgbm_auc"] - r["z_probe_lgbm_auc"] >= PROBE_DROP
            and ctrl["auc_roc"] - r["auc_roc"] <= AUC_LOSS
        ]  # fmt: skip
        print(f"== stage {stage}: {len(ok)} working setting(s)", flush=True)
        if ok:
            break
        if (time.time() - t_start) / 3600 > 1.0:
            print("== time box (1 h) reached", flush=True)
            break
    save_json(out / "summary.json", {"criterion": {"probe_drop": PROBE_DROP, "auc_loss": AUC_LOSS},
                                     "control": ctrl, "attempts": rows,
                                     "hours": (time.time() - t_start) / 3600})  # fmt: skip


if __name__ == "__main__":
    main()
