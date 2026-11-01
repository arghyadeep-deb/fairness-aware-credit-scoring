"""Phase 4b-1: Optuna tuning parity for JT at a FIXED lambda point, then 3-seed retrain (validation only).

python scripts/phase4b_tune_jt.py <out_dir> <f,c,a>
Budget identical to the other tuned arms (configs/train.yaml optuna: 10 trials, 30% subsample,
8 epochs, n_startup_trials 4, sampler seed 42, objective val AUC-PR). Tuning runs use a compressed
schedule (2 warm-up + 2 ramp); the 3-seed retrain uses the production 3 + 5 schedule.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optuna  # noqa: E402

from src.data.tensors import stratified_subsample  # noqa: E402
from src.run_arms import save_json  # noqa: E402
from src.run_joint import load_setup, run_point  # noqa: E402
from src.train import sample_hp  # noqa: E402
from src.train_joint import fit_joint  # noqa: E402
from src.utils import git_hash, load_config  # noqa: E402

TUNE_SCHEDULE = {"warmup_epochs": 2, "ramp_epochs": 2}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = Path(sys.argv[1])
    lam = tuple(float(x) for x in sys.argv[2].split(","))
    cfg = load_config("configs/train.yaml")
    device, train, val, cards, weights, hp0 = load_setup(cfg)
    oc = cfg["optuna"]
    sub = stratified_subsample(train, oc["subsample"], oc["sampler_seed"])
    arm = {"model": "tabtransformer", "loss": "wce"}
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hp(trial, arm, cfg)
        r = fit_joint(
            cfg, hp, sub, val, cards, weights, oc["train_seed"], device, lam,
            max_epochs=oc["epochs"], schedule=TUNE_SCHEDULE,
        )  # fmt: skip
        return r["best_val_auc_pr"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=oc["sampler_seed"], n_startup_trials=oc["n_startup_trials"]
        ),
    )
    study.optimize(objective, n_trials=oc["trials"])
    tuning = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "lambdas": lam,
        "tune_schedule": TUNE_SCHEDULE,
        "untuned_hp (A2)": hp0,
        "trials": [{"params": t.params, "value": t.value} for t in study.trials],
        "seconds": time.time() - t0,
        "git_hash": git_hash(),
    }
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "jt_tuning.json", tuning)
    print("best", study.best_params, study.best_value, f"({time.time() - t0:.0f}s)")
    setup = (device, train, val, cards, weights, study.best_params)
    for s in cfg["sweep"]["seeds"]:
        run_point(cfg, out, lam, s, setup, name="jt_tuned")
    print(f"RUN_DIR={out}")


if __name__ == "__main__":
    main()
