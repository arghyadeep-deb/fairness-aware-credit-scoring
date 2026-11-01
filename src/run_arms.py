"""Phase 2 runner: tune, train, post-process and evaluate every baseline arm (resumable).

python -m src.run_arms --arms a0 a1 a2 a3 a4 a5 g1 s1 s2 ref gi --seeds 0 1 2
Every invocation writes to a NEW timestamped run folder unless --resume <folder> is given
(then completed (arm, seed) pairs are skipped). Evaluation is on VALIDATION only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch

from src.data.prep import data_dirs
from src.data.tensors import load_arrays
from src.eval.threshold import apply_threshold, flag_threshold
from src.posthoc.prior_correction import correct_logits
from src.posthoc.threshold_optimizer import (
    apply_group_thresholds,
    fairlearn_threshold_flags,
    fit_group_thresholds,
)
from src.train import fit, tune
from src.utils import git_hash, load_config, mlflow_experiment, resolve

log = logging.getLogger(__name__)
NEURAL = ["a0", "a1", "a2", "a3", "a4", "a5", "g1"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype="float64")))


def arm_cfg(name: str) -> dict[str, Any]:
    return load_config(f"configs/arm_{name}.yaml")["arm"]


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def save_predictions(
    d: Path, prob: np.ndarray, flags: np.ndarray, extra: dict | None = None
) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "val_prob.npy", prob.astype("float64"))
    np.save(d / "val_flags.npy", flags.astype(bool))
    save_json(d / "pred_meta.json", extra or {})


def flags_from_shared_rule(prob: np.ndarray, flag_rate: float) -> tuple[np.ndarray, float]:
    thr = flag_threshold(prob, flag_rate)
    return apply_threshold(prob, thr), thr


def log_mlflow(cfg, exp_id, name, params, metrics, run_dir: Path) -> None:
    import mlflow

    with mlflow.start_run(experiment_id=exp_id, run_name=name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_params({"run_dir": run_dir.name, "git_hash": git_hash()[:12]})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def train_arm(
    name,
    cfg,
    run_dir,
    train,
    val,
    cards,
    weights,
    seeds,
    device,
    exp_id,
    hp_override=None,
    out_name=None,
    max_epochs=None,
    tuning_kwargs=None,
) -> dict[str, Any]:
    arm = arm_cfg(name)
    out = run_dir / (out_name or name)
    if hp_override is not None:
        hp = hp_override
    else:
        tfile = out / "tuning.json"
        if tfile.exists():
            hp = json.loads(tfile.read_text())["best_params"]
        else:
            t0 = time.time()
            res = tune(arm, cfg, train, val, cards, weights, device, **(tuning_kwargs or {}))
            res["seconds"] = time.time() - t0
            save_json(tfile, res)
            hp = res["best_params"]
            log.info("tuned %s in %.0fs: %s", name, res["seconds"], hp)
    summary = {"hp": hp, "seeds": {}}
    for s in seeds:
        d = out / f"seed_{s}"
        if (d / "val_prob.npy").exists():
            log.info("skip %s seed %d (done)", name, s)
            summary["seeds"][s] = json.loads((d / "summary.json").read_text())
            continue
        t0 = time.time()
        r = fit(arm, cfg, hp, train, val, cards, weights, s, device, max_epochs)
        prob = sigmoid(r["val_logits"])
        flags, thr = flags_from_shared_rule(prob, cfg["evaluation"]["flag_rate"])
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "val_logits.npy", r["val_logits"])
        torch.save(r["state"], d / "best.pt")
        save_json(d / "history.json", r["history"])
        info = {
            "best_val_auc_pr": r["best_val_auc_pr"],
            "best_epoch": r["best_epoch"],
            "epochs_run": len(r["history"]),
            "seconds": time.time() - t0,
            "first_epoch_seconds": r["history"][0]["seconds"],
            "threshold": thr,
        }
        save_predictions(d, prob, flags, info)
        save_json(d / "summary.json", info)
        summary["seeds"][s] = info
        log_mlflow(
            cfg,
            exp_id,
            f"{out_name or name}_s{s}",
            {"arm": name, "seed": s, **hp},
            {k: v for k, v in info.items() if isinstance(v, (int, float))},
            run_dir,
        )
        log.info(
            "%s seed %d: AUC-PR %.4f epochs %d (%.0fs)",
            name,
            s,
            r["best_val_auc_pr"],
            len(r["history"]),
            info["seconds"],
        )
    save_json(out / "summary.json", summary)
    return summary


def run_s1(cfg, run_dir, val, weights, seeds, exp_id) -> None:
    for s in seeds:
        d = run_dir / "s1" / f"seed_{s}"
        if (d / "val_prob.npy").exists():
            continue
        logits = np.load(run_dir / "a2" / f"seed_{s}" / "val_logits.npy")
        prob = sigmoid(correct_logits(logits, weights))
        flags, thr = flags_from_shared_rule(prob, cfg["evaluation"]["flag_rate"])
        save_predictions(d, prob, flags, {"threshold": thr, "source": "a2"})


def run_s2(cfg, run_dir, val, seeds) -> None:
    fr = cfg["evaluation"]["flag_rate"]
    for s in seeds:
        d = run_dir / "s2" / f"seed_{s}"
        if (d / "val_prob.npy").exists():
            continue
        prob = np.load(run_dir / "s1" / f"seed_{s}" / "val_prob.npy")
        thr = fit_group_thresholds(val.y, prob, val.a, fr)
        flags = apply_group_thresholds(prob, val.a, thr)
        save_predictions(d, prob, flags, {**thr, "source": "s1", "fit_on": "validation"})
        d2 = run_dir / "s2_fairlearn" / f"seed_{s}"
        fl = fairlearn_threshold_flags(val.y, prob, val.a, seed=s)
        save_predictions(
            d2, prob, fl, {"note": "fairlearn native operating point, not rate-matched"}
        )


def run_ref(cfg, run_dir, train, val, seeds) -> None:
    p = dict(cfg["reference"]["lightgbm"])
    stop = p.pop("early_stopping_rounds")
    n1 = train.y.sum()
    for s in seeds:
        d = run_dir / "ref" / f"seed_{s}"
        if (d / "val_prob.npy").exists():
            continue
        X = np.concatenate([train.cont, train.cat], axis=1)
        Xv = np.concatenate([val.cont, val.cat], axis=1)
        cats = list(range(train.cont.shape[1], X.shape[1]))
        m = lgb.LGBMClassifier(
            **p,
            objective="binary",
            random_state=s,
            n_jobs=-1,
            verbose=-1,
            scale_pos_weight=float((len(train.y) - n1) / n1),
        )
        m.fit(
            X,
            train.y,
            eval_set=[(Xv, val.y)],
            eval_metric="average_precision",
            categorical_feature=cats,
            callbacks=[lgb.early_stopping(stop, verbose=False)],
        )
        prob = m.predict_proba(Xv)[:, 1]
        flags, thr = flags_from_shared_rule(prob, cfg["evaluation"]["flag_rate"])
        save_predictions(
            d, prob, flags, {"threshold": thr, "best_iteration": int(m.best_iteration_ or 0)}
        )


def best_trained_arm(run_dir: Path, candidates: list[str]) -> str:
    scores = {}
    for c in candidates:
        f = run_dir / c / "summary.json"
        if f.exists():
            sm = json.loads(f.read_text())["seeds"]
            scores[c] = float(np.mean([v["best_val_auc_pr"] for v in sm.values()]))
    if not scores:
        raise RuntimeError("No trained arm to pick GI from")
    return max(scores, key=scores.get)


def measure(cfg, device) -> None:
    """Time one real epoch per model family on full train data and project total compute."""
    train, cards = load_arrays(cfg, "train")
    val, _ = load_arrays(cfg, "val")
    weights = json.loads((data_dirs(cfg)[1] / "class_weights.json").read_text())
    hp = {"lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "emb_dim": 32, "focal_gamma": 2.0}
    secs = {}
    for name in ["a2", "a3", "a4", "g1"]:
        r = fit(arm_cfg(name), cfg, hp, train, val, cards, weights, 0, device, max_epochs=2)
        secs[name] = [h["seconds"] for h in r["history"]]
        print(f"{name}: epoch seconds {secs[name]}")
    t0 = time.time()
    from src.data.tensors import smote_nc

    sm = smote_nc(train, cfg["smote"]["sampling_ratio"], cfg["smote"]["k_neighbors"], 0)
    print(f"SMOTE-NC on full train: {time.time() - t0:.1f}s -> {len(sm.y)} rows")
    ep = {k: v[-1] for k, v in secs.items()}
    tf = ep["a2"]
    final = (
        3 * 30 * (6 * tf + 1.4 * tf + ep["a3"] + ep["a4"]) / 1.0
    )  # worst case: 30 epochs, A1 +40% rows
    tune_t = 10 * 8 * 0.3 * (6 * tf + 1.4 * tf + ep["a3"] + ep["a4"])
    print(f"projected worst-case: final {final / 60:.1f} min, tuning {tune_t / 60:.1f} min")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--arms", nargs="+", default=NEURAL + ["s1", "s2", "ref", "gi"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--measure", action="store_true")
    p.add_argument("--resume", default=None, help="existing run folder to continue")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA unavailable (hard stop; no --allow-cpu given)")
    if args.measure:
        return measure(cfg, device)
    seeds = args.seeds if args.seeds is not None else cfg["training"]["seeds"]
    smoke_kw: dict[str, Any] = {}
    if args.smoke:
        seeds, smoke_kw = seeds[:1], {"trials": 2, "epochs": 2}
    runs_root = (
        resolve(cfg["paths"]["smoke_root"]) / "runs" if args.smoke else resolve(cfg["runs_dir"])
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.resume) if args.resume else runs_root / f"{stamp}_phase2"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        run_dir / "run_meta.json",
        {
            "git_hash": git_hash(),
            "arms": args.arms,
            "seeds": seeds,
            "smoke": args.smoke,
            "started_utc": stamp,
            "split": "validation only",
        },
    )
    exp_id = mlflow_experiment(cfg, cfg["mlflow_experiment"] + ("_smoke" if args.smoke else ""))
    train, cards = load_arrays(cfg, "train", smoke=args.smoke)
    val, _ = load_arrays(cfg, "val", smoke=args.smoke)
    weights = json.loads((data_dirs(cfg, args.smoke)[1] / "class_weights.json").read_text())
    max_ep = 2 if args.smoke else None
    for name in [a for a in NEURAL if a in args.arms]:
        train_arm(
            name,
            cfg,
            run_dir,
            train,
            val,
            cards,
            weights,
            seeds,
            device,
            exp_id,
            max_epochs=max_ep,
            tuning_kwargs=smoke_kw,
        )
    if "s1" in args.arms:
        run_s1(cfg, run_dir, val, weights, seeds, exp_id)
    if "s2" in args.arms:
        run_s2(cfg, run_dir, val, seeds)
    if "ref" in args.arms:
        run_ref(cfg, run_dir, train, val, seeds)
    if "gi" in args.arms:
        best = best_trained_arm(run_dir, [a for a in ["a0", "a1", "a2", "a3", "a4", "a5"]])
        hp = json.loads((run_dir / best / "tuning.json").read_text())["best_params"]
        tr_g, cards = load_arrays(cfg, "train", smoke=args.smoke, include_gender=True)
        va_g, _ = load_arrays(cfg, "val", smoke=args.smoke, include_gender=True)
        gi_arm = {**arm_cfg(best), "name": "gi"}
        # GI reuses the best arm's config and tuned hp, with gender included
        save_json(run_dir / "gi_source.json", {"source_arm": best, "hp": hp})
        _train_gi(
            gi_arm, cfg, run_dir, tr_g, va_g, cards, weights, seeds, device, exp_id, hp, max_ep
        )
    print(f"RUN_DIR={run_dir}")


def _train_gi(arm, cfg, run_dir, train, val, cards, weights, seeds, device, exp_id, hp, max_ep):
    out = run_dir / "gi"
    for s in seeds:
        d = out / f"seed_{s}"
        if (d / "val_prob.npy").exists():
            continue
        t0 = time.time()
        r = fit(arm, cfg, hp, train, val, cards, weights, s, device, max_ep)
        prob = sigmoid(r["val_logits"])
        flags, thr = flags_from_shared_rule(prob, cfg["evaluation"]["flag_rate"])
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "val_logits.npy", r["val_logits"])
        save_json(d / "history.json", r["history"])
        info = {
            "best_val_auc_pr": r["best_val_auc_pr"],
            "best_epoch": r["best_epoch"],
            "epochs_run": len(r["history"]),
            "seconds": time.time() - t0,
            "threshold": thr,
            "gender_included": True,
            "deployable": False,
        }
        save_predictions(d, prob, flags, info)
        save_json(d / "summary.json", info)
        log_mlflow(
            cfg,
            exp_id,
            f"gi_s{s}",
            {"arm": "gi", "seed": s, **hp},
            {k: v for k, v in info.items() if isinstance(v, (int, float))},
            run_dir,
        )


if __name__ == "__main__":
    main()
