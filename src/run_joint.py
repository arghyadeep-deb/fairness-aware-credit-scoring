"""Phase 4 runner: smoke check, lambda sweep (resumable) and ablations for the joint model (JT).

python -m src.run_joint smoke
python -m src.run_joint sweep --run-dir <dir> --points coarse --seeds 0 1 2
python -m src.run_joint sweep --run-dir <dir> --points custom --lambdas 2,2,1 4,2,1 --seeds 0 1 2

Everything is evaluated on VALIDATION only. Hyper-parameters are A2's tuned values (no re-tuning).
A point is the folder  <run_dir>/jt/f{lf}_c{lc}_a{la}/seed_{s}/  and is skipped when already done.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.prep import data_dirs
from src.data.tensors import load_arrays
from src.eval.metrics import compute_metrics
from src.run_arms import flags_from_shared_rule, save_json, save_predictions, sigmoid
from src.train_joint import fit_joint
from src.utils import git_hash, load_config, resolve

log = logging.getLogger(__name__)


def tag(lam: tuple[float, float, float]) -> str:
    f = lambda v: f"{v:g}".replace(".", "p")  # noqa: E731
    return f"f{f(lam[0])}_c{f(lam[1])}_a{f(lam[2])}"


def load_setup(cfg: dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA unavailable (hard stop; no --allow-cpu given)")
    train, cards = load_arrays(cfg, "train")
    val, _ = load_arrays(cfg, "val")
    weights = json.loads((data_dirs(cfg)[1] / "class_weights.json").read_text())
    hp = json.loads(resolve(cfg["joint"]["hp_source"]).read_text())["best_params"]
    return device, train, val, cards, weights, hp


def run_point(
    cfg,
    out_root: Path,
    lam,
    seed,
    setup,
    name: str = "jt",
    ablation_of: str | None = None,
    schedule: dict | None = None,
) -> dict[str, Any]:
    device, train, val, cards, weights, hp = setup
    d = out_root / name / tag(lam) / f"seed_{seed}"
    if (d / "summary.json").exists():
        return json.loads((d / "summary.json").read_text())
    t0 = time.time()
    r = fit_joint(cfg, hp, train, val, cards, weights, seed, device, lam, schedule=schedule)
    prob = sigmoid(r["val_logits"])
    flags, thr = flags_from_shared_rule(prob, cfg["evaluation"]["flag_rate"])
    m = compute_metrics(val.y, prob, val.a, flags, cfg["evaluation"]["ece_bins"])
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "val_logits.npy", r["val_logits"])
    torch.save(r["state"], d / "best.pt")
    save_json(d / "history.json", r["history"])
    info = {
        "lambdas": {"f": lam[0], "c": lam[1], "a": lam[2]},
        "seed": seed,
        "best_epoch": r["best_epoch"],
        "epochs_run": len(r["history"]),
        "seconds": time.time() - t0,
        "below_auc_floor": r["below_auc_floor"],
        "lr_scale": r["lr_scale"],
        "norm_reference": r["norm_reference"],
        "threshold": thr,
        "adv_head_auc_best": r["history"][r["best_epoch"]]["adv_head_auc"],
        "ablation_of": ablation_of,
        "schedule_overrides": schedule or {},
        **{f"val_{k}": v for k, v in m.items()},
    }
    save_predictions(d, prob, flags, info)
    save_json(d / "summary.json", info)
    log.info(
        "%s %s s%d: AUC-ROC %.4f AUC-PR %.4f EOD %.4f wECE %.4f advAUC %.3f ep %d (%.0fs)",
        name,
        tag(lam),
        seed,
        m["auc_roc"],
        m["auc_pr"],
        m["eod"],
        m["worst_group_ece"],
        info["adv_head_auc_best"],
        r["best_epoch"],
        info["seconds"],
    )
    return info


def coarse_grid(cfg) -> list[tuple[float, float, float]]:
    g = cfg["sweep"]["grid"]
    return [
        (float(a), float(b), float(c))
        for a, b, c in itertools.product(g["lambda_f"], g["lambda_c"], g["lambda_a"])
    ]


def smoke(cfg) -> None:
    """3-epoch compressed-schedule run (1 warm-up + 1 ramp + 1 full) with the gender probe on z."""
    setup = load_setup(cfg)
    device, train, val, cards, weights, hp = setup
    sm = cfg["joint"]["smoke"]
    sched = {"warmup_epochs": sm["warmup_epochs"], "ramp_epochs": sm["ramp_epochs"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = resolve(cfg["paths"]["smoke_root"]) / "joint_smoke" / stamp
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, lam in {
        "control_0_0_0": (0.0, 0.0, 0.0),
        "adv_only_0_0_1": (0.0, 0.0, 1.0),
        "full_2_2_1": (2.0, 2.0, 1.0),
    }.items():
        r = fit_joint(
            cfg, hp, train, val, cards, weights, 0, device, lam,
            max_epochs=sm["epochs"], schedule=sched, probe=True,
        )  # fmt: skip
        rows[name] = {
            "lambdas": lam,
            "history": r["history"],
            "norm_reference": r["norm_reference"],
        }
        for h in r["history"]:
            log.info(
                "%-16s ep%d ramp %.2f ROC %.4f EOD %.4f wECE %.4f advAUC %.3f zProbe %.3f",
                name, h["epoch"], h["ramp"], h["val_auc_roc"], h["val_eod"],
                h["val_worst_group_ece"], h["adv_head_auc"], h["z_probe_auc"],
            )  # fmt: skip
    save_json(out / "smoke.json", {"schedule": sched, "git_hash": git_hash(), "runs": rows})
    print(f"SMOKE_DIR={out}")


def sweep(cfg, args) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup = load_setup(cfg)
    if args.points == "coarse":
        pts = coarse_grid(cfg)
    else:
        pts = [tuple(float(x) for x in s.split(",")) for s in args.lambdas]
    seeds = args.seeds if args.seeds is not None else cfg["sweep"]["seeds"]
    save_json(
        run_dir / f"sweep_meta_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
        {"git_hash": git_hash(), "points": pts, "seeds": seeds, "name": args.name,
         "split": "validation only", "hp": setup[5]},
    )  # fmt: skip
    sched = {"adv_mode": args.adv_mode} if args.adv_mode != "grl" else None
    t0 = time.time()
    done = 0
    for lam in pts:
        for s in seeds:
            run_point(cfg, run_dir, lam, s, setup, args.name, args.ablation_of, sched)
            done += 1
            el = time.time() - t0
            log.info("progress %d/%d runs, %.1f min elapsed", done, len(pts) * len(seeds), el / 60)
    print(f"RUN_DIR={run_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["smoke", "sweep"])
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--run-dir")
    p.add_argument("--points", default="coarse", choices=["coarse", "custom"])
    p.add_argument("--lambdas", nargs="*", default=[], help="f,c,a triples for --points custom")
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--name", default="jt")
    p.add_argument("--ablation-of", default=None)
    p.add_argument("--adv-mode", default="grl", choices=["grl", "confusion"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    smoke(cfg) if args.mode == "smoke" else sweep(cfg, args)


if __name__ == "__main__":
    main()
