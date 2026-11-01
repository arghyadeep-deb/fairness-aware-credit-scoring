"""ONE trainer for every neural arm (model + loss + sampling chosen by config) and Optuna tuning."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

import numpy as np
import optuna
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.tensors import Arrays, smote_nc, stratified_subsample
from src.losses.calibration import soft_group_ece
from src.losses.focal import focal_loss
from src.losses.weighted_ce import weighted_bce
from src.models.nets import build_model
from src.utils import seed_everything

log = logging.getLogger(__name__)


def make_loss(
    arm: dict[str, Any],
    cfg: dict[str, Any],
    hp: dict[str, Any],
    weights: dict[str, float],
    prior_pos: float,
):
    """Return loss(logits, y, a) -> scalar for the arm's loss name."""
    kind = arm["loss"]
    if kind == "ce":
        return lambda lg, y, a: weighted_bce(lg, y)
    if kind in ("wce", "wce_gcal"):
        w0, w1 = weights["0"], weights["1"]
        gc = cfg["group_calibration"]

        def fn(lg, y, a):
            base = weighted_bce(lg, y, w_neg=w0, w_pos=w1)
            if kind == "wce":
                return base
            return base + gc["lambda_c"] * soft_group_ece(lg, y, a, gc["bins"], gc["sigma"])

        return fn
    if kind == "focal":
        gamma = float(hp["focal_gamma"])
        return lambda lg, y, a: focal_loss(lg, y, gamma, alpha_pos=1.0 - prior_pos)
    raise ValueError(f"Unknown loss {kind}")


@torch.no_grad()
def predict_logits(model, arr: Arrays, device: torch.device, batch: int = 8192) -> np.ndarray:
    model.eval()
    xc = torch.from_numpy(arr.cont).to(device)
    xk = torch.from_numpy(arr.cat).to(device)
    out = []
    for i in range(0, len(xc), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg, _ = model(xc[i : i + batch], xk[i : i + batch])
        out.append(lg.float().cpu())
    return torch.cat(out).numpy()


def fit(
    arm: dict[str, Any],
    cfg: dict[str, Any],
    hp: dict[str, Any],
    train: Arrays,
    val: Arrays,
    cardinalities: list[int],
    weights: dict[str, float],
    seed: int,
    device: torch.device,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Train one (arm, seed). Early stop on val AUC-PR; returns best-epoch val logits + history."""
    tc = cfg["training"]
    max_epochs = max_epochs or tc["max_epochs"]
    seed_everything(seed)
    if arm["sampling"] == "smote_nc":
        train = smote_nc(train, cfg["smote"]["sampling_ratio"], cfg["smote"]["k_neighbors"], seed)
    prior_pos = float(np.mean(train.y))
    model = build_model(arm["model"], train.cont.shape[1], cardinalities, cfg["model_defaults"], hp)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = make_loss(arm, cfg, hp, weights, prior_pos)
    xc = torch.from_numpy(train.cont).to(device)
    xk = torch.from_numpy(train.cat).to(device)
    yt = torch.from_numpy(train.y).float().to(device)
    at = torch.from_numpy(train.a).long().to(device)
    n, bs = len(yt), tc["batch_size"]
    best, best_state, best_logits, bad, hist = -1.0, None, None, 0, []
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device)
        total, steps = 0.0, 0
        for i in range(0, n, bs):
            ix = perm[i : i + bs]
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and tc["bf16"]
            ):
                logit, _ = model(xc[ix], xk[ix])
            loss = loss_fn(logit.float(), yt[ix], at[ix])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch} (arm {arm['name']})")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            total += loss.item()
            steps += 1
        lg = predict_logits(model, val, device)
        p = 1 / (1 + np.exp(-lg.astype("float64")))
        ap, roc = float(average_precision_score(val.y, p)), float(roc_auc_score(val.y, p))
        hist.append(
            {
                "epoch": epoch,
                "train_loss": total / steps,
                "val_auc_pr": ap,
                "val_auc_roc": roc,
                "seconds": time.time() - t0,
            }
        )
        if ap > best:
            best, bad, best_logits = ap, 0, lg
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    return {
        "val_logits": best_logits,
        "best_val_auc_pr": best,
        "history": hist,
        "best_epoch": int(np.argmax([h["val_auc_pr"] for h in hist])),
        "state": best_state,
    }


def sample_hp(trial: optuna.Trial, arm: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Search space from config. Core 3 params for every arm; extras only where specs/09 says."""
    sp = cfg["optuna"]["space"]
    hp = {
        "lr": trial.suggest_float("lr", sp["lr"]["low"], sp["lr"]["high"], log=sp["lr"]["log"]),
        "weight_decay": trial.suggest_float(
            "weight_decay", sp["weight_decay"]["low"], sp["weight_decay"]["high"], log=True
        ),
        "dropout": trial.suggest_float("dropout", sp["dropout"]["low"], sp["dropout"]["high"]),
    }
    if arm["model"] == "tabtransformer":
        hp["emb_dim"] = trial.suggest_categorical("emb_dim", sp["emb_dim"])
    if arm["loss"] == "focal":
        hp["focal_gamma"] = trial.suggest_float(
            "focal_gamma", sp["focal_gamma"]["low"], sp["focal_gamma"]["high"]
        )
    return hp


def tune(
    arm: dict[str, Any],
    cfg: dict[str, Any],
    train: Arrays,
    val: Arrays,
    cardinalities: list[int],
    weights: dict[str, float],
    device: torch.device,
    trials: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Optuna TPE: same trials/epochs/subsample for every tuned arm; objective = val AUC-PR."""
    oc = cfg["optuna"]
    trials, epochs = trials or oc["trials"], epochs or oc["epochs"]
    sub = stratified_subsample(train, oc["subsample"], oc["sampler_seed"])

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hp(trial, arm, cfg)
        res = fit(arm, cfg, hp, sub, val, cardinalities, weights, oc["train_seed"], device, epochs)
        return res["best_val_auc_pr"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=oc["sampler_seed"])
    )
    study.optimize(objective, n_trials=trials)
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials": [{"params": t.params, "value": t.value} for t in study.trials],
    }
