"""Joint trainer (Phase 4): TabTransformer + adversary head, L = wCE + lf*Lfair + lc*Lcal + la*Ladv.

Same optimiser, batch size, bf16, grad-clip and early-stopping metric (val AUC-PR) as the baseline
trainer in src/train.py; adds the stratified batch sampler, the warm-up/ramp schedule with
per-term normalisation, a non-finite-loss recovery (halve lr once, then stop), and per-epoch
monitoring of the gender head. Early stopping only starts once the lambdas are at full strength.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.sampler import stratified_epoch
from src.data.tensors import Arrays
from src.eval.metrics import compute_metrics
from src.eval.threshold import apply_threshold, flag_threshold
from src.losses.joint import JointConfig, JointLoss
from src.models.heads import JointModel
from src.models.nets import build_model
from src.utils import seed_everything

log = logging.getLogger(__name__)


@torch.no_grad()
def predict_joint(model: JointModel, arr: Arrays, device, batch: int = 8192):
    """Returns (logit, adversary logit, z) as numpy arrays."""
    model.eval()
    xc, xk = torch.from_numpy(arr.cont).to(device), torch.from_numpy(arr.cat).to(device)
    lg, ad, zs = [], [], []
    for i in range(0, len(xc), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out_l, z, d = model(xc[i : i + batch], xk[i : i + batch])
        d = d[0] if isinstance(d, tuple) else d
        lg.append(out_l.float().cpu())
        ad.append(d.float().cpu())
        zs.append(z.float().cpu())
    return torch.cat(lg).numpy(), torch.cat(ad).numpy(), torch.cat(zs).numpy()


def z_gender_probe(ztr: np.ndarray, atr: np.ndarray, zva: np.ndarray, ava: np.ndarray) -> float:
    """AUC (on val) of a logistic regression that predicts gender from the representation z."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(ztr)
    m = LogisticRegression(max_iter=200).fit(sc.transform(ztr), atr)
    return float(roc_auc_score(ava, m.predict_proba(sc.transform(zva))[:, 1]))


def _once(cfg, hp, train, val, cards, weights, seed, device, jc, max_epochs, lr_scale, tr_probe):
    tc = cfg["training"]
    seed_everything(seed)
    gen = torch.Generator().manual_seed(seed)
    backbone = build_model("tabtransformer", train.cont.shape[1], cards, cfg["model_defaults"], hp)
    model = JointModel(backbone, cfg["joint"]["adversary_hidden"], jc.adv_mode).to(device)
    lr = hp["lr"] * lr_scale
    groups = [
        {"params": list(model.backbone.parameters()), "lr": lr},
        {"params": list(model.adversary.parameters()), "lr": lr * jc.adv_lr_mult},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=hp["weight_decay"])
    loss_fn = JointLoss(jc, weights["0"], weights["1"])
    xc, xk = torch.from_numpy(train.cont).to(device), torch.from_numpy(train.cat).to(device)
    yt = torch.from_numpy(train.y).float().to(device)
    at = torch.from_numpy(train.a).long().to(device)
    bs, floor = tc["batch_size"], cfg["joint"]["sampler"]["min_per_cell"]
    first_full = (
        int(np.ceil(jc.warmup_epochs + jc.ramp_epochs)) - 1
    )  # first epoch ending at full lambda
    best, best_state, best_logits, best_ep, bad, hist = -1.0, None, None, -1, 0, []
    lg = None
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        batches = stratified_epoch(yt, at, bs, floor, gen)
        steps = batches.shape[0]
        sums: dict[str, float] = {}
        for step in range(steps):
            ix = batches[step]
            loss_fn.set_epoch(epoch + step / steps)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and tc["bf16"]
            ):
                logit, z, adv = model(xc[ix], xk[ix])
            loss, parts = loss_fn(logit.float(), yt[ix], at[ix], adv)
            if jc.adv_pretrain and jc.lambda_a > 0 and loss_fn.epoch < jc.warmup_epochs:
                # diagnostic only: Head B learns on detached z, so the trunk is untouched
                pre = model.adversary.net(z.detach().float()).squeeze(-1).float()
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    pre, at[ix].float()
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sums["loss"] = sums.get("loss", 0.0) + loss.item()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v)
        lg, adv_lg, zva = predict_joint(model, val, device)
        p = 1 / (1 + np.exp(-lg.astype("float64")))
        ap, roc = float(average_precision_score(val.y, p)), float(roc_auc_score(val.y, p))
        flags = apply_threshold(p, flag_threshold(p, cfg["evaluation"]["flag_rate"]))
        m = compute_metrics(val.y, p, val.a, flags, cfg["evaluation"]["ece_bins"])
        row = {
            "epoch": epoch,
            **{f"train_{k}": v / steps for k, v in sums.items()},
            "val_auc_pr": ap,
            "val_auc_roc": roc,
            "val_eod": m["eod"],
            "val_dir": m["dir"],
            "val_ece": m["ece"],
            "val_worst_group_ece": m["worst_group_ece"],
            "adv_head_auc": float(roc_auc_score(val.a, adv_lg)),
            "ramp": loss_fn.ramp(),
            "seconds": time.time() - t0,
        }
        if tr_probe is not None:
            ztr = predict_joint(model, tr_probe, device)[2]
            row["z_probe_auc"] = z_gender_probe(ztr, tr_probe.a, zva, val.a)
        hist.append(row)
        if epoch >= first_full:  # model selection / early stopping only at full strength
            if ap > best:
                best, bad, best_logits, best_ep = ap, 0, lg, epoch
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            else:
                bad += 1
                if bad >= tc["patience"]:
                    break
    if best_logits is None:  # run shorter than the schedule (smoke): keep the last epoch
        best_logits, best_ep, best = lg, len(hist) - 1, hist[-1]["val_auc_pr"]
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return {
        "val_logits": best_logits,
        "best_val_auc_pr": best,
        "best_epoch": best_ep,
        "history": hist,
        "state": best_state,
        "norm_reference": loss_fn.ref,
    }


def fit_joint(
    cfg: dict[str, Any],
    hp: dict[str, Any],
    train: Arrays,
    val: Arrays,
    cards: list[int],
    weights: dict[str, float],
    seed: int,
    device,
    lambdas: tuple[float, float, float],
    max_epochs: int | None = None,
    schedule: dict | None = None,
    probe: bool = False,
) -> dict[str, Any]:
    """Train one joint model. A non-finite loss halves the lr once; a second one is a hard stop."""
    jc = JointConfig.from_config(cfg, lambda_f=lambdas[0], lambda_c=lambdas[1], lambda_a=lambdas[2])
    for k, v in (schedule or {}).items():
        setattr(jc, k, v)
    max_epochs = max_epochs or cfg["training"]["max_epochs"]
    tr_probe = None
    if probe:
        rs = np.random.default_rng(0)
        tr_probe = train.subset(np.sort(rs.choice(len(train.y), 30000, replace=False)))
    for lr_scale in (1.0, 0.5):
        try:
            out = _once(
                cfg,
                hp,
                train,
                val,
                cards,
                weights,
                seed,
                device,
                jc,
                max_epochs,
                lr_scale,
                tr_probe,
            )
            out["lr_scale"] = lr_scale
            out["below_auc_floor"] = bool(
                roc_auc_score(val.y, out["val_logits"]) < cfg["joint"]["auc_floor"]
            )
            return out
        except FloatingPointError as e:
            log.warning("%s; retrying once with lr x0.5", e)
    raise FloatingPointError("non-finite loss persisted after lr halving (hard stop)")
