"""Phase 4: per-term normalisation, stratified sampler, joint trainer behaviour."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.sampler import stratified_epoch
from src.data.tensors import Arrays
from src.losses.joint import JointConfig, JointLoss
from src.utils import load_config

CFG = load_config("configs/train.yaml")
W = {"0": 0.54, "1": 6.2}
CARDS = [5, 7, 4]


def batch(b=2048, seed=0, shift=0.0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(b, generator=g) < 0.1).float()
    a = (torch.rand(b, generator=g) < 0.5).long()
    logit = scale * (1.5 * y + torch.randn(b, generator=g)) + shift * a
    return logit, y, a


def make(norm="initial", lf=1.0, lc=1.0, la=1.0, warm=2, ramp=2):
    cfg = JointConfig.from_config(CFG, lambda_f=lf, lambda_c=lc, lambda_a=la)
    cfg.warmup_epochs, cfg.ramp_epochs, cfg.normalize = warm, ramp, norm
    return JointLoss(cfg, W["0"], W["1"])


def adv_logit(a, noise=1.0, seed=3):
    g = torch.Generator().manual_seed(seed)
    return (a.float() * 2 - 1) * 0.8 + noise * torch.randn(len(a), generator=g)


def run_warmup(loss, scale, steps=6):
    for s in range(steps):
        lg, y, a = batch(seed=s, shift=0.5, scale=scale)
        loss.set_epoch(1.0 + s / steps)  # last warm-up epoch (warm=2)
        loss(lg, y, a, adv_logit(a))


def test_normalised_terms_start_at_about_one_even_when_scales_differ():
    """A term 10x larger in raw magnitude gets a 10x smaller weight, so lambdas are comparable."""
    a_loss, b_loss = make(), make()
    run_warmup(a_loss, scale=1.0)
    run_warmup(b_loss, scale=3.0)  # logits 3x larger: fairness/calibration raw values change a lot
    for loss in (a_loss, b_loss):
        loss.set_epoch(2.5)
        lg, y, a = batch(seed=99, shift=0.5, scale=3.0 if loss is b_loss else 1.0)
        _, parts = loss(lg, y, a, adv_logit(a))
        assert loss.ref is not None
        for k in ("fair", "cal", "adv"):
            # the normalised term at ramp start is O(1), whatever its raw scale
            assert 0.2 < float(parts[k]) / loss.ref[k] < 5.0
    # the raw terms really are on different scales, so normalising matters
    refs = sorted(a_loss.ref.values())
    assert refs[-1] / refs[0] > 1.3


def test_reference_is_frozen_after_the_ramp_starts():
    loss = make()
    run_warmup(loss, 1.0)
    loss.set_epoch(2.2)
    lg, y, a = batch(seed=7)
    loss(lg, y, a, adv_logit(a))
    ref = dict(loss.ref)
    for scale in (0.2, 5.0):  # later batches with very different magnitudes
        loss.set_epoch(3.5)
        lg, y, a = batch(seed=8, scale=scale)
        loss(lg, y, a, adv_logit(a))
    assert loss.ref == ref


def test_weight_of_a_term_is_lambda_over_reference_after_ramp():
    loss = make(lf=0.5, lc=2.0, la=1.0, warm=2, ramp=2)
    run_warmup(loss, 1.0)
    loss.set_epoch(50)
    lg, y, a = batch(seed=5)
    ad = adv_logit(a)
    total, p = loss(lg, y, a, ad)
    exp = (
        p["wce"]
        + 0.5 * p["fair"] / loss.ref["fair"]
        + 2.0 * p["cal"] / loss.ref["cal"]
        + p["adv"] / loss.ref["adv"]
    )
    assert total.item() == pytest.approx(exp.item(), rel=1e-5)


def test_warmup_loss_and_gradient_unchanged_by_normalisation_observation():
    """During warm-up the observation is no_grad: value and gradients equal plain weighted CE."""
    from src.losses.weighted_ce import weighted_bce

    loss = make()
    lg, y, a = batch()
    lg = lg.clone().requires_grad_(True)
    loss.set_epoch(1.5)
    total, _ = loss(lg, y, a, adv_logit(a))
    total.backward()
    g1 = lg.grad.clone()
    lg2 = lg.detach().clone().requires_grad_(True)
    weighted_bce(lg2, y, W["0"], W["1"]).backward()
    assert total.item() == pytest.approx(weighted_bce(lg2, y, W["0"], W["1"]).item(), rel=1e-6)
    assert torch.allclose(g1, lg2.grad, atol=1e-7)


def test_lambda_zero_equals_A2_with_normalisation_on():
    from src.losses.weighted_ce import weighted_bce

    loss = make(lf=0.0, lc=0.0, la=0.0)
    for epoch in (0.0, 2.5, 10.0):
        loss.set_epoch(epoch)
        lg, y, a = batch(seed=2)
        total, parts = loss(lg, y, a, adv_logit(a))
        assert total.item() == pytest.approx(weighted_bce(lg, y, W["0"], W["1"]).item(), abs=0)
        assert list(parts) == ["wce"]


def test_no_warmup_falls_back_to_first_ramp_batch_and_floor_applies():
    loss = make(warm=0, ramp=1)
    loss.set_epoch(0.5)
    lg, y, a = batch(seed=1)
    loss(lg, y, a, adv_logit(a))
    assert all(v >= loss.cfg.ref_floor for v in loss.ref.values())
    const = make(warm=0, ramp=1)  # identical predictions: fairness term is exactly 0 -> floored
    const.set_epoch(0.5)
    _, y, a = batch(seed=1)
    const(torch.zeros_like(y), y, a, adv_logit(a))
    assert const.ref["fair"] == pytest.approx(const.cfg.ref_floor)


def test_normalize_none_is_the_phase3_behaviour():
    loss = make(norm="none")
    loss.set_epoch(50)
    lg, y, a = batch()
    total, p = loss(lg, y, a, adv_logit(a))
    assert loss.ref is None
    assert total.item() == pytest.approx(
        (p["wce"] + p["fair"] + p["cal"] + p["adv"]).item(), rel=1e-5
    )


# ---------- stratified sampler ----------
def test_stratified_epoch_guarantees_min_per_cell_and_keeps_natural_mix():
    g = torch.Generator().manual_seed(0)
    n = 40000
    y = (torch.rand(n, generator=g) < 0.05).long()  # rare positives
    a = (torch.rand(n, generator=g) < 0.35).long()
    idx = stratified_epoch(y, a, 2048, 64, g)
    assert idx.shape == (n // 2048, 2048)
    cell = (y * 2 + a)[idx]
    for c in range(4):
        assert (cell == c).sum(dim=1).min() >= 64
    nat = torch.bincount(cell.flatten(), minlength=4).float() / cell.numel()
    ref = torch.bincount(y * 2 + a, minlength=4).float() / n
    assert nat[3] == pytest.approx(64 / 2048, abs=1e-4)  # smallest cell lifted to the floor
    assert torch.allclose(nat[:3], ref[:3], atol=0.02)  # the others keep their natural mix
    # rows drawn without replacement inside an epoch for the large cells
    big = idx[cell == 0]
    assert len(torch.unique(big)) == len(big)


def test_stratified_epoch_oversamples_a_tiny_cell_to_the_floor():
    g = torch.Generator().manual_seed(1)
    n = 20000
    y = torch.zeros(n, dtype=torch.long)
    a = torch.zeros(n, dtype=torch.long)
    a[10000:] = 1
    y[:100] = 1  # (y=1, a=0): 100 rows, natural share ~5 per batch of 1024
    y[10000:10100] = 1  # (y=1, a=1): 100 rows
    idx = stratified_epoch(y, a, 1024, 64, g)
    cell = (y * 2 + a)[idx]
    assert idx.shape[0] == n // 1024
    assert (cell == 2).sum(dim=1).min() >= 64 and (cell == 3).sum(dim=1).min() >= 64


def test_empty_cell_raises():
    y = torch.zeros(1000, dtype=torch.long)
    a = torch.zeros(1000, dtype=torch.long)
    with pytest.raises(ValueError):
        stratified_epoch(y, a, 128, 8, torch.Generator().manual_seed(0))


# ---------- trainer on a tiny synthetic problem ----------
def tiny_arrays(n=6000, seed=0):
    r = np.random.default_rng(seed)
    a = (r.random(n) < 0.4).astype(int)
    x = r.normal(size=(n, 8)).astype("float32")
    logit = -2.2 + 0.9 * x[:, 0] - 0.6 * x[:, 1] + 0.5 * a
    y = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cat = np.stack([r.integers(0, c, n) for c in CARDS], 1).astype("int64")
    names = [f"f{i}" for i in range(8)]
    return Arrays(x, cat, y, a, names, ["c0", "c1", "c2"], [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer uses CUDA")
def test_fit_joint_runs_logs_gender_head_and_selects_only_after_ramp():
    from src.train_joint import fit_joint

    dev = torch.device("cuda")
    tr, va = tiny_arrays(8000, 0), tiny_arrays(3000, 1)
    cfg = dict(CFG)
    cfg["training"] = {**CFG["training"], "batch_size": 512}
    hp = {"lr": 2e-3, "weight_decay": 1e-5, "dropout": 0.1, "emb_dim": 16}
    res = fit_joint(
        cfg,
        hp,
        tr,
        va,
        CARDS,
        W,
        0,
        dev,
        (1.0, 1.0, 1.0),
        max_epochs=7,
        schedule={"warmup_epochs": 1, "ramp_epochs": 2},
    )
    h = res["history"]
    assert len(h) >= 3 and {"adv_head_auc", "val_eod", "val_worst_group_ece"} <= set(h[0])
    assert res["best_epoch"] >= 2  # first epoch ending at full strength = ceil(1+2)-1
    assert res["norm_reference"] is not None and not res["below_auc_floor"]
    assert [x["ramp"] for x in h][0] == 0.0 and h[-1]["ramp"] == 1.0
