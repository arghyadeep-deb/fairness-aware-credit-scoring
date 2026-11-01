"""Backbones, losses, SMOTE-NC and trainer sanity (small synthetic data, CPU-safe)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.tensors import Arrays, load_arrays, smote_nc, stratified_subsample
from src.losses.calibration import soft_group_ece
from src.losses.focal import focal_loss
from src.losses.weighted_ce import weighted_bce
from src.models.nets import build_model
from src.train import fit, make_loss, sample_hp
from src.utils import load_config

CFG = load_config("configs/train.yaml")
CARDS = [5, 7, 4]


def batch(b=64, n_cont=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    xc = torch.randn(b, n_cont, generator=g)
    xk = torch.stack([torch.randint(0, c, (b,), generator=g) for c in CARDS], 1)
    return xc, xk


@pytest.mark.parametrize("name", ["tabtransformer", "cnn1d", "mlp"])
def test_models_return_logit_and_z(name):
    m = build_model(name, 10, CARDS, CFG["model_defaults"], {})
    lg, z = m(*batch())
    assert lg.shape == (64,) and z.shape == (64, m.z_dim)
    assert torch.isfinite(lg).all()
    lg.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_tabtransformer_spec_defaults_and_zdim_64():
    m = build_model("tabtransformer", 10, CARDS, CFG["model_defaults"], {})
    assert m.z_dim == 64 and len(m.encoder.layers) == 3
    assert m.emb.tables[0].embedding_dim == 32
    m2 = build_model("tabtransformer", 10, CARDS, CFG["model_defaults"], {"emb_dim": 16})
    assert m2.emb.tables[0].embedding_dim == 16


def test_ce_equals_torch_bce_and_weights_scale_classes():
    lg = torch.randn(200)
    y = (torch.rand(200) < 0.3).float()
    assert weighted_bce(lg, y).item() == pytest.approx(
        F.binary_cross_entropy_with_logits(lg, y).item(), 1e-6
    )
    per = F.binary_cross_entropy_with_logits(lg, y, reduction="none")
    expect = (per * torch.where(y > 0.5, 5.0, 0.5)).mean()
    assert weighted_bce(lg, y, w_neg=0.5, w_pos=5.0).item() == pytest.approx(expect.item(), 1e-6)


def test_focal_reduces_to_scaled_bce_at_gamma_zero_and_downweights_easy_examples():
    lg = torch.randn(300)
    y = (torch.rand(300) < 0.3).float()
    f0 = focal_loss(lg, y, gamma=0.0, alpha_pos=0.5)
    assert f0.item() == pytest.approx(0.5 * F.binary_cross_entropy_with_logits(lg, y).item(), 1e-6)
    easy_l, easy_y = torch.tensor([8.0]), torch.tensor([1.0])
    assert focal_loss(easy_l, easy_y, 2.0, 0.5) < 1e-3 * focal_loss(easy_l, easy_y, 0.0, 0.5) + 1e-9


def test_soft_group_ece_zero_when_calibrated_and_positive_when_not():
    g = torch.Generator().manual_seed(0)
    p = torch.rand(200000, generator=g) * 0.9 + 0.05
    y = (torch.rand(200000, generator=g) < p).float()
    a = (torch.rand(200000, generator=g) < 0.5).long()
    lg = torch.log(p / (1 - p))
    assert soft_group_ece(lg, y, a).item() < 0.01
    assert soft_group_ece(lg + 2.0, y, a).item() > 0.15
    assert (
        soft_group_ece(lg, y, torch.zeros_like(a)).item() < 0.01
    )  # one group absent -> still defined


def test_soft_group_ece_is_differentiable():
    lg = torch.randn(500, requires_grad=True)
    soft_group_ece(lg, (torch.rand(500) < 0.2).float(), (torch.rand(500) < 0.5).long()).backward()
    assert lg.grad is not None and torch.isfinite(lg.grad).all()


def small_arrays(n=1500, seed=0):
    r = np.random.default_rng(seed)
    cont = r.normal(size=(n, 6)).astype("float32")
    cont[:, 5] = (r.random(n) < 0.4).astype("float32")  # a binary flag
    cat = np.stack([r.integers(0, c, n) for c in CARDS], 1).astype("int64")
    logit = -2.2 + 1.2 * cont[:, 0] - 0.8 * cont[:, 1] + 0.5 * (cat[:, 0] == 1)
    y = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    a = r.integers(0, 2, n)
    names = [f"f{i}" for i in range(6)]
    return Arrays(cont, cat, y, a, names, ["c0", "c1", "c2"], ["f5"])


def test_smote_nc_train_only_ratio_valid_categories_binary_flags():
    tr = small_arrays()
    out = smote_nc(tr, 0.5, 5, 0)
    assert len(out.y) > len(tr.y)
    assert (out.y[: len(tr.y)] == tr.y).all()  # originals first, unchanged
    assert out.y.mean() > tr.y.mean()
    assert out.y.sum() / (out.y == 0).sum() == pytest.approx(0.5, abs=0.01)
    assert all(out.cat[:, i].max() < c and out.cat[:, i].min() >= 0 for i, c in enumerate(CARDS))
    assert set(np.unique(out.cont[:, 5])) <= {0.0, 1.0}


def test_stratified_subsample_keeps_cells():
    arr = small_arrays(4000)
    sub = stratified_subsample(arr, 0.3, 0)
    assert len(sub.y) == 1200
    for t in (0, 1):
        for g in (0, 1):
            full = ((arr.y == t) & (arr.a == g)).mean()
            part = ((sub.y == t) & (sub.a == g)).mean()
            assert abs(full - part) < 0.01


def test_trainer_learns_signal_deterministically_and_stops_early():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {**CFG, "training": {**CFG["training"], "batch_size": 256, "patience": 3}}
    tr, va = small_arrays(6000, 0), small_arrays(2000, 1)
    arm = {"name": "t", "model": "mlp", "loss": "wce", "sampling": "none"}
    hp = {"lr": 3e-3, "weight_decay": 1e-5, "dropout": 0.0}
    w = {"0": 0.55, "1": 5.0}
    r1 = fit(arm, cfg, hp, tr, va, CARDS, w, 0, dev, max_epochs=12)
    r2 = fit(arm, cfg, hp, tr, va, CARDS, w, 0, dev, max_epochs=12)
    assert r1["best_val_auc_pr"] > va.y.mean() * 1.8  # beats the base rate by a wide margin
    assert (
        r1["history"][0]["train_loss"] > r1["history"][-1]["train_loss"] or len(r1["history"]) < 3
    )
    assert r1["best_val_auc_pr"] == pytest.approx(r2["best_val_auc_pr"], abs=1e-3)


def test_trainer_can_overfit_a_tiny_batch():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = small_arrays(200, 0)
    cfg = {**CFG, "training": {**CFG["training"], "batch_size": 200, "patience": 100}}
    arm = {"name": "t", "model": "mlp", "loss": "ce", "sampling": "none"}
    hp = {"lr": 5e-3, "weight_decay": 0.0, "dropout": 0.0}
    r = fit(arm, cfg, hp, tr, tr, CARDS, {"0": 1.0, "1": 1.0}, 0, dev, max_epochs=300)
    assert r["history"][-1]["train_loss"] < 0.25 * r["history"][0]["train_loss"]


def test_make_loss_dispatch_and_g1_adds_penalty():
    lg = torch.randn(128)
    y = (torch.rand(128) < 0.3).float()
    a = (torch.rand(128) < 0.5).long()
    w = {"0": 0.55, "1": 5.0}
    base = make_loss({"loss": "wce"}, CFG, {}, w, 0.08)(lg, y, a)
    g1 = make_loss({"loss": "wce_gcal"}, CFG, {}, w, 0.08)(lg, y, a)
    assert g1 > base
    assert make_loss({"loss": "focal"}, CFG, {"focal_gamma": 2.0}, w, 0.08)(lg, y, a).item() > 0
    with pytest.raises(ValueError):
        make_loss({"loss": "nope"}, CFG, {}, w, 0.08)


def test_search_space_size_equal_core_params_per_arm():
    import optuna

    study = optuna.create_study()
    sizes = {}
    for name, arm in {
        "a0": {"model": "tabtransformer", "loss": "ce"},
        "a3": {"model": "cnn1d", "loss": "wce"},
        "a5": {"model": "tabtransformer", "loss": "focal"},
    }.items():
        t = study.ask()
        sizes[name] = set(sample_hp(t, arm, CFG))
    assert {"lr", "weight_decay", "dropout"} <= set.intersection(*sizes.values())
    assert sizes["a3"] == {"lr", "weight_decay", "dropout"}
    assert "focal_gamma" in sizes["a5"] and "emb_dim" in sizes["a0"]


def test_phase2_cannot_read_test_split():
    with pytest.raises(PermissionError):
        load_arrays(CFG, "test")
