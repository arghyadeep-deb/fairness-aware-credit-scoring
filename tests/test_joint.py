"""Stage 4 unit tests (Phase 3): GRL, fairness surrogate, calibration surrogate, joint loss."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.losses.calibration import soft_group_ece
from src.losses.fairness import fairness_loss, soft_flags
from src.losses.grl import GradientReversal, grad_reverse
from src.losses.joint import JointConfig, JointLoss
from src.losses.weighted_ce import weighted_bce
from src.models.heads import AdversaryHead, JointModel
from src.models.nets import build_model
from src.train import make_loss
from src.utils import load_config, resolve

CFG = load_config("configs/train.yaml")
W = json.loads((resolve(CFG["paths"]["prep_artifacts"]) / "class_weights.json").read_text())
CARDS = [5, 7, 4]


def toy(b=256, seed=0, n_cont=8):
    g = torch.Generator().manual_seed(seed)
    xc = torch.randn(b, n_cont, generator=g)
    xk = torch.stack([torch.randint(0, c, (b,), generator=g) for c in CARDS], 1)
    y = (torch.rand(b, generator=g) < 0.2).float()
    a = (torch.rand(b, generator=g) < 0.5).long()
    return xc, xk, y, a


def jl(lf=0.0, lc=0.0, la=0.0, **kw):
    cfg = JointConfig.from_config(CFG, lambda_f=lf, lambda_c=lc, lambda_a=la)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return JointLoss(cfg, W["0"], W["1"])


# ---------- GRL ----------
@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0, 2.5])
def test_grl_forward_identity_backward_negated_scaled(lam):
    x = torch.randn(32, 6, requires_grad=True)
    out = grad_reverse(x, lam)
    assert torch.equal(out, x)  # forward is the identity
    w = torch.randn(32, 6)
    (out * w).sum().backward()
    assert torch.allclose(x.grad, -lam * w)  # backward multiplies by -lambda


def test_grl_module_lambda_can_change_between_steps():
    layer = GradientReversal(1.0)
    x = torch.ones(4, 3, requires_grad=True)
    layer(x).sum().backward()
    assert torch.allclose(x.grad, -torch.ones(4, 3))
    layer.lambd = 0.5
    x.grad = None
    layer(x).sum().backward()
    assert torch.allclose(x.grad, -0.5 * torch.ones(4, 3))


def test_grl_reverses_trunk_gradient_through_a_real_head():
    torch.manual_seed(0)
    trunk, head = nn.Linear(5, 4), nn.Linear(4, 1)
    x, t = torch.randn(64, 5), torch.rand(64, 1)

    def trunk_grad(use_grl):
        trunk.zero_grad()
        z = trunk(x)
        z = grad_reverse(z, 1.0) if use_grl else z
        F.mse_loss(head(z), t).backward()
        return trunk.weight.grad.clone()

    assert torch.allclose(trunk_grad(True), -trunk_grad(False), atol=1e-6)


# ---------- fairness surrogate ----------
def test_fairness_zero_when_both_groups_get_identical_predictions():
    _, _, y, a = toy(4000)
    constant = torch.zeros(4000)
    for kind in ("eo", "dp"):
        assert fairness_loss(constant, y, a, kind).item() == pytest.approx(0.0, abs=1e-6)


def test_fairness_near_zero_for_group_independent_scores_large_otherwise():
    g = torch.Generator().manual_seed(1)
    n = 40000
    y = (torch.rand(n, generator=g) < 0.2).float()
    a = (torch.rand(n, generator=g) < 0.5).long()
    fair = 2.0 * y + torch.randn(n, generator=g)  # depends on y only
    biased = fair + 1.5 * a  # group 1 scored higher
    for kind in ("eo", "dp"):
        assert fairness_loss(fair, y, a, kind).item() < 0.03
        assert fairness_loss(biased, y, a, kind).item() > 0.10
    # the soft dp gap tracks the hard flag-rate gap at the same threshold
    hard = (biased > torch.quantile(biased, 0.90)).float()
    hard_gap = (hard[a == 0].mean() - hard[a == 1].mean()).abs().item()
    assert fairness_loss(biased, y, a, "dp").item() == pytest.approx(hard_gap, abs=0.03)


def test_fairness_differentiable_and_descent_reduces_gap():
    g = torch.Generator().manual_seed(2)
    n = 4000
    y = (torch.rand(n, generator=g) < 0.2).float()
    a = (torch.rand(n, generator=g) < 0.5).long()
    lg = (2.0 * y + torch.randn(n, generator=g) + 1.5 * a).requires_grad_(True)
    opt = torch.optim.SGD([lg], lr=50.0)
    first = fairness_loss(lg, y, a).item()
    for _ in range(60):
        opt.zero_grad()
        loss = fairness_loss(lg, y, a)
        loss.backward()
        assert torch.isfinite(lg.grad).all()
        opt.step()
    assert fairness_loss(lg, y, a).item() < 0.6 * first


def test_fairness_missing_group_returns_zero_not_error():
    lg = torch.randn(50)
    y = (torch.rand(50) < 0.3).float()
    assert fairness_loss(lg, y, torch.zeros(50, dtype=torch.long)).item() == 0.0


def test_soft_flags_hit_flag_rate_and_scale_free():
    lg = torch.randn(5000)
    assert soft_flags(lg, 0.10, 0.1).mean().item() == pytest.approx(0.10, abs=0.03)
    assert torch.allclose(
        soft_flags(lg, 0.10, 0.1), soft_flags(7.0 * lg + 3.0, 0.10, 0.1), atol=1e-4
    )


def test_unknown_fairness_type_raises():
    with pytest.raises(ValueError):
        fairness_loss(torch.randn(10), torch.ones(10), torch.zeros(10), kind="nope")


# ---------- calibration surrogate ----------
def test_calibration_loss_near_zero_on_perfectly_calibrated_data_and_positive_otherwise():
    g = torch.Generator().manual_seed(0)
    n = 200000
    p = torch.rand(n, generator=g) * 0.9 + 0.05
    y = (torch.rand(n, generator=g) < p).float()
    a = (torch.rand(n, generator=g) < 0.5).long()
    lg = torch.log(p / (1 - p))
    assert soft_group_ece(lg, y, a).item() < 0.01
    assert soft_group_ece(lg + 1.5, y, a).item() > 0.1


# ---------- joint loss ----------
def test_all_lambdas_zero_joint_loss_equals_A2_loss_value_and_gradient():
    xc, xk, y, a = toy()
    torch.manual_seed(0)
    model = JointModel(build_model("tabtransformer", 8, CARDS, CFG["model_defaults"], {}))
    model.eval()  # dropout off so both passes see the same function
    a2 = make_loss({"loss": "wce"}, CFG, {}, W, 0.08)
    for epoch in (0, 4, 20):  # warm-up, mid-ramp, fully ramped: still all lambdas 0
        loss = jl()
        loss.set_epoch(epoch)
        logit, _, adv = model(xc, xk)
        total, parts = loss(logit, y, a, adv)
        model.zero_grad()
        total.backward()
        g_joint = [p.grad.clone() for p in model.backbone.parameters() if p.grad is not None]
        logit2 = model.backbone(xc, xk)[0]
        ref = a2(logit2, y, a)
        model.zero_grad()
        ref.backward()
        g_ref = [p.grad.clone() for p in model.backbone.parameters() if p.grad is not None]
        assert torch.equal(total, ref) and set(parts) == {"wce"}
        assert all(torch.equal(u, v) for u, v in zip(g_joint, g_ref))


def test_joint_loss_is_weighted_sum_of_terms_after_ramp():
    xc, xk, y, a = toy()
    model = JointModel(build_model("mlp", 8, CARDS, CFG["model_defaults"], {}))
    lf, lc, la = 0.7, 1.3, 0.4
    loss = jl(lf, lc, la, normalize="none")
    loss.set_epoch(100)
    logit, _, adv = model(xc, xk)
    total, parts = loss(logit, y, a, adv)
    expected = parts["wce"] + lf * parts["fair"] + lc * parts["cal"] + la * parts["adv"]
    assert total.item() == pytest.approx(expected.item(), rel=1e-5)
    assert parts["wce"].item() == pytest.approx(weighted_bce(logit, y, W["0"], W["1"]).item(), 1e-6)


def test_warmup_ramp_schedule():
    loss = jl(2.0, 1.0, 0.5)
    seen = {}
    for e in (0, 1.5, 2.99, 3, 5.5, 8, 50):
        loss.set_epoch(e)
        seen[e] = loss.lambdas()
    assert all(v == 0 for e in (0, 1.5, 2.99) for v in seen[e].values())
    assert seen[3]["f"] == 0  # ramp starts at 0 right after warm-up
    assert seen[5.5]["f"] == pytest.approx(2.0 * 2.5 / 5)  # linear
    assert seen[8] == {"f": 2.0, "c": 1.0, "a": 0.5} and seen[50] == seen[8]
    ramp = [loss.set_epoch(e) or loss.ramp() for e in (3, 4, 5, 6, 7, 8)]
    assert ramp == sorted(ramp) and ramp[0] == 0 and ramp[-1] == 1


def test_adversary_inactive_during_warmup_and_trunk_matches_A2():
    xc, xk, y, a = toy()
    torch.manual_seed(0)
    model = JointModel(build_model("tabtransformer", 8, CARDS, CFG["model_defaults"], {}))
    model.eval()
    loss = jl(1.0, 1.0, 1.0)
    loss.set_epoch(1)  # inside warm-up
    logit, _, adv = model(xc, xk)
    total, parts = loss(logit, y, a, adv)
    total.backward()
    assert set(parts) == {"wce"}  # fairness, calibration and adversary terms not even computed
    assert all(
        p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.adversary.parameters()
    )
    g = [p.grad.clone() for p in model.backbone.parameters() if p.grad is not None]
    model.zero_grad()
    weighted_bce(model.backbone(xc, xk)[0], y, W["0"], W["1"]).backward()
    assert all(
        torch.equal(u, p.grad)
        for u, p in zip(g, [q for q in model.backbone.parameters() if q.grad is not None])
    )


def test_adversary_active_after_warmup_and_reverses_trunk_gradient():
    xc, xk, y, a = toy()
    torch.manual_seed(0)
    model = JointModel(build_model("mlp", 8, CARDS, CFG["model_defaults"], {"dropout": 0.0}))
    model.eval()

    def trunk_grad(epoch, la):
        loss = jl(la=la)
        loss.set_epoch(epoch)
        model.zero_grad()
        logit, _, adv = model(xc, xk)
        total, parts = loss(logit, y, a, adv)
        total.backward()
        return model.backbone.mlp[0].weight.grad.clone(), parts

    g_warm, p_warm = trunk_grad(1, 1.0)
    g_on, p_on = trunk_grad(20, 1.0)
    assert "adv" not in p_warm and "adv" in p_on
    assert float(model.adversary.net[0].weight.grad.abs().sum()) > 0  # head now learns
    assert not torch.allclose(g_warm, g_on)
    # trunk gradient of the adversarial term alone is the negative of the plain-head gradient
    model.zero_grad()
    logit, z, adv = model(xc, xk)
    F.binary_cross_entropy_with_logits(adv, a.float()).backward()
    reversed_g = model.backbone.mlp[0].weight.grad.clone()
    model.zero_grad()
    z2 = model.backbone(xc, xk)[1]
    plain = nn.Sequential(model.adversary.net)  # same head weights, no reversal
    F.binary_cross_entropy_with_logits(plain(z2).squeeze(-1), a.float()).backward()
    assert torch.allclose(reversed_g, -model.backbone.mlp[0].weight.grad, atol=1e-6)


def test_adversary_term_requires_adversary_logit():
    loss = jl(la=1.0)
    loss.set_epoch(20)
    with pytest.raises(ValueError):
        loss(torch.randn(8), torch.ones(8), torch.zeros(8, dtype=torch.long), None)


def test_inference_path_does_not_touch_head_b():
    xc, xk, _, _ = toy(16)
    model = JointModel(build_model("mlp", 8, CARDS, CFG["model_defaults"], {}))
    model.eval()
    ref = model.backbone(xc, xk)[0]
    for p in model.adversary.parameters():
        nn.init.constant_(p, 123.0)  # wreck Head B
    assert torch.equal(model.predict(xc, xk), ref)
    assert isinstance(model.adversary, AdversaryHead) and model.adversary.net[0].in_features == 64


def test_joint_loss_overfits_a_200_row_batch_to_near_zero():
    torch.manual_seed(0)
    xc, xk, y, a = toy(200, n_cont=8)
    y[:12] = 1.0  # make sure there are positives in both groups
    a[:12] = torch.tensor([0, 1] * 6)
    model = JointModel(build_model("mlp", 8, CARDS, CFG["model_defaults"], {"dropout": 0.0}))
    loss = jl(
        lf=1.0, lc=1.0, la=0.0, warmup_epochs=0, ramp_epochs=0, normalize="none"
    )  # fair + cal on
    loss.set_epoch(0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = None
    for _ in range(600):
        opt.zero_grad()
        logit, _, adv = model(xc, xk)
        total, parts = loss(logit, y, a, adv)
        total.backward()
        opt.step()
        first = first if first is not None else total.item()
    assert total.item() < 0.02 and total.item() < 0.05 * first
