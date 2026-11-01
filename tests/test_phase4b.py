"""Phase 4b: GRL rescue variants (clip, Ganin ramp, smoothing) and Phase 5 helpers."""

import math

import numpy as np
import torch

from src.losses.grl import grad_reverse
from src.losses.joint import JointConfig, JointLoss
from src.posthoc.phase5 import gt_fraction_flags, pergroup_calibrate


def _grad(clip: float, scale: float = 1.0):
    x = (torch.randn(64, 8) * 3).requires_grad_(True)
    stats: list = []
    y = grad_reverse(x, 1.0, clip, 1.0, stats)
    (y * torch.randn(64, 8) * scale).sum().backward()
    return x.grad, stats


def test_grl_defaults_reverse_exactly():
    x = torch.randn(5, 4, requires_grad=True)
    w = torch.randn(5, 4)
    grad_reverse(x, 0.7).mul(w).sum().backward()
    assert torch.allclose(x.grad, -0.7 * w)


def test_grl_clip_bounds_row_norm_and_logs_unclipped_norm():
    g, stats = _grad(clip=0.5, scale=1.0)
    # weight = 1, B = 64: the unweighted per-row norm is norm * B, clipped to 0.5
    assert torch.all(g.norm(dim=1) * 64 <= 0.5 + 1e-5)
    assert len(stats) == 1 and stats[0] > 0


def test_ganin_ramp_values_and_default_linear():
    cfg = JointConfig(lambda_a=1.0, warmup_epochs=1, ramp_epochs=2, adv_ramp="ganin")
    loss = JointLoss(cfg, 0.5, 5.0)
    loss.set_epoch(1.0)
    assert loss.lambdas()["a"] == 0.0
    loss.set_epoch(3.0)
    assert abs(loss.lambdas()["a"] - (2 / (1 + math.exp(-10)) - 1)) < 1e-9
    loss.set_epoch(1.2)  # r = 0.1: ganin is above linear
    assert loss.lambdas()["a"] > 0.1
    lin = JointLoss(JointConfig(lambda_a=1.0, warmup_epochs=1, ramp_epochs=2), 0.5, 5.0)
    lin.set_epoch(2.0)
    assert abs(lin.lambdas()["a"] - 0.5) < 1e-9


def test_label_smoothing_changes_adversary_term_only():
    logit, y = torch.zeros(6), torch.tensor([0, 1, 0, 1, 0, 1.0])
    a, adv = torch.tensor([0, 1, 0, 1, 1, 0]), torch.randn(6)
    base = JointLoss(JointConfig(lambda_a=1.0, warmup_epochs=0, ramp_epochs=0), 0.5, 5.0)
    sm = JointLoss(
        JointConfig(lambda_a=1.0, warmup_epochs=0, ramp_epochs=0, adv_smooth=0.3), 0.5, 5.0
    )
    t0 = base._raw_terms(logit, y, a, adv, ["adv"])["adv"]
    t1 = sm._raw_terms(logit, y, a, adv, ["adv"])["adv"]
    assert not torch.isclose(t0, t1)
    t_half = JointLoss(JointConfig(adv_smooth=1.0), 0.5, 5.0)._raw_terms(logit, y, a, adv, ["adv"])
    assert torch.isclose(
        t_half["adv"],
        torch.nn.functional.binary_cross_entropy_with_logits(adv, torch.full((6,), 0.5)),
    )


def test_pergroup_calibration_is_cross_fitted_and_valid():
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.integers(0, 2, n)
    lg = rng.normal(-2.0 + 0.5 * a, 1.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(lg - 1.0)))).astype(int)
    for kind in ("platt", "isotonic"):
        p = pergroup_calibrate(y, lg, a, kind, seed=1)
        assert p.shape == (n,) and p.min() >= 0 and p.max() <= 1 + 1e-6
        assert not np.allclose(p, 1 / (1 + np.exp(-lg)))
        assert np.abs(p.mean() - y.mean()) < 0.03  # recalibrated towards the base rate


def test_gt_fraction_endpoints():
    p = np.array([0.1, 0.5, 0.9, 0.3])
    a = np.array([0, 1, 0, 1])
    thr = {"threshold_f": 0.8, "threshold_m": 0.2}
    assert np.array_equal(gt_fraction_flags(p, a, thr, 0.4, 0.0), p >= 0.4)
    assert np.array_equal(gt_fraction_flags(p, a, thr, 0.4, 1.0), p >= np.where(a == 0, 0.8, 0.2))
