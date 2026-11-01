"""Exploration for the S2-EG diagnosis (seed 0, validation only): what does EG do at each eps?"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.tensors import load_arrays  # noqa: E402
from src.models.nets import build_model  # noqa: E402
from src.run_arms import extract_z, match_flag_weight  # noqa: E402
from src.run_joint import load_setup  # noqa: E402
from src.utils import load_config  # noqa: E402


def eod_soft(y, a, p):
    out = []
    for t in (1, 0):
        r = [p[(a == g) & (y == t)].mean() for g in (0, 1)]
        out.append(abs(r[0] - r[1]))
    return max(out)


def main() -> None:
    p2 = Path("artifacts/runs/20261101T055729Z_phase2")
    cfg = load_config("configs/train.yaml")
    device, train, val, cards, weights, _ = load_setup(cfg)
    hp = json.loads((p2 / "a2" / "tuning.json").read_text())["best_params"]
    ec = cfg["exponentiated_gradient"]
    model = build_model("tabtransformer", train.cont.shape[1], cards, cfg["model_defaults"], hp)
    model.load_state_dict(torch.load(p2 / "a2" / "seed_0" / "best.pt"))
    model.to(device)
    sc = StandardScaler().fit(extract_z(model, train, device))
    ztr = sc.transform(extract_z(model, train, device))
    zva = sc.transform(extract_z(model, val, device))
    w_pos = match_flag_weight(ztr, train.y, cfg["evaluation"]["flag_rate"], ec["logistic_max_iter"])
    print("w_pos", w_pos, "max_iter in config", ec["max_iter"])
    base = LogisticRegression(max_iter=ec["logistic_max_iter"], class_weight={0: 1.0, 1: w_pos})
    base.fit(ztr, train.y)
    h = base.predict(ztr)
    print("base classifier alone: train flag rate", h.mean().round(4), "train EOD(hard)",
          round(eod_soft(train.y, train.a, h.astype(float)), 4),
          "val EOD(hard)", round(eod_soft(val.y, val.a, base.predict(zva).astype(float)), 4))  # fmt: skip
    for eps in [0.002, 0.01, 0.08, 0.3]:
        for mi in (50, 200):
            eg = ExponentiatedGradient(
                LogisticRegression(max_iter=ec["logistic_max_iter"], class_weight={0: 1.0, 1: w_pos}),
                constraints=EqualizedOdds(), eps=eps, max_iter=mi,
            )  # fmt: skip
            eg.fit(ztr, train.y, sensitive_features=train.a)
            ptr, pva = eg._pmf_predict(ztr)[:, 1], eg._pmf_predict(zva)[:, 1]
            print(f"eps {eps:<6} max_iter {mi:<3} oracle_calls {eg.n_oracle_calls_:<3} "
                  f"best_iter {eg.best_iter_} last_iter {eg.last_iter_} best_gap {eg.best_gap_:.5f} "
                  f"predictors {len(eg.predictors_)} | train soft-EOD {eod_soft(train.y, train.a, ptr):.4f} "
                  f"val soft-EOD {eod_soft(val.y, val.a, pva):.4f} natural rate {pva.mean():.4f} "
                  f"frac(pmf in {{0,1}}) {np.mean((pva < 1e-9) | (pva > 1 - 1e-9)):.3f}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
