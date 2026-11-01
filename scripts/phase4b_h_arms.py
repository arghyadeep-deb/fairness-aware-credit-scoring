"""H1-H3 (declared in RUN_LOG commit 2ef4769): joint training with lambda_a = lambda_c = 0 and
lambda_f in {0.05, 0.1, 0.25}, then the S1 analytic prior correction. Tuned-JT hyper-parameters,
3 seeds, validation only.   python scripts/phase4b_h_arms.py <phase4b_dir>
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.posthoc.prior_correction import correct_logits  # noqa: E402
from src.run_arms import flags_from_shared_rule, save_predictions, sigmoid  # noqa: E402
from src.run_joint import load_setup, run_point, tag  # noqa: E402
from src.utils import load_config  # noqa: E402

H = {"h1": 0.05, "h2": 0.1, "h3": 0.25}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = Path(sys.argv[1])
    cfg = load_config("configs/train.yaml")
    device, train, val, cards, weights, _ = load_setup(cfg)
    hp = json.loads((out / "jt_tuning.json").read_text())["best_params"]
    setup = (device, train, val, cards, weights, hp)
    for name, lf in H.items():
        for s in cfg["sweep"]["seeds"]:
            run_point(cfg, out, (lf, 0.0, 0.0), s, setup, name=f"{name}_raw")
            d = out / f"{name}_raw" / tag((lf, 0.0, 0.0)) / f"seed_{s}"
            prob = sigmoid(correct_logits(np.load(d / "val_logits.npy"), weights))
            flags, thr = flags_from_shared_rule(prob, cfg["evaluation"]["flag_rate"])
            save_predictions(out / name / f"seed_{s}", prob, flags,
                             {"threshold": thr, "lambda_f": lf, "source": f"{name}_raw + S1 correction",
                              "deployable": True})  # fmt: skip
    print("done")


if __name__ == "__main__":
    main()
