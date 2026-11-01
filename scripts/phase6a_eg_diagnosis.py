"""Step 1: S2-EG diagnosis (validation only) and fix.

python scripts/phase6a_eg_diagnosis.py <phase2_dir> <phase6_dir>
For every eps and seed it reports, for the OLD setting (base-matched class weight, as in Phase 2)
and the FIXED setting (class weight matched to EG's own natural rate):
 (a) train-set EOD of the EG mixture before any thresholding (expected, using the pmf),
 (b) whether predictions use the randomized predictor (fraction of rows with pmf strictly in (0,1)),
 (c) natural-rate results (no flag-rate matching): flag rate, EOD, recall, precision, AUC-PR of the
     EG score, ECE of the S1 probability; and the shared-10%-rule result (what Phase 2 reported).
Writes eg_diagnosis.csv and the corrected arm folder s2_eg_v2/ (val_prob = S1 prob, val_flags =
shared rule on pmf + 1e-6 * S1 prob) into <phase6_dir>.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.tensors import load_arrays  # noqa: E402
from src.eval.metrics import compute_metrics  # noqa: E402
from src.models.nets import build_model  # noqa: E402
from src.posthoc.eg import fit_eg, match_eg_weight, pmf  # noqa: E402
from src.posthoc.phase5 import flags_at_rate  # noqa: E402
from src.run_arms import extract_z, match_flag_weight, save_json, save_predictions  # noqa: E402
from src.run_joint import load_setup  # noqa: E402
from src.utils import load_config  # noqa: E402

S = [0, 1, 2]
EPS = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
EPS_DEFAULT = 0.01


def soft_eod(y, a, p):
    gaps = [abs(p[(a == 0) & (y == t)].mean() - p[(a == 1) & (y == t)].mean()) for t in (1, 0)]
    return float(max(gaps))


def main() -> None:
    p2, p6 = Path(sys.argv[1]), Path(sys.argv[2])
    p6.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/train.yaml")
    fr, eb = cfg["evaluation"]["flag_rate"], cfg["evaluation"]["ece_bins"]
    device, train, val, cards, weights, _ = load_setup(cfg)
    hp = json.loads((p2 / "a2" / "tuning.json").read_text())["best_params"]
    ec = cfg["exponentiated_gradient"]
    rows = []
    for s in S:
        t0 = time.time()
        model = build_model("tabtransformer", train.cont.shape[1], cards, cfg["model_defaults"], hp)
        model.load_state_dict(torch.load(p2 / "a2" / f"seed_{s}" / "best.pt"))
        model.to(device)
        sc = StandardScaler().fit(extract_z(model, train, device))
        ztr = sc.transform(extract_z(model, train, device))
        zva = sc.transform(extract_z(model, val, device))
        s1p = np.load(p2 / "s1" / f"seed_{s}" / "val_prob.npy")
        w_old = match_flag_weight(ztr, train.y, fr, ec["logistic_max_iter"])
        w_new, tr_rate = match_eg_weight(
            ztr, train.y, train.a, fr, EPS_DEFAULT, ec["max_iter"], ec["logistic_max_iter"]
        )
        print(f"seed {s}: w_old {w_old:.3f}, w_new {w_new:.3f} (EG train rate {tr_rate:.4f}) "
              f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
        rng = np.random.default_rng(s)
        for version, w in (("old (base-matched w)", w_old), ("fixed (EG-matched w)", w_new)):
            for eps in EPS:
                eg = fit_eg(ztr, train.y, train.a, w, eps, ec["max_iter"], ec["logistic_max_iter"])
                ptr, pva = pmf(eg, ztr), pmf(eg, zva)
                score = pva + 1e-6 * s1p
                nat = rng.random(len(pva)) < pva  # one draw of the randomized predictor
                m_nat = compute_metrics(val.y, s1p, val.a, nat, eb)
                fl_sh = flags_at_rate(score, fr)
                m_sh = compute_metrics(val.y, s1p, val.a, fl_sh, eb)
                tp = float((nat & (val.y == 1)).sum())
                rows.append({
                    "version": version, "eps": eps, "seed": s, "w_pos": w,
                    "n_predictors": len(eg.predictors_), "oracle_calls": eg.n_oracle_calls_,
                    "best_gap": float(eg.best_gap_),
                    "train_soft_eod": soft_eod(train.y, train.a, ptr),
                    "frac_pmf_fractional": float(np.mean((pva > 1e-9) & (pva < 1 - 1e-9))),
                    "natural_flag_rate": float(nat.mean()),
                    "natural_soft_eod": soft_eod(val.y, val.a, pva),
                    "natural_eod": m_nat["eod"], "natural_recall": m_nat["recall"],
                    "natural_precision": tp / max(1.0, float(nat.sum())),
                    "natural_dir": m_nat["dir"],
                    "auc_pr_eg_score": float(average_precision_score(val.y, score)),
                    "auc_pr_s1": m_nat["auc_pr"], "ece_s1": m_nat["ece"],
                    "worst_group_ece_s1": m_nat["worst_group_ece"],
                    "shared10_flag_rate": float(fl_sh.mean()),
                    "shared10_eod": m_sh["eod"], "shared10_recall": m_sh["recall"],
                    "shared10_dir": m_sh["dir"],
                })  # fmt: skip
                if version.startswith("fixed") and eps == EPS_DEFAULT:
                    save_predictions(
                        p6 / "s2_eg_v2" / f"seed_{s}", s1p, fl_sh,
                        {"eps": eps, "w_pos": w, "version": "v2", "deployable": True,
                         "natural_flag_rate": float(nat.mean())},
                    )  # fmt: skip
        print(f"seed {s} done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(p6 / "eg_diagnosis.csv", index=False)
    df = pd.DataFrame(rows)
    g = df.groupby(["version", "eps"]).mean(numeric_only=True).reset_index()
    cols = ["version", "eps", "n_predictors", "train_soft_eod", "frac_pmf_fractional",
            "natural_flag_rate", "natural_eod", "natural_recall", "auc_pr_eg_score",
            "worst_group_ece_s1", "shared10_flag_rate", "shared10_eod", "shared10_recall"]  # fmt: skip
    pd.set_option("display.width", 250)
    print(g[cols].round(4).to_string())
    save_json(p6 / "eg_diagnosis_summary.json", g.to_dict("records"))


if __name__ == "__main__":
    main()
