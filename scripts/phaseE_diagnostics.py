"""Section E diagnostics (VALIDATION ONLY; declared after the test run; not added to the test table).

python scripts/phaseE_diagnostics.py <out_dir>

JT-W (importance-weighted ECE term) is conditional on a sampler mismatch. `artifacts/e_sampler_check.py`
printed batch default rate 0.0806 vs population 0.0807 and gender share 0.3418 vs 0.3416 (floor 64 never binds),
so the importance weights equal 1.0 and JT-W would be identical to JT chosen: not run (recorded in the report).

JT-A+PH: JT-A (spec GRL, adv_steps 5, lambda_a 0.25, the variant with the lowest fresh z-probe) with
lambda_f 0.05 and lambda_c 0, then S1 prior correction and S2-EG v2 (EG on frozen z, class weight matched to the
EG mixture's own flag rate, eps 0.01, shared 10% rule). 3 seeds. Compared with JT chosen and S2-EG v2 on
validation with paired stratified bootstrap CIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.eval.bootstrap import bootstrap_metrics  # noqa: E402
from src.eval.metrics import METRIC_NAMES, compute_metrics  # noqa: E402
from src.models.heads import JointModel  # noqa: E402
from src.models.nets import build_model  # noqa: E402
from src.posthoc.eg import fit_eg, match_eg_weight, pmf  # noqa: E402
from src.posthoc.phase5 import flags_at_rate  # noqa: E402
from src.posthoc.prior_correction import prior_shift  # noqa: E402
from src.run_arms import flags_from_shared_rule, save_json, sigmoid  # noqa: E402
from src.run_joint import load_setup, run_point  # noqa: E402
from src.train_joint import predict_joint  # noqa: E402
from src.utils import load_config  # noqa: E402

SEEDS = [0, 1, 2]
LAM = (0.05, 0.0, 0.25)
N_BOOT = 500
EPS = 0.01


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    p6 = ROOT / Path((ROOT / "artifacts" / "phase6_rundir.txt").read_text().strip())
    cfg = load_config("configs/train.yaml")
    fr, eb = cfg["evaluation"]["flag_rate"], cfg["evaluation"]["ece_bins"]
    setup = load_setup(cfg)
    device, train, val, cards, weights, hp = setup
    ec = cfg["exponentiated_gradient"]
    shift = prior_shift(weights)
    arms: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        "JT chosen": {},
        "S2-EG": {},
        "JT-A+PH raw": {},
        "JT-A+PH + S1": {},
        "JT-A+PH + S1 + S2-EG": {},
    }
    extra = []
    for s in SEEDS:
        # references (validation predictions already on disk)
        d = p6 / "jta_ctl" / "f0p05_c2_a0" / f"seed_{s}"
        pj = np.load(d / "val_prob.npy")
        arms["JT chosen"][s] = (pj, flags_from_shared_rule(pj, fr)[0])
        d = p6 / "s2_eg_v2" / f"seed_{s}"
        arms["S2-EG"][s] = (np.load(d / "val_prob.npy"), np.load(d / "val_flags.npy").astype(bool))
        # JT-A+PH: train
        run_point(cfg, out, LAM, s, setup, name="jtaph", schedule={"adv_steps": 5})
        rd = out / "jtaph" / "f0p05_c0_a0p25" / f"seed_{s}"
        raw_logit = np.load(rd / "val_logits.npy").astype("float64")
        p_raw = sigmoid(raw_logit)
        arms["JT-A+PH raw"][s] = (p_raw, flags_from_shared_rule(p_raw, fr)[0])
        p_s1 = sigmoid(raw_logit - shift)
        arms["JT-A+PH + S1"][s] = (p_s1, flags_from_shared_rule(p_s1, fr)[0])
        # S2-EG on the JT-A+PH z
        backbone = build_model("tabtransformer", train.cont.shape[1], cards, cfg["model_defaults"], hp)
        model = JointModel(backbone, cfg["joint"]["adversary_hidden"]).to(device)
        model.load_state_dict(torch.load(rd / "best.pt"))
        ztr0, zva0 = predict_joint(model, train, device)[2], predict_joint(model, val, device)[2]
        sc = StandardScaler().fit(ztr0)
        ztr, zva = sc.transform(ztr0), sc.transform(zva0)
        w, tr_rate = match_eg_weight(ztr, train.y, train.a, fr, EPS, ec["max_iter"], ec["logistic_max_iter"])
        eg = fit_eg(ztr, train.y, train.a, w, EPS, ec["max_iter"], ec["logistic_max_iter"])
        score = pmf(eg, zva) + 1e-6 * p_s1
        arms["JT-A+PH + S1 + S2-EG"][s] = (p_s1, flags_at_rate(score, fr))
        extra.append({"seed": s, "eg_w_pos": w, "eg_train_rate": tr_rate, "n_predictors": len(eg.predictors_)})
        print(f"seed {s} done", flush=True)
        del model, backbone, ztr, zva, ztr0, zva0
        torch.cuda.empty_cache()
    # point metrics (mean over seeds) and paired bootstrap
    rows, boots = [], {}
    for name, per in arms.items():
        ms = [compute_metrics(val.y, per[s][0], val.a, per[s][1], eb) for s in SEEDS]
        rows.append({"arm": name, **{k: float(np.mean([m[k] for m in ms])) for k in ms[0] if isinstance(ms[0][k], float)}})
        boots[name] = np.nanmean(
            np.stack([bootstrap_metrics(val.y, per[s][0], val.a, per[s][1], N_BOOT, s, eb, n_jobs=4, chunk_size=100) for s in SEEDS]),
            axis=0,
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "diag_validation_table.csv", index=False)
    diffs = []
    for name in ("JT-A+PH raw", "JT-A+PH + S1", "JT-A+PH + S1 + S2-EG", "JT chosen"):
        for ref in ("JT chosen", "S2-EG"):
            if name == ref or (name == "JT chosen" and ref == "JT chosen"):
                continue
            dd = boots[name] - boots[ref]
            for j, mn in enumerate(METRIC_NAMES):
                if mn in ("auc_roc", "auc_pr", "worst_group_ece", "eod", "dir"):
                    lo, hi = np.nanpercentile(dd[:, j], [2.5, 97.5])
                    diffs.append({"arm": name, "minus": ref, "metric": mn, "diff": float(np.nanmean(dd[:, j])), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)})
    pd.DataFrame(diffs).to_csv(out / "diag_paired_diffs.csv", index=False)
    save_json(out / "diag_meta.json", {"lambdas": LAM, "adv_steps": 5, "seeds": SEEDS, "n_boot": N_BOOT, "eg": extra,
                                       "jt_w": "not run: sampler check shows importance weights = 1.0 (rejected)"})
    pd.set_option("display.width", 250)
    cols = [c for c in ("arm", "auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall") if c in tab.columns]
    print(tab[cols].round(4).to_string(index=False))
    print(pd.DataFrame(diffs).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
