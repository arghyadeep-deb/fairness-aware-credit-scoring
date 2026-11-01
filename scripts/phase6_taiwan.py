"""UCI Taiwan retrain (Phase 6; protocol declared in RUN_LOG commit f6d3e3d). No re-tuning:
Home Credit hyper-parameters and lambdas, SEX as gender (never in X), 3 seeds.

python scripts/phase6_taiwan.py <p2> <p4b> <p6>
Arms: A0, A2, S1, S2-EG, S2-GT, JT chosen (0.05,2,0), JT tuned-hp (0.05,2,0), H1-H3.
Reported on validation and on the Taiwan TEST split (read once, logged in
artifacts/guard/taiwan_access.jsonl) at the shared 10% flag rate and at the train default rate.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.taiwan import prepare  # noqa: E402
from src.eval.bootstrap import bootstrap_metrics, pooled_ci  # noqa: E402
from src.eval.final import load_backbone, load_joint, logits_and_z  # noqa: E402
from src.eval.metrics import METRIC_NAMES, compute_metrics  # noqa: E402
from src.eval.threshold import apply_threshold, flag_threshold  # noqa: E402
from src.posthoc.eg import fit_eg, match_eg_weight, pmf  # noqa: E402
from src.posthoc.prior_correction import correct_logits  # noqa: E402
from src.posthoc.threshold_optimizer import (
    apply_group_thresholds,
    fit_group_thresholds,
)  # noqa: E402
from src.run_arms import arm_cfg, save_json, sigmoid  # noqa: E402
from src.train import fit  # noqa: E402
from src.train_joint import fit_joint  # noqa: E402
from src.utils import git_hash, load_config, resolve  # noqa: E402

S = [0, 1, 2]
DEPLOYABLE = {"A0": True, "A2": True, "S1": True, "S2-EG": True, "S2-GT": False,
              "JT chosen (0.05,2,0)": True, "JT tuned-hp (0.05,2,0)": True,
              "H1 (0.05)+S1": True, "H2 (0.10)+S1": True, "H3 (0.25)+S1": True}  # fmt: skip


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p2, p4b, p6 = (Path(x) for x in sys.argv[1:4])
    out = p6 / "taiwan"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/train.yaml")
    device = torch.device("cuda")
    arr, cards, weights = prepare(ROOT)
    tr, va, te = arr["train"], arr["val"], arr["test"]
    rates = {"10% flag rate": 0.10, "train default rate": float(tr.y.mean())}
    hp_a2 = json.loads((p2 / "a2" / "tuning.json").read_text())["best_params"]
    hp_a0 = json.loads((p2 / "a0" / "tuning.json").read_text())["best_params"]
    hp_tuned = json.loads((p4b / "jt_tuning.json").read_text())["best_params"]
    ec = cfg["exponentiated_gradient"]
    pred: dict[tuple, dict] = {}  # (arm, seed) -> val/test probabilities and scores per rate
    t0 = time.time()
    for s in S:
        # ---- train (validation drives early stopping, exactly as on Home Credit) ----
        r_a0 = fit(arm_cfg("a0"), cfg, hp_a0, tr, va, cards, weights, s, device)
        r_a2 = fit(arm_cfg("a2"), cfg, hp_a2, tr, va, cards, weights, s, device)
        joint = {
            "JT chosen (0.05,2,0)": ((0.05, 2.0, 0.0), hp_a2),
            "JT tuned-hp (0.05,2,0)": ((0.05, 2.0, 0.0), hp_tuned),
            "H1 (0.05)+S1": ((0.05, 0.0, 0.0), hp_tuned),
            "H2 (0.10)+S1": ((0.10, 0.0, 0.0), hp_tuned),
            "H3 (0.25)+S1": ((0.25, 0.0, 0.0), hp_tuned),
        }
        models = {
            "A0": load_backbone("tabtransformer", tr, cards, cfg, hp_a0, r_a0["state"], device),
            "A2": load_backbone("tabtransformer", tr, cards, cfg, hp_a2, r_a2["state"], device),
        }
        for nm, (lam, hp) in joint.items():
            r = fit_joint(cfg, hp, tr, va, cards, weights, s, device, lam)
            models[nm] = load_joint(tr, cards, cfg, hp, r["state"], device)
            print(f"seed {s} {nm}: val AUC-PR {r['best_val_auc_pr']:.4f} epochs {len(r['history'])}", flush=True)  # fmt: skip
        print(f"seed {s} A0 {r_a0['best_val_auc_pr']:.4f} A2 {r_a2['best_val_auc_pr']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
        lg = {
            nm: (logits_and_z(m, va, device), logits_and_z(m, te, device))
            for nm, m in models.items()
        }
        # ---- scores ----
        for nm in models:
            lv, lt = lg[nm][0][0], lg[nm][1][0]
            if nm.startswith("H"):
                lv, lt = correct_logits(lv, weights), correct_logits(lt, weights)
            pred[(nm, s)] = {
                "pv": sigmoid(lv),
                "pt": sigmoid(lt),
                "sv": sigmoid(lv),
                "st": sigmoid(lt),
            }
        lv2, lt2 = lg["A2"][0][0], lg["A2"][1][0]
        p1v, p1t = sigmoid(correct_logits(lv2, weights)), sigmoid(correct_logits(lt2, weights))
        pred[("S1", s)] = {"pv": p1v, "pt": p1t, "sv": p1v, "st": p1t}
        pred[("S2-GT", s)] = {"pv": p1v, "pt": p1t, "gt": True}
        zt = logits_and_z(models["A2"], tr, device)[1]
        sc = StandardScaler().fit(zt)
        zs = {
            k: sc.transform(v)
            for k, v in (("tr", zt), ("va", lg["A2"][0][1]), ("te", lg["A2"][1][1]))
        }
        eg_entry = {"pv": p1v, "pt": p1t, "eg": {}}
        for rn, rate in rates.items():  # EG class weight matched to each flag rate (train only)
            w, got = match_eg_weight(
                zs["tr"], tr.y, tr.a, rate, 0.01, ec["max_iter"], ec["logistic_max_iter"]
            )
            eg = fit_eg(zs["tr"], tr.y, tr.a, w, 0.01, ec["max_iter"], ec["logistic_max_iter"])
            eg_entry["eg"][rn] = (pmf(eg, zs["va"]) + 1e-6 * p1v, pmf(eg, zs["te"]) + 1e-6 * p1t)
        pred[("S2-EG", s)] = eg_entry
    # ---- the Taiwan test split is read once above (logged), evaluation below ----
    log = resolve("artifacts/guard/taiwan_access.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "purpose": "phase6:taiwan_all_arms", "git": git_hash()}) + "\n")  # fmt: skip
    rows = []
    for arm in DEPLOYABLE:
        for rn, rate in rates.items():
            met = {"val": [], "test": []}
            boots = []
            for s in S:
                e = pred[(arm, s)]
                if e.get("gt"):
                    thr = fit_group_thresholds(va.y, e["pv"], va.a, rate)
                    fl = {"val": apply_group_thresholds(e["pv"], va.a, thr), "test": apply_group_thresholds(e["pt"], te.a, thr)}  # fmt: skip
                else:
                    sv, st = e["eg"][rn] if "eg" in e else (e["sv"], e["st"])
                    th = flag_threshold(sv, rate)
                    fl = {"val": apply_threshold(sv, th), "test": apply_threshold(st, th)}
                met["val"].append(compute_metrics(va.y, e["pv"], va.a, fl["val"], 15))
                met["test"].append(compute_metrics(te.y, e["pt"], te.a, fl["test"], 15))
                boots.append(bootstrap_metrics(te.y, e["pt"], te.a, fl["test"], 1000, 0, 15))
            lo, hi = pooled_ci(boots)
            for split in ("val", "test"):
                row = {
                    "arm": arm,
                    "deployable": DEPLOYABLE[arm],
                    "rate": rn,
                    "flag_rate": rate,
                    "split": split,
                }
                for i, k in enumerate(METRIC_NAMES):
                    row[k] = float(np.nanmean([m[k] for m in met[split]]))
                    if split == "test":
                        row[k + "_lo"], row[k + "_hi"] = float(lo[i]), float(hi[i])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "taiwan_results.csv", index=False)
    save_json(out / "taiwan_meta.json", {"git": git_hash(), "class_weights": weights, "rates": rates,
                                          "hp_a2": hp_a2, "hp_a0": hp_a0, "hp_tuned": hp_tuned,
                                          "n": {k: int(len(v.y)) for k, v in arr.items()},
                                          "seconds": time.time() - t0})  # fmt: skip
    pd.set_option("display.width", 250)
    cols = [
        "arm",
        "deployable",
        "rate",
        "split",
        "auc_roc",
        "auc_pr",
        "worst_group_ece",
        "eod",
        "dir",
        "recall",
    ]
    print(df[cols].round(4).to_string())


if __name__ == "__main__":
    main()
