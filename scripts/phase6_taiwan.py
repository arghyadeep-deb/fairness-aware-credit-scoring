"""UCI Taiwan retrain (Phase 6; specs/08). NO re-tuning: the Home Credit hyper-parameters and the
rule-selected JT lambdas are reused unchanged, SEX is the gender array (never in X), 3 seeds.

python scripts/phase6_taiwan.py fit     <p2> <p6>   # train A0, A2, JT chosen, REF per seed (train/val only)
python scripts/phase6_taiwan.py access  <p2> <p6>   # ONE guarded Taiwan-test access per arm; predictions -> disk
python scripts/phase6_taiwan.py analyze <p2> <p6>   # metrics, bootstrap CIs, z-probe: saved files only

Arms: A0, A2, S1, S2-EG, S2-GT (analysis only), JT chosen, REF, plus the gender-from-features probe (LEAK).
Reported on validation and TEST at the shared 10% flag rate and at the train default rate.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):  # fmt: skip
    os.environ[_v] = "4"

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.taiwan import access_test, odd_code_counts, prepare  # noqa: E402
from src.eval.bootstrap import bootstrap_metrics, pooled_ci  # noqa: E402
from src.eval.final import load_backbone, load_joint, logits_and_z, z_probe_test  # noqa: E402
from src.eval.leakage import fit_lgbm_probe  # noqa: E402
from src.eval.metrics import METRIC_NAMES, compute_metrics  # noqa: E402
from src.eval.threshold import apply_threshold, flag_threshold  # noqa: E402
from src.posthoc.eg import fit_eg, match_eg_weight, pmf  # noqa: E402
from src.posthoc.prior_correction import correct_logits  # noqa: E402
from src.posthoc.threshold_optimizer import (
    apply_group_thresholds,
    fit_group_thresholds,
)  # noqa: E402
from src.run_arms import arm_cfg, save_json, sigmoid  # noqa: E402
from src.run_joint import tag  # noqa: E402
from src.train import fit  # noqa: E402
from src.train_joint import fit_joint  # noqa: E402
from src.utils import git_hash, load_config  # noqa: E402

SEEDS = [0, 1, 2]
JT_LAMBDAS = (0.05, 2.0, 0.0)  # the rule-selected JT chosen point (Home Credit); NOT re-tuned
HP_SOURCES = {"A0": "a0/tuning.json", "A2": "a2/tuning.json", "JT chosen": "a2/tuning.json"}
EG_EPS = 0.01
ARMS = {
    "A0": True,
    "A2": True,
    "S1": True,
    "S2-EG": True,
    "S2-GT": False,
    "JT chosen": True,
    "REF": True,
}
GUARD = ROOT / "artifacts/guard/taiwan_access.jsonl"
PREDS = ROOT / "artifacts/test_preds_taiwan"
log = logging.getLogger("taiwan")


def slug(n: str) -> str:
    return n.replace(" ", "_")


def hps(p2: Path) -> dict:
    return {k: json.loads((p2 / v).read_text())["best_params"] for k, v in HP_SOURCES.items()}


def models_dir(p6: Path, s: int) -> Path:
    d = p6 / "taiwan" / "models" / f"seed_{s}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def xmat(a):
    return np.concatenate([a.cont, a.cat], axis=1).astype("float32")


# ----------------------------------------------------------------------------------------- fit
def stage_fit(p2, p6, cfg, mem):
    device = torch.device("cuda")
    arr, cards, weights = prepare(ROOT)
    tr, va = arr["train"], arr["val"]
    hp = hps(p2)
    save_json(p6 / "taiwan" / "prep.json", {"odd_codes_mapped_to_0": odd_code_counts(ROOT), "cards": cards,
                                             "class_weights": weights, "n_train": len(tr.y), "n_val": len(va.y),
                                             "default_rate_train": float(tr.y.mean()),
                                             "female_share_train": float(1 - tr.a.mean()),
                                             "gender_in_X": any("SEX" in n for n in tr.cont_names + tr.cat_names),
                                             "hp": hp, "jt_lambdas": JT_LAMBDAS})  # fmt: skip
    for s in SEEDS:
        d = models_dir(p6, s)
        if (d / "done.json").exists():
            continue
        r0 = fit(arm_cfg("a0"), cfg, hp["A0"], tr, va, cards, weights, s, device)
        r2 = fit(arm_cfg("a2"), cfg, hp["A2"], tr, va, cards, weights, s, device)
        rj = fit_joint(cfg, hp["JT chosen"], tr, va, cards, weights, s, device, JT_LAMBDAS)
        torch.save(r0["state"], d / "a0.pt")
        torch.save(r2["state"], d / "a2.pt")
        torch.save(rj["state"], d / "jt.pt")
        pr = dict(cfg["reference"]["lightgbm"])
        stop = pr.pop("early_stopping_rounds")
        X, Xv = xmat(tr), xmat(va)
        cats = list(range(tr.cont.shape[1], X.shape[1]))
        ref = lgb.LGBMClassifier(**pr, objective="binary", random_state=s, n_jobs=4, verbose=-1)
        ref.fit(X, tr.y, eval_set=[(Xv, va.y)], categorical_feature=cats,
                callbacks=[lgb.early_stopping(stop, first_metric_only=True, verbose=False)])  # fmt: skip
        ref.booster_.save_model(str(d / "ref.txt"))
        save_json(d / "done.json", {"val_auc_pr": {"A0": r0["best_val_auc_pr"], "A2": r2["best_val_auc_pr"], "JT": rj["best_val_auc_pr"]}})  # fmt: skip
        print(f"seed {s}: A0 {r0['best_val_auc_pr']:.4f} A2 {r2['best_val_auc_pr']:.4f} JT {rj['best_val_auc_pr']:.4f}", flush=True)  # fmt: skip
        del r0, r2, rj, ref
        gc.collect()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------- access
def _load_nn(arm, tr, cards, cfg, hp, d, device):
    if arm == "A0":
        return load_backbone(
            "tabtransformer", tr, cards, cfg, hp["A0"], torch.load(d / "a0.pt"), device
        )
    if arm == "JT chosen":
        return load_joint(tr, cards, cfg, hp["JT chosen"], torch.load(d / "jt.pt"), device)
    return load_backbone(
        "tabtransformer", tr, cards, cfg, hp["A2"], torch.load(d / "a2.pt"), device
    )


def stage_access(p2, p6, cfg, mem, only):
    device = torch.device("cuda")
    arr, cards, weights = prepare(ROOT)
    tr, va = arr["train"], arr["val"]
    hp = hps(p2)
    ec = cfg["exponentiated_gradient"]
    rates = {"10% flag rate": 0.10, "train default rate": float(tr.y.mean())}
    for arm in only or ["LEAK"] + list(ARMS):
        root = PREDS / slug(arm)
        if (root / "access_done.json").exists():
            continue
        if (root / "access_started.json").exists():
            log.warning("SECOND TAIWAN TEST ACCESS for %s: log as exception", arm)
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / "access_started.json", {"git": git_hash(), "time": time.time()})
        te = access_test(ROOT, f"phase6:taiwan:{arm}", GUARD, git_hash())  # THE access for this arm
        np.save(root / "y.npy", te.y)
        np.save(root / "a.npy", te.a)
        if arm == "LEAK":
            cols = [f"x{i}" for i in range(tr.cont.shape[1] + tr.cat.shape[1])]
            df = lambda a_: pd.DataFrame(xmat(a_), columns=cols)  # noqa: E731
            res = {}
            for s in SEEDS:
                g = fit_lgbm_probe(df(tr), tr.a, df(va), va.a, [], cfg["leakage"]["lightgbm"], s)
                res[f"seed_{s}"] = {"lgbm_val_auc": float(roc_auc_score(va.a, g.predict_proba(df(va))[:, 1])),
                                    "lgbm_test_auc": float(roc_auc_score(te.a, g.predict_proba(df(te))[:, 1]))}  # fmt: skip
            save_json(root / "leak.json", res)
        for s in SEEDS if arm != "LEAK" else []:
            d, sd = models_dir(p6, s), root / f"seed_{s}"
            sd.mkdir(parents=True, exist_ok=True)
            extra = {}
            if arm == "REF":
                ref = lgb.Booster(model_file=str(d / "ref.txt"))
                pv, pt = ref.predict(xmat(va)), ref.predict(xmat(te))
                sv, st = pv, pt
            else:
                m = _load_nn(
                    "A2" if arm in ("S1", "S2-EG", "S2-GT") else arm, tr, cards, cfg, hp, d, device
                )
                lv, zv = logits_and_z(m, va, device)
                lt, zt = logits_and_z(m, te, device)
                if arm in ("S1", "S2-EG", "S2-GT"):
                    lv, lt = correct_logits(lv, weights), correct_logits(lt, weights)
                pv, pt = sigmoid(lv), sigmoid(lt)
                sv, st = pv, pt
                if arm in ("A2", "JT chosen"):
                    np.save(sd / "ztr.npy", logits_and_z(m, tr, device)[1].astype("float32"))
                    np.save(sd / "zv.npy", zv.astype("float32"))
                    np.save(sd / "zt.npy", zt.astype("float32"))
                if arm == "S2-EG":
                    ztr = logits_and_z(m, tr, device)[1]
                    sc = StandardScaler().fit(ztr)
                    ws = {}
                    for (
                        rn,
                        rate,
                    ) in rates.items():  # EG class weight matched to each flag rate (train only)
                        w, got = match_eg_weight(sc.transform(ztr), tr.y, tr.a, rate, EG_EPS, ec["max_iter"], ec["logistic_max_iter"])  # fmt: skip
                        eg = fit_eg(
                            sc.transform(ztr),
                            tr.y,
                            tr.a,
                            w,
                            EG_EPS,
                            ec["max_iter"],
                            ec["logistic_max_iter"],
                        )
                        key = "10" if rate == 0.10 else "def"
                        np.save(sd / f"sv_eg_{key}.npy", pmf(eg, sc.transform(zv)) + 1e-6 * pv)
                        np.save(sd / f"st_eg_{key}.npy", pmf(eg, sc.transform(zt)) + 1e-6 * pt)
                        ws[rn] = {"w_pos": float(w), "train_flag_rate": float(got)}
                    extra["eg"] = ws
                del m
            np.save(sd / "pv.npy", pv)
            np.save(sd / "pt.npy", pt)
            save_json(sd / "meta.json", extra)
            gc.collect()
            torch.cuda.empty_cache()
        save_json(root / "access_done.json", {"arm": arm, "git": git_hash()})
        del te
        gc.collect()


# --------------------------------------------------------------------------------------- analyze
def stage_analyze(p2, p6, cfg, mem):
    arr, cards, weights = prepare(ROOT)
    tr, va = arr["train"], arr["val"]
    rates = {"10% flag rate": 0.10, "train default rate": float(tr.y.mean())}
    rows, probes = [], []
    for arm, dep in ARMS.items():
        root = PREDS / slug(arm)
        y, a = np.load(root / "y.npy"), np.load(root / "a.npy")
        for rn, rate in rates.items():
            met, boots = {"val": [], "test": []}, []
            for s in SEEDS:
                sd = root / f"seed_{s}"
                pv, pt = np.load(sd / "pv.npy"), np.load(sd / "pt.npy")
                if arm == "S2-EG":
                    k = "10" if rate == 0.10 else "def"
                    sv, st = np.load(sd / f"sv_eg_{k}.npy"), np.load(sd / f"st_eg_{k}.npy")
                else:
                    sv, st = pv, pt
                if arm == "S2-GT":
                    thr = fit_group_thresholds(va.y, pv, va.a, rate)
                    fl = {
                        "val": apply_group_thresholds(pv, va.a, thr),
                        "test": apply_group_thresholds(pt, a, thr),
                    }
                else:
                    th = flag_threshold(sv, rate)
                    fl = {"val": apply_threshold(sv, th), "test": apply_threshold(st, th)}
                met["val"].append(compute_metrics(va.y, pv, va.a, fl["val"], 15))
                met["test"].append(compute_metrics(y, pt, a, fl["test"], 15))
                boots.append(bootstrap_metrics(y, pt, a, fl["test"], 1000, 0, 15, n_jobs=mem["n_jobs"], chunk_size=mem["bootstrap_chunk_size"]))  # fmt: skip
            lo, hi = pooled_ci(boots)
            for split in ("val", "test"):
                row = {"arm": arm, "deployable": dep, "rate": rn, "flag_rate": rate, "split": split}
                for i, k in enumerate(METRIC_NAMES):
                    row[k] = float(np.nanmean([m[k] for m in met[split]]))
                    if split == "test":
                        row[k + "_lo"], row[k + "_hi"] = float(lo[i]), float(hi[i])
                rows.append(row)
            np.save(p6 / "taiwan" / f"boot_{slug(arm)}_{'10' if rate == 0.10 else 'def'}.npy", np.stack(boots).astype("float32"))  # fmt: skip
            del boots
            gc.collect()
    for arm in ("A2", "JT chosen"):
        root = PREDS / slug(arm)
        a = np.load(root / "a.npy")
        for s in SEEDS:
            sd = root / f"seed_{s}"
            r = z_probe_test(
                cfg,
                np.load(sd / "ztr.npy"),
                tr.a,
                np.load(sd / "zv.npy"),
                va.a,
                np.load(sd / "zt.npy"),
                a,
                s,
            )
            probes.append({"arm": arm, "seed": s, **r})
    out = p6 / "taiwan"
    pd.DataFrame(rows).to_csv(out / "taiwan_results.csv", index=False)
    pd.DataFrame(probes).to_csv(out / "taiwan_zprobe_after.csv", index=False)
    save_json(
        out / "taiwan_leak_before.json", json.loads((PREDS / "LEAK" / "leak.json").read_text())
    )
    save_json(out / "taiwan_meta.json", {"git": git_hash(), "rates": rates, "hp": hps(p2), "jt_lambdas": JT_LAMBDAS,
                                          "jt_dir_tag": tag(JT_LAMBDAS), "n_test": int(len(y))})  # fmt: skip
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
    print(pd.DataFrame(rows)[cols].round(4).to_string())


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    stage = sys.argv[1]
    p2, p6 = Path(sys.argv[2]), Path(sys.argv[3])
    only = sys.argv[4:]
    cfg = load_config("configs/train.yaml")
    mem = load_config("configs/phase6_memory.yaml")
    torch.set_num_threads(mem["cpu_threads"])
    (p6 / "taiwan").mkdir(parents=True, exist_ok=True)
    {"fit": lambda: stage_fit(p2, p6, cfg, mem), "access": lambda: stage_access(p2, p6, cfg, mem, only),
     "analyze": lambda: stage_analyze(p2, p6, cfg, mem)}[stage]()  # fmt: skip


if __name__ == "__main__":
    main()
