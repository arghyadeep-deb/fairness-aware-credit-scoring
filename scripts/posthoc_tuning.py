"""Post-hoc tuning round: REF-T, ENS, REF-T-EG, ENS-EG (added after the original test run; declared in RUN_LOG).

python scripts/posthoc_tuning.py tune <run_dir>     Optuna on split-42 TRAIN only
python scripts/posthoc_tuning.py val <run_dir>      final fits, EG, validation tables
python scripts/posthoc_tuning.py test <run_dir>     ONE guarded test access per arm passing its gate
python scripts/posthoc_tuning.py report <run_dir>   test tables, from saved files only
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.tensors import load_arrays
from src.eval.bootstrap import bootstrap_metrics, pooled_ci
from src.eval.final import load_backbone, logits_and_z
from src.eval.metrics import METRIC_NAMES, compute_metrics
from src.eval.threshold import apply_threshold, flag_threshold
from src.posthoc.eg import fit_eg, match_eg_weight, pmf
from src.posthoc.prior_correction import correct_logits
from src.run_arms import sigmoid
from src.utils import load_config

SEEDS, EPS, FR, N_BOOT = [0, 1, 2], 0.01, 0.10, 1000
LABEL = "added after the original test run, post-hoc"
ARMS = ["REF-T", "ENS", "REF-T-EG", "ENS-EG"]
METRICS = ["auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]
PREDS = ROOT / "artifacts" / "test_preds"
P2 = ROOT / "artifacts" / "runs" / "20261101T055729Z_phase2"
TRIALS, TIMEOUT, STOP, MAX_TREES = 40, 70 * 60, 200, 5000


def p6() -> Path:
    return ROOT / (ROOT / "artifacts" / "phase6_rundir.txt").read_text().strip()


def posthoc_dir() -> Path:
    return ROOT / (ROOT / "artifacts" / "posthoc_ref_rundir.txt").read_text().strip()


def xm(a) -> np.ndarray:
    return np.concatenate([a.cont, a.cat], axis=1).astype("float32")


def contrib(b, X) -> np.ndarray:
    return b.predict(X, pred_contrib=True)[:, :-1].astype("float32")


def dump(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1), encoding="utf-8")


def ecdf(sorted_val: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_val, x, side="right") / len(sorted_val)


def thr_flags(score_val, score) -> np.ndarray:
    return apply_threshold(score, flag_threshold(score_val, FR))


def stage_tune(out: Path, cfg) -> None:
    tr, _ = load_arrays(cfg, "train")
    X = xm(tr)
    cats = list(range(tr.cont.shape[1], X.shape[1]))
    strata = tr.y * 2 + tr.a
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, strata))

    def objective(t: optuna.Trial) -> float:
        p = {
            "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 10, 300, log=True),
            "colsample_bytree": t.suggest_float("feature_fraction", 0.3, 1.0),
            "subsample": t.suggest_float("bagging_fraction", 0.5, 1.0),
            "reg_alpha": t.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "reg_lambda": t.suggest_float("lambda_l2", 1e-3, 10, log=True),
            "max_bin": t.suggest_int("max_bin", 63, 511),
        }
        aucs, iters = [], []
        for a_, b_ in folds:
            m = lgb.LGBMClassifier(**p, n_estimators=MAX_TREES, subsample_freq=1, metric="auc", objective="binary",
                                   random_state=42, n_jobs=4, verbose=-1)  # fmt: skip
            m.fit(X[a_], tr.y[a_], eval_set=[(X[b_], tr.y[b_])], categorical_feature=cats,
                  callbacks=[lgb.early_stopping(STOP, first_metric_only=True, verbose=False)])  # fmt: skip
            aucs.append(roc_auc_score(tr.y[b_], m.predict_proba(X[b_])[:, 1]))
            iters.append(int(m.best_iteration_ or 0))
        t.set_user_attr("best_iterations", iters)
        return float(np.mean(aucs))

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=TRIALS, timeout=TIMEOUT, show_progress_bar=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
    dump(out / "tuning.json", {"best_params": study.best_params, "best_cv_auc_roc": study.best_value,
                               "n_completed": len(done), "planned": TRIALS, "seconds": time.time() - t0, "label": LABEL})  # fmt: skip
    print("TUNED", study.best_params, study.best_value, "completed", len(done), "s", round(time.time() - t0), flush=True)


def lgb_params(best: dict) -> dict:
    return {"num_leaves": best["num_leaves"], "learning_rate": best["learning_rate"],
            "min_child_samples": best["min_child_samples"], "colsample_bytree": best["feature_fraction"],
            "subsample": best["bagging_fraction"], "reg_alpha": best["lambda_l1"], "reg_lambda": best["lambda_l2"],
            "max_bin": best["max_bin"], "subsample_freq": 1, "n_estimators": MAX_TREES, "metric": "auc"}  # fmt: skip


def a2_model(s, tr, cards, cfg, device):
    hp = json.loads((P2 / "a2" / "tuning.json").read_text())["best_params"]
    state = torch.load(P2 / "a2" / f"seed_{s}" / "best.pt")
    return load_backbone("tabtransformer", tr, cards, cfg, hp, state, device)


def fit_eg_z(ztr, zv, tr, ec, prob_v, d: Path, name: str) -> dict:
    sc = StandardScaler().fit(ztr)
    zs = sc.transform(ztr).astype("float32")
    w, rate = match_eg_weight(zs, tr.y, tr.a, FR, EPS, ec["max_iter"], ec["logistic_max_iter"])
    eg = fit_eg(zs, tr.y, tr.a, w, EPS, ec["max_iter"], ec["logistic_max_iter"])
    pm = pmf(eg, sc.transform(zv).astype("float32"))
    np.save(d / f"sv_{name}.npy", pm + 1e-6 * prob_v)
    joblib.dump({"eg": eg, "scaler": sc}, d / f"eg_{name}.joblib")
    return {"w_pos": float(w), "eg_train_flag_rate": float(rate), "eg_val_natural_rate": float(pm.mean()),
            "n_predictors": len(eg.predictors_)}  # fmt: skip


def table(arms: dict, y, a, eb: int, refs: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, boots = [], {}
    for name, per in arms.items():
        ms = [compute_metrics(y, p, a, f, eb) for p, f in per]
        bs = [bootstrap_metrics(y, p, a, f, N_BOOT, 0, eb, n_jobs=4, chunk_size=100) for p, f in per]
        boots[name] = np.nanmean(np.stack(bs), axis=0)
        lo, hi = pooled_ci(bs)
        row = {"arm": name, "flag_rate": float(np.mean([f.mean() for _, f in per]))}
        for i, k in enumerate(METRIC_NAMES):
            row[k] = float(np.mean([m[k] for m in ms]))
            row[k + "_lo"], row[k + "_hi"] = float(lo[i]), float(hi[i])
        rows.append(row)
    pt = {r["arm"]: r for r in rows}
    diffs = []
    for arm in [n for n in ARMS if n in arms]:
        for ref in refs:
            for m in METRICS:
                j = METRIC_NAMES.index(m)
                l_, h_ = np.nanpercentile(boots[arm][:, j] - boots[ref][:, j], [2.5, 97.5])
                diffs.append({"arm": arm, "minus": ref, "metric": m, "diff": pt[arm][m] - pt[ref][m],
                              "lo": float(l_), "hi": float(h_), "sig": bool(l_ > 0 or h_ < 0)})  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def show(tab, diffs) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(tab[["arm", "flag_rate", *METRICS]].round(4).to_string(index=False))
    print(diffs.round(4).to_string(index=False))


def stage_val(out: Path, cfg) -> None:
    device = torch.device("cuda")
    torch.set_num_threads(4)
    eb, ec = cfg["evaluation"]["ece_bins"], cfg["exponentiated_gradient"]
    best = json.loads((out / "tuning.json").read_text())["best_params"]
    tr, cards = load_arrays(cfg, "train")
    va, _ = load_arrays(cfg, "val")
    weights = json.loads((ROOT / cfg["paths"]["prep_artifacts"] / "class_weights.json").read_text())
    X, Xv = xm(tr), xm(va)
    cats = list(range(tr.cont.shape[1], X.shape[1]))
    t0 = time.time()
    for s in SEEDS:
        d = out / "final" / f"seed_{s}"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "ref_t.txt").exists():
            m = lgb.LGBMClassifier(**lgb_params(best), objective="binary", random_state=s, n_jobs=4, verbose=-1)
            m.fit(X, tr.y, eval_set=[(Xv, va.y)], categorical_feature=cats,
                  callbacks=[lgb.early_stopping(STOP, first_metric_only=True, verbose=False)])  # fmt: skip
            m.booster_.save_model(str(d / "ref_t.txt"), num_iteration=m.best_iteration_)
        b = lgb.Booster(model_file=str(d / "ref_t.txt"))
        np.save(d / "pv_ref_t.npy", b.predict(Xv))
        m2 = a2_model(s, tr, cards, cfg, device)
        lv, zv = logits_and_z(m2, va, device)
        np.save(d / "pv_s1.npy", sigmoid(correct_logits(lv, weights)))
        np.save(d / "zv_a2.npy", zv.astype("float32"))
        del m2
        torch.cuda.empty_cache()
        print(f"seed {s} models ready {time.time() - t0:.0f}s", flush=True)
    grid = [round(x * 0.1, 1) for x in range(11)]
    res = {}
    for w in grid:
        vals = []
        for s in SEEDS:
            d = out / "final" / f"seed_{s}"
            pt_, ps_ = np.load(d / "pv_ref_t.npy"), np.load(d / "pv_s1.npy")
            r = w * ecdf(np.sort(pt_), pt_) + (1 - w) * ecdf(np.sort(ps_), ps_)
            vals.append(roc_auc_score(va.y, r))
        res[w] = float(np.mean(vals))
    w = max(grid, key=lambda g: (round(res[g], 10), g))
    print("weights", res, "chosen", w, flush=True)
    info = {"weight_grid_mean_val_auc_roc": {str(k): v for k, v in res.items()}, "w": w, "seeds": []}
    per = {a: [] for a in ARMS}
    for s in SEEDS:
        d = out / "final" / f"seed_{s}"
        b = lgb.Booster(model_file=str(d / "ref_t.txt"))
        pvt, pvs = np.load(d / "pv_ref_t.npy"), np.load(d / "pv_s1.npy")
        st_, ss_ = np.sort(pvt), np.sort(pvs)
        r = w * ecdf(st_, pvt) + (1 - w) * ecdf(ss_, pvs)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.clip(r, 1e-6, 1)[:, None], va.y)
        pe = lr.predict_proba(np.clip(r, 1e-6, 1)[:, None])[:, 1]
        joblib.dump({"sorted_ref_t": st_, "sorted_s1": ss_, "platt": lr, "w": w}, d / "ens.joblib")
        np.save(d / "pv_ens.npy", pe)
        ztr = contrib(b, X)
        zvt = contrib(b, Xv)
        i1 = fit_eg_z(ztr, zvt, tr, ec, pvt, d, "ref_t")
        print(f"seed {s} REF-T-EG done {time.time() - t0:.0f}s", i1, flush=True)
        m2 = a2_model(s, tr, cards, cfg, device)
        zta = logits_and_z(m2, tr, device)[1].astype("float32")
        del m2
        torch.cuda.empty_cache()
        i2 = fit_eg_z(np.concatenate([ztr, zta], axis=1), np.concatenate([zvt, np.load(d / "zv_a2.npy")], axis=1), tr, ec, pe, d, "ens")
        print(f"seed {s} ENS-EG done {time.time() - t0:.0f}s", i2, flush=True)
        info["seeds"].append({"seed": s, "REF-T-EG": i1, "ENS-EG": i2})
        per["REF-T"].append((pvt, thr_flags(pvt, pvt)))
        per["ENS"].append((pe, thr_flags(pe, pe)))
        sv1 = np.load(d / "sv_ref_t.npy")
        per["REF-T-EG"].append((pvt, thr_flags(sv1, sv1)))
        sv2 = np.load(d / "sv_ens.npy")
        per["ENS-EG"].append((pe, thr_flags(sv2, sv2)))
    gate = {a: all(abs(x[a]["eg_train_flag_rate"] - FR) < 0.01 for x in info["seeds"]) for a in ("REF-T-EG", "ENS-EG")}
    info["gate_passed"] = {"REF-T": True, "ENS": True, **gate}
    info["seconds"] = round(time.time() - t0)
    info["label"] = LABEL
    dump(out / "validation_meta.json", info)
    q, ph = p6(), posthoc_dir()
    per["REF"] = [(np.load(ph / "hc" / f"seed_{s}" / "pv.npy"),) * 2 for s in SEEDS]
    per["REF"] = [(p, thr_flags(p, p)) for p, _ in per["REF"]]
    per["REF-EG"] = []
    for s in SEEDS:
        sv = np.load(ph / "hc" / f"seed_{s}" / "sv.npy")
        per["REF-EG"].append((np.load(ph / "hc" / f"seed_{s}" / "pv.npy"), thr_flags(sv, sv)))
    per["S2-EG"] = [(np.load(q / "s2_eg_v2" / f"seed_{s}" / "val_prob.npy"), np.load(q / "s2_eg_v2" / f"seed_{s}" / "val_flags.npy").astype(bool)) for s in SEEDS]
    tab, diffs = table(per, va.y, va.a, eb, ("REF", "REF-EG", "S2-EG"))
    tab.to_csv(out / "validation_table.csv", index=False)
    diffs.to_csv(out / "validation_diffs.csv", index=False)
    show(tab, diffs)
    print("GATES", info["gate_passed"], "w", w, flush=True)
    print("VAL DONE", flush=True)


def stage_test(out: Path, cfg) -> None:
    device = torch.device("cuda")
    torch.set_num_threads(4)
    meta = json.loads((out / "validation_meta.json").read_text())
    weights = json.loads((ROOT / cfg["paths"]["prep_artifacts"] / "class_weights.json").read_text())
    tr, cards = load_arrays(cfg, "train")
    for arm in ARMS:
        if not meta["gate_passed"][arm]:
            print(arm, "failed its gate: no test access", flush=True)
            continue
        if arm == "ENS" and meta["w"] == 1.0:
            # w = 1: ENS is a monotone map of REF-T, so it comes from REF-T's saved predictions, with no access.
            root, src = PREDS / "ENS", PREDS / "REF-T"
            if root.exists():
                raise SystemExit(f"HARD STOP: {root} exists")
            root.mkdir(parents=True)
            np.save(root / "y.npy", np.load(src / "y.npy"))
            np.save(root / "a.npy", np.load(src / "a.npy"))
            for s in SEEDS:
                d, sd = out / "final" / f"seed_{s}", root / f"seed_{s}"
                sd.mkdir()
                e = joblib.load(d / "ens.joblib")
                r = ecdf(e["sorted_ref_t"], np.load(src / f"seed_{s}" / "pt.npy"))
                pt = e["platt"].predict_proba(np.clip(r, 1e-6, 1)[:, None])[:, 1]
                sv = np.load(d / "pv_ens.npy")
                np.save(sd / "pt.npy", pt)
                np.save(sd / "st.npy", pt)
                np.save(sd / "flags.npy", thr_flags(sv, pt))
                dump(sd / "meta.json", {"threshold_from_val": float(flag_threshold(sv, FR)), "label": LABEL})
            dump(root / "access_done.json", {"arm": arm, "label": LABEL, "derived_from_saved_REF-T_no_test_access": True})
            print(arm, "derived from saved REF-T, no access", flush=True)
            continue
        root = PREDS / arm
        if root.exists():
            raise SystemExit(f"HARD STOP: {root} exists; refusing a second access")
        root.mkdir(parents=True)
        dump(root / "access_started.json", {"label": LABEL, "time": time.time()})
        os.environ["ALLOW_TEST"] = "1"
        try:
            te, _ = load_arrays(cfg, "test", purpose=f"posthoc:{arm}")
        finally:
            os.environ.pop("ALLOW_TEST", None)
        np.save(root / "y.npy", te.y)
        np.save(root / "a.npy", te.a)
        Xt = xm(te)
        for s in SEEDS:
            d, sd = out / "final" / f"seed_{s}", root / f"seed_{s}"
            sd.mkdir()
            b = lgb.Booster(model_file=str(d / "ref_t.txt"))
            ptt = b.predict(Xt)
            if arm == "REF-T":
                pt, st, sv = ptt, ptt, np.load(d / "pv_ref_t.npy")
            elif arm == "REF-T-EG":
                art = joblib.load(d / "eg_ref_t.joblib")
                pt = ptt
                st = pmf(art["eg"], art["scaler"].transform(contrib(b, Xt)).astype("float32")) + 1e-6 * pt
                sv = np.load(d / "sv_ref_t.npy")
            else:
                m2 = a2_model(s, tr, cards, cfg, device)
                lt, zt = logits_and_z(m2, te, device)
                del m2
                torch.cuda.empty_cache()
                ps = sigmoid(correct_logits(lt, weights))
                e = joblib.load(d / "ens.joblib")
                r = e["w"] * ecdf(e["sorted_ref_t"], ptt) + (1 - e["w"]) * ecdf(e["sorted_s1"], ps)
                pt = e["platt"].predict_proba(np.clip(r, 1e-6, 1)[:, None])[:, 1]
                if arm == "ENS":
                    st, sv = pt, np.load(d / "pv_ens.npy")
                else:
                    art = joblib.load(d / "eg_ens.joblib")
                    z = np.concatenate([contrib(b, Xt), zt.astype("float32")], axis=1)
                    st = pmf(art["eg"], art["scaler"].transform(z).astype("float32")) + 1e-6 * pt
                    sv = np.load(d / "sv_ens.npy")
            np.save(sd / "pt.npy", pt)
            np.save(sd / "st.npy", st)
            np.save(sd / "flags.npy", thr_flags(sv, st))
            dump(sd / "meta.json", {"threshold_from_val": float(flag_threshold(sv, FR)), "label": LABEL})
        dump(root / "access_done.json", {"arm": arm, "label": LABEL})
        print(arm, "accessed", flush=True)
    print("TEST DONE", flush=True)


def stage_report(out: Path, cfg) -> None:
    eb = cfg["evaluation"]["ece_bins"]
    y, a = np.load(PREDS / "REF" / "y.npy"), np.load(PREDS / "REF" / "a.npy")
    per = {}
    for arm in [*ARMS, "REF", "REF-EG", "S2-EG"]:
        if not (PREDS / arm / "access_done.json").exists():
            continue
        assert (np.load(PREDS / arm / "y.npy") == y).all() and (np.load(PREDS / arm / "a.npy") == a).all()
        per[arm] = [(np.load(PREDS / arm / f"seed_{s}" / "pt.npy"), np.load(PREDS / arm / f"seed_{s}" / "flags.npy").astype(bool)) for s in SEEDS]
    tab, diffs = table(per, y, a, eb, ("REF", "REF-EG", "S2-EG"))
    tab.to_csv(out / "test_table.csv", index=False)
    diffs.to_csv(out / "test_diffs.csv", index=False)
    show(tab, diffs)


def main() -> None:
    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/train.yaml")
    {"tune": stage_tune, "val": stage_val, "test": stage_test, "report": stage_report}[stage](out, cfg)


if __name__ == "__main__":
    main()
