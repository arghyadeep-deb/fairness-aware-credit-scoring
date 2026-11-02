"""Post-hoc arms REF-S1 and REF-EG (added after the original test run; declared in RUN_LOG, commit 6e3fdfc).

python scripts/posthoc_ref_eg.py val <run_dir>          validation only: train, fit EG, table
python scripts/posthoc_ref_eg.py test <run_dir>         ONE guarded Home Credit access for REF-EG; saves predictions
python scripts/posthoc_ref_eg.py report <run_dir>       test tables, from saved files only
python scripts/posthoc_ref_eg.py taiwan_val <run_dir>
python scripts/posthoc_ref_eg.py taiwan_test <run_dir>  ONE guarded Taiwan access for REF-EG; then its report

REF has no class weights, so the prior-correction constant is log(1/1) = 0 and REF-S1 equals REF.
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
import pandas as pd
import psutil
from sklearn.preprocessing import StandardScaler

from src.data.taiwan import access_test, prepare
from src.data.tensors import load_arrays
from src.eval.bootstrap import bootstrap_metrics, pooled_ci
from src.eval.metrics import METRIC_NAMES, compute_metrics
from src.eval.threshold import apply_threshold, flag_threshold
from src.posthoc.eg import fit_eg, match_eg_weight, pmf
from src.utils import git_hash, load_config

SEEDS, EPS, N_BOOT = [0, 1, 2], 0.01, 1000
LABEL = "added after the original test run, post-hoc"
METRICS = ["auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]
PREDS = ROOT / "artifacts" / "test_preds"
PREDS_TW = ROOT / "artifacts" / "test_preds_taiwan"
GUARD_TW = ROOT / "artifacts" / "guard" / "taiwan_access.jsonl"
P2 = ROOT / "artifacts" / "runs" / "20261101T055729Z_phase2"


def p6() -> Path:
    return ROOT / (ROOT / "artifacts" / "phase6_rundir.txt").read_text().strip()


def tw_models(s: int) -> Path:
    return p6() / "taiwan" / "models" / f"seed_{s}"


def xm(a) -> np.ndarray:
    return np.concatenate([a.cont, a.cat], axis=1).astype("float32")


def contrib(booster, X) -> np.ndarray:
    """TreeSHAP contribution per feature (bias column dropped): REF's stand-in for z."""
    return booster.predict(X, pred_contrib=True)[:, :-1].astype("float32")


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def peak_gb() -> float:
    m = psutil.Process().memory_info()
    return round(getattr(m, "peak_wset", m.rss) / 1e9, 2)


def thr_flags(score_val, score) -> np.ndarray:
    return apply_threshold(score, flag_threshold(score_val, 0.10))


def fit_eg_on(booster, tr, va, cfg, d: Path, seed: int) -> dict:
    ec = cfg["exponentiated_gradient"]
    Xt, Xv = xm(tr), xm(va)
    pv = booster.predict(Xv)
    ztr = contrib(booster, Xt)
    sc = StandardScaler().fit(ztr)
    ztr = sc.transform(ztr).astype("float32")
    zv = sc.transform(contrib(booster, Xv)).astype("float32")
    w, rate = match_eg_weight(ztr, tr.y, tr.a, 0.10, EPS, ec["max_iter"], ec["logistic_max_iter"])
    eg = fit_eg(ztr, tr.y, tr.a, w, EPS, ec["max_iter"], ec["logistic_max_iter"])
    pm = pmf(eg, zv)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "pv.npy", pv)
    np.save(d / "sv.npy", pm + 1e-6 * pv)
    joblib.dump({"eg": eg, "scaler": sc}, d / "eg.joblib")
    return {
        "seed": seed,
        "w_pos": float(w),
        "eg_train_flag_rate": float(rate),
        "eg_val_natural_rate": float(pm.mean()),
        "n_predictors": len(eg.predictors_),
        "best_gap": float(eg.best_gap_),
        "frac_fractional_val": float(np.mean((pm > 1e-9) & (pm < 1 - 1e-9))),
        "z_dim": int(zv.shape[1]),
        "z_finite": bool(np.isfinite(zv).all()),
    }


def table(arms: dict, y, a, eb: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """arms: name -> list of (prob, flags) per seed. Replicates are paired: same resample indices."""
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
    for arm in ("REF", "REF-EG"):
        for ref in ("S2-EG", "JT chosen", "REF"):
            if ref == arm:
                continue
            for m in METRICS:
                j = METRIC_NAMES.index(m)
                lo_, hi_ = np.nanpercentile(boots[arm][:, j] - boots[ref][:, j], [2.5, 97.5])
                diffs.append({"arm": arm, "minus": ref, "metric": m, "diff": pt[arm][m] - pt[ref][m],
                              "lo": float(lo_), "hi": float(hi_), "sig": bool(lo_ > 0 or hi_ < 0)})  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def show(tab, diffs) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(tab[["arm", "flag_rate", *METRICS]].round(4).to_string(index=False))
    print(diffs.round(4).to_string(index=False))


def stage_val(out: Path, cfg) -> None:
    eb = cfg["evaluation"]["ece_bins"]
    tr, _ = load_arrays(cfg, "train")
    va, _ = load_arrays(cfg, "val")
    X, Xv = xm(tr), xm(va)
    cats = list(range(tr.cont.shape[1], X.shape[1]))
    info, ref, reg = [], [], []
    for s in SEEDS:
        d = out / "hc" / f"seed_{s}"
        t0 = time.time()
        if not (d / "ref.txt").exists():
            d.mkdir(parents=True, exist_ok=True)
            pr = dict(cfg["reference"]["lightgbm"])
            stop = pr.pop("early_stopping_rounds")
            m = lgb.LGBMClassifier(**pr, objective="binary", random_state=s, n_jobs=4, verbose=-1)
            m.fit(X, tr.y, eval_set=[(Xv, va.y)], categorical_feature=cats,
                  callbacks=[lgb.early_stopping(stop, first_metric_only=True, verbose=False)])  # fmt: skip
            m.booster_.save_model(str(d / "ref.txt"), num_iteration=m.best_iteration_)
        b = lgb.Booster(model_file=str(d / "ref.txt"))
        r = fit_eg_on(b, tr, va, cfg, d, s)
        pv, sv = np.load(d / "pv.npy"), np.load(d / "sv.npy")
        saved = np.load(P2 / "ref" / f"seed_{s}" / "val_prob.npy")
        r["retrain_max_abs_val_diff_vs_phase2"] = float(np.abs(pv - saved).max())
        r["seconds"] = round(time.time() - t0, 1)
        info.append(r)
        print(r, flush=True)
        ref.append((pv, thr_flags(pv, pv)))
        reg.append((pv, thr_flags(sv, sv)))
    q = p6()
    s2 = [
        (np.load(q / "s2_eg_v2" / f"seed_{s}" / "val_prob.npy"),
         np.load(q / "s2_eg_v2" / f"seed_{s}" / "val_flags.npy").astype(bool))
        for s in SEEDS
    ]  # fmt: skip
    jt = [np.load(q / "jta_ctl" / "f0p05_c2_a0" / f"seed_{s}" / "val_prob.npy") for s in SEEDS]
    arms = {"REF": ref, "REF-S1": ref, "REF-EG": reg, "S2-EG": s2,
            "JT chosen": [(p, thr_flags(p, p)) for p in jt]}  # fmt: skip
    tab, diffs = table(arms, va.y, va.a, eb)
    tab.to_csv(out / "validation_table.csv", index=False)
    diffs.to_csv(out / "validation_diffs.csv", index=False)
    clean = all(i["z_finite"] and abs(i["eg_train_flag_rate"] - 0.10) < 0.01 for i in info)
    dump(out / "validation_meta.json",
         {"label": LABEL, "seeds": info, "clean": clean, "git": git_hash(), "peak_gb": peak_gb()})  # fmt: skip
    show(tab, diffs)
    print("CLEAN", clean, "peak GB", peak_gb())


def stage_test(out: Path, cfg) -> None:
    if not json.loads((out / "validation_meta.json").read_text())["clean"]:
        raise SystemExit("validation not clean: no test access")
    root = PREDS / "REF-EG"
    if root.exists():
        raise SystemExit("artifacts/test_preds/REF-EG already exists: refusing a second access")
    root.mkdir(parents=True)
    dump(root / "access_started.json", {"git": git_hash(), "time": time.time(), "label": LABEL})
    os.environ["ALLOW_TEST"] = "1"
    try:
        te, _ = load_arrays(cfg, "test", purpose="posthoc:REF-EG")
    finally:
        os.environ.pop("ALLOW_TEST", None)
    np.save(root / "y.npy", te.y)
    np.save(root / "a.npy", te.a)
    Xt = xm(te)
    for s in SEEDS:
        d, sd = out / "hc" / f"seed_{s}", root / f"seed_{s}"
        sd.mkdir()
        b = lgb.Booster(model_file=str(d / "ref.txt"))
        art = joblib.load(d / "eg.joblib")
        pt = b.predict(Xt)
        zt = art["scaler"].transform(contrib(b, Xt)).astype("float32")
        st = pmf(art["eg"], zt) + 1e-6 * pt
        sv = np.load(d / "sv.npy")
        np.save(sd / "pt.npy", pt)
        np.save(sd / "st.npy", st)
        np.save(sd / "flags.npy", thr_flags(sv, st))
        saved = np.load(PREDS / "REF" / f"seed_{s}" / "pt.npy")
        dump(sd / "meta.json", {"threshold_from_val": float(flag_threshold(sv, 0.10)), "label": LABEL,
                                "max_abs_test_diff_vs_saved_REF": float(np.abs(pt - saved).max())})  # fmt: skip
    dump(root / "access_done.json", {"arm": "REF-EG", "git": git_hash(), "peak_gb": peak_gb()})
    print("access done; peak GB", peak_gb())


def stage_report(out: Path, cfg) -> None:
    """Saved files only: no test loading."""
    eb = cfg["evaluation"]["ece_bins"]
    y, a = np.load(PREDS / "REF-EG" / "y.npy"), np.load(PREDS / "REF-EG" / "a.npy")

    def saved(folder: str):
        assert (np.load(PREDS / folder / "y.npy") == y).all()
        assert (np.load(PREDS / folder / "a.npy") == a).all()
        return [
            (np.load(PREDS / folder / f"seed_{s}" / "pt.npy"),
             np.load(PREDS / folder / f"seed_{s}" / "flags.npy").astype(bool))
            for s in SEEDS
        ]  # fmt: skip

    arms = {"REF": saved("REF"), "REF-S1": saved("REF"), "REF-EG": saved("REF-EG"),
            "S2-EG": saved("S2-EG"), "JT chosen": saved("JT_chosen")}  # fmt: skip
    tab, diffs = table(arms, y, a, eb)
    tab.to_csv(out / "test_table.csv", index=False)
    diffs.to_csv(out / "test_diffs.csv", index=False)
    show(tab, diffs)


def stage_taiwan_val(out: Path, cfg) -> None:
    t0 = time.time()
    arr, _, _ = prepare(ROOT)
    tr, va = arr["train"], arr["val"]
    info, ref, reg, s2, jt = [], [], [], [], []
    for s in SEEDS:
        d = out / "taiwan" / f"seed_{s}"
        b = lgb.Booster(model_file=str(tw_models(s) / "ref.txt"))
        info.append(fit_eg_on(b, tr, va, cfg, d, s))
        pv, sv = np.load(d / "pv.npy"), np.load(d / "sv.npy")
        ref.append((pv, thr_flags(pv, pv)))
        reg.append((pv, thr_flags(sv, sv)))
        e = PREDS_TW / "S2-EG" / f"seed_{s}"
        sv2 = np.load(e / "sv_eg_10.npy")
        s2.append((np.load(e / "pv.npy"), thr_flags(sv2, sv2)))
        pj = np.load(PREDS_TW / "JT_chosen" / f"seed_{s}" / "pv.npy")
        jt.append((pj, thr_flags(pj, pj)))
    arms = {"REF": ref, "REF-S1": ref, "REF-EG": reg, "S2-EG": s2, "JT chosen": jt}
    tab, diffs = table(arms, va.y, va.a, 15)
    tab.to_csv(out / "taiwan_validation_table.csv", index=False)
    diffs.to_csv(out / "taiwan_validation_diffs.csv", index=False)
    clean = all(i["z_finite"] and abs(i["eg_train_flag_rate"] - 0.10) < 0.01 for i in info)
    dump(out / "taiwan_validation_meta.json",
         {"label": LABEL, "seeds": info, "clean": clean, "seconds": time.time() - t0})  # fmt: skip
    print(info)
    show(tab, diffs)
    print("CLEAN", clean, "seconds", round(time.time() - t0))


def stage_taiwan_test(out: Path, cfg) -> None:
    if not json.loads((out / "taiwan_validation_meta.json").read_text())["clean"]:
        raise SystemExit("Taiwan validation not clean: no test access")
    root = PREDS_TW / "REF-EG"
    if root.exists():
        raise SystemExit("artifacts/test_preds_taiwan/REF-EG exists: refusing a second access")
    root.mkdir(parents=True)
    dump(root / "access_started.json", {"git": git_hash(), "label": LABEL})
    te = access_test(ROOT, "posthoc:taiwan:REF-EG", GUARD_TW, git_hash())
    np.save(root / "y.npy", te.y)
    np.save(root / "a.npy", te.a)
    Xt = xm(te)
    ref, reg, s2, jt = [], [], [], []
    for s in SEEDS:
        d, sd = out / "taiwan" / f"seed_{s}", root / f"seed_{s}"
        sd.mkdir()
        b = lgb.Booster(model_file=str(tw_models(s) / "ref.txt"))
        art = joblib.load(d / "eg.joblib")
        pt = b.predict(Xt)
        zt = art["scaler"].transform(contrib(b, Xt)).astype("float32")
        st = pmf(art["eg"], zt) + 1e-6 * pt
        sv, pv = np.load(d / "sv.npy"), np.load(d / "pv.npy")
        fl = thr_flags(sv, st)
        np.save(sd / "pt.npy", pt)
        np.save(sd / "st.npy", st)
        np.save(sd / "flags.npy", fl)
        saved = np.load(PREDS_TW / "REF" / f"seed_{s}" / "pt.npy")
        dump(sd / "meta.json", {"threshold_from_val": float(flag_threshold(sv, 0.10)), "label": LABEL,
                                "max_abs_test_diff_vs_saved_REF": float(np.abs(pt - saved).max())})  # fmt: skip
        reg.append((pt, fl))
        ref.append((saved, thr_flags(pv, saved)))
        e = PREDS_TW / "S2-EG" / f"seed_{s}"
        s2.append((np.load(e / "pt.npy"), thr_flags(np.load(e / "sv_eg_10.npy"), np.load(e / "st_eg_10.npy"))))
        j = PREDS_TW / "JT_chosen" / f"seed_{s}"
        pjt = np.load(j / "pt.npy")
        jt.append((pjt, thr_flags(np.load(j / "pv.npy"), pjt)))
    dump(root / "access_done.json", {"arm": "REF-EG", "git": git_hash()})
    arms = {"REF": ref, "REF-S1": ref, "REF-EG": reg, "S2-EG": s2, "JT chosen": jt}
    tab, diffs = table(arms, te.y, te.a, 15)
    tab.to_csv(out / "taiwan_test_table.csv", index=False)
    diffs.to_csv(out / "taiwan_test_diffs.csv", index=False)
    show(tab, diffs)


def main() -> None:
    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/train.yaml")
    {"val": stage_val, "test": stage_test, "report": stage_report,
     "taiwan_val": stage_taiwan_val, "taiwan_test": stage_taiwan_test}[stage](out, cfg)  # fmt: skip


if __name__ == "__main__":
    main()
