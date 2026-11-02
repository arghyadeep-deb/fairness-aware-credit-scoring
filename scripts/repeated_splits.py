"""Repeated-splits robustness replication (post-hoc; declared in RUN_LOG before running).

python scripts/repeated_splits.py run <run_dir>      prep, fit, one guarded test access per arm per split
python scripts/repeated_splits.py report <run_dir>   tables and figure, from saved files only

The original hyperparameters and lambdas were chosen on split 42, whose rows can fall into the new
test sets, so this is a robustness check and NOT a clean out-of-sample test.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.data.prep import run_prep
from src.data.tensors import load_arrays
from src.eval.final import load_backbone, load_joint, logits_and_z
from src.eval.metrics import compute_metrics
from src.eval.threshold import apply_threshold, flag_threshold
from src.posthoc.eg import fit_eg, match_eg_weight, pmf
from src.posthoc.prior_correction import correct_logits
from src.run_arms import arm_cfg, sigmoid
from src.train import fit
from src.train_joint import fit_joint
from src.utils import load_config, seed_everything

SPLITS = [43, 44, 45, 46, 47]
MODEL_SEED, EPS, FR = 0, 0.01, 0.10
JT, JTA = (0.05, 2.0, 0.0), (0.05, 2.0, 0.25)
ARMS = ["A0", "A2", "S1", "S2-EG", "JT chosen", "JT-A k5 la0.25", "REF", "REF-EG"]
METRICS = ["auc_roc", "auc_pr", "worst_group_ece", "eod", "dir"]
ORIG_GUARD_SHA = "12036637fa50efc5def751618be6332fb56d4b2d18c05d71afc64d7c9dab5a9c"
GUARD = "artifacts/guard/repeated_splits.jsonl"
PREDS = ROOT / "artifacts" / "test_preds_repeated"
P2 = ROOT / "artifacts" / "runs" / "20261101T055729Z_phase2"
BASE = ROOT / "artifacts" / "repeated_splits"


def slug(a: str) -> str:
    return a.replace(" ", "_")


def split_cfg(cfg: dict, s: int) -> dict:
    c = copy.deepcopy(cfg)
    c["split"]["seed"] = s
    c["paths"]["processed"] = f"artifacts/repeated_splits/split_{s}/processed"
    c["paths"]["prep_artifacts"] = f"artifacts/repeated_splits/split_{s}/prep"
    c["paths"]["guard_log"] = GUARD
    return c


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def originals() -> dict:
    files = [ROOT / "artifacts/guard/test_access.jsonl", ROOT / "configs/selection_rule.yaml"]
    files += sorted((ROOT / "data/processed").glob("*")) + sorted((ROOT / "artifacts/prep").glob("*"))
    return {str(p.relative_to(ROOT)): sha_file(p) for p in files if p.is_file()}


def xm(a) -> np.ndarray:
    return np.concatenate([a.cont, a.cat], axis=1).astype("float32")


def contrib(b, X) -> np.ndarray:
    return b.predict(X, pred_contrib=True)[:, :-1].astype("float32")


def dump(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1), encoding="utf-8")


def hps() -> dict:
    return {a: json.loads((P2 / a / "tuning.json").read_text())["best_params"] for a in ("a0", "a2")}


def eg_fit(z, tr, cfg) -> dict:
    ec = cfg["exponentiated_gradient"]
    sc = StandardScaler().fit(z)
    zs = sc.transform(z).astype("float32")
    w, rate = match_eg_weight(zs, tr.y, tr.a, FR, EPS, ec["max_iter"], ec["logistic_max_iter"])
    eg = fit_eg(zs, tr.y, tr.a, w, EPS, ec["max_iter"], ec["logistic_max_iter"])
    return {"eg": eg, "scaler": sc, "w": float(w), "rate": float(rate)}


def save_val(d: Path, arm: str, pv, sv) -> None:
    v = d / "val" / slug(arm)
    v.mkdir(parents=True, exist_ok=True)
    np.save(v / "pv.npy", pv.astype("float32"))
    np.save(v / "sv.npy", sv.astype("float64"))


def load_model(arm, d, tr, cards, cfg, hp, device):
    if arm == "A0":
        return load_backbone("tabtransformer", tr, cards, cfg, hp["a0"], torch.load(d / "a0.pt"), device)
    if arm in ("A2", "S1", "S2-EG"):
        return load_backbone("tabtransformer", tr, cards, cfg, hp["a2"], torch.load(d / "a2.pt"), device)
    return load_joint(tr, cards, cfg, hp["a2"], torch.load(d / ("jt.pt" if arm == "JT chosen" else "jta.pt")), device)


def fit_split(s: int, out: Path, cfg0: dict, hp: dict, device) -> None:
    cfg = split_cfg(cfg0, s)
    d = out / f"split_{s}"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "fit_done.json").exists():
        return
    t0 = time.time()
    if not (BASE / f"split_{s}" / "prep" / "scaler.joblib").exists():
        rep = run_prep(cfg)
        dump(d / "prep_report.json", rep)
    tr, cards = load_arrays(cfg, "train")
    va, _ = load_arrays(cfg, "val")
    weights = json.loads((BASE / f"split_{s}" / "prep" / "class_weights.json").read_text())
    np.save(d / "val_y.npy", va.y)
    np.save(d / "val_a.npy", va.a)
    a0, a2 = arm_cfg("a0"), arm_cfg("a2")
    for name, fn in (
        ("a0.pt", lambda: fit(a0, cfg, hp["a0"], tr, va, cards, weights, MODEL_SEED, device)),
        ("a2.pt", lambda: fit(a2, cfg, hp["a2"], tr, va, cards, weights, MODEL_SEED, device)),
        ("jt.pt", lambda: fit_joint(cfg, hp["a2"], tr, va, cards, weights, MODEL_SEED, device, JT)),
        ("jta.pt", lambda: fit_joint(cfg, hp["a2"], tr, va, cards, weights, MODEL_SEED, device, JTA, schedule={"adv_steps": 5})),
    ):
        if not (d / name).exists():
            torch.save(fn()["state"], d / name)
            print(f"split {s} {name} {time.time() - t0:.0f}s", flush=True)
            torch.cuda.empty_cache()
    info = {}
    for arm in ("A0", "A2", "JT chosen", "JT-A k5 la0.25"):
        m = load_model(arm, d, tr, cards, cfg, hp, device)
        lv = logits_and_z(m, va, device)[0]
        save_val(d, arm, sigmoid(lv), sigmoid(lv))
        if arm == "A2":
            pv1 = sigmoid(correct_logits(lv, weights))
            save_val(d, "S1", pv1, pv1)
            ztr = logits_and_z(m, tr, device)[1]
            art = eg_fit(ztr, tr, cfg)
            zv = art["scaler"].transform(logits_and_z(m, va, device)[1]).astype("float32")
            save_val(d, "S2-EG", pv1, pmf(art["eg"], zv) + 1e-6 * pv1)
            joblib.dump({"eg": art["eg"], "scaler": art["scaler"]}, d / "eg_a2.joblib")
            info["S2-EG"] = {"w_pos": art["w"], "eg_train_flag_rate": art["rate"]}
            del ztr
        del m
        torch.cuda.empty_cache()
    print(f"split {s} EG on A2 done {time.time() - t0:.0f}s", flush=True)
    X, Xv = xm(tr), xm(va)
    pr = dict(cfg["reference"]["lightgbm"])
    stop = pr.pop("early_stopping_rounds")
    mdl = lgb.LGBMClassifier(**pr, objective="binary", random_state=MODEL_SEED, n_jobs=4, verbose=-1)
    mdl.fit(X, tr.y, eval_set=[(Xv, va.y)], categorical_feature=list(range(tr.cont.shape[1], X.shape[1])),
            callbacks=[lgb.early_stopping(stop, first_metric_only=True, verbose=False)])  # fmt: skip
    mdl.booster_.save_model(str(d / "ref.txt"), num_iteration=mdl.best_iteration_)
    b = lgb.Booster(model_file=str(d / "ref.txt"))
    pv = b.predict(Xv)
    save_val(d, "REF", pv, pv)
    art = eg_fit(contrib(b, X), tr, cfg)
    zv = art["scaler"].transform(contrib(b, Xv)).astype("float32")
    save_val(d, "REF-EG", pv, pmf(art["eg"], zv) + 1e-6 * pv)
    joblib.dump({"eg": art["eg"], "scaler": art["scaler"]}, d / "eg_ref.joblib")
    info["REF-EG"] = {"w_pos": art["w"], "eg_train_flag_rate": art["rate"]}
    ids = lambda p, n: set(pd.read_parquet(p / f"{n}_gender.parquet", columns=[cfg["id_column"]])[cfg["id_column"]])  # noqa: E731
    new_test = ids(BASE / f"split_{s}" / "processed", "test")
    orig = ROOT / "data" / "processed"
    info["overlap_new_test_share"] = {n: len(new_test & ids(orig, n)) / len(new_test) for n in ("train", "val", "test")}
    info["seconds"] = round(time.time() - t0)
    dump(d / "fit_done.json", info)
    print(f"split {s} fit done", info, flush=True)
    gc.collect()


def access_split(s: int, out: Path, cfg0: dict, hp: dict, device) -> None:
    cfg = split_cfg(cfg0, s)
    d = out / f"split_{s}"
    info = json.loads((d / "fit_done.json").read_text())
    tr, cards = load_arrays(cfg, "train")
    weights = json.loads((BASE / f"split_{s}" / "prep" / "class_weights.json").read_text())
    for arm in ARMS:
        root = PREDS / f"split_{s}" / slug(arm)
        if (root / "access_done.json").exists():
            continue
        if root.exists():
            raise SystemExit(f"HARD STOP: {root} started but unfinished; refusing a second access")
        if arm in info and abs(info[arm]["eg_train_flag_rate"] - FR) >= 0.01:
            dump(d / f"gate_failed_{slug(arm)}.json", info[arm])
            print(f"split {s} {arm}: EG gate failed, no test access", info[arm], flush=True)
            continue
        root.mkdir(parents=True)
        os.environ["ALLOW_TEST"] = "1"
        try:
            te, _ = load_arrays(cfg, "test", purpose=f"repeated:split{s}:{arm}")
        finally:
            os.environ.pop("ALLOW_TEST", None)
        np.save(root / "y.npy", te.y)
        np.save(root / "a.npy", te.a)
        if arm in ("REF", "REF-EG"):
            b = lgb.Booster(model_file=str(d / "ref.txt"))
            Xt = xm(te)
            pt = b.predict(Xt)
            st = pt
            if arm == "REF-EG":
                art = joblib.load(d / "eg_ref.joblib")
                st = pmf(art["eg"], art["scaler"].transform(contrib(b, Xt)).astype("float32")) + 1e-6 * pt
        else:
            m = load_model(arm, d, tr, cards, cfg, hp, device)
            lt, zt = logits_and_z(m, te, device)
            pt = sigmoid(correct_logits(lt, weights)) if arm in ("S1", "S2-EG") else sigmoid(lt)
            st = pt
            if arm == "S2-EG":
                art = joblib.load(d / "eg_a2.joblib")
                st = pmf(art["eg"], art["scaler"].transform(zt).astype("float32")) + 1e-6 * pt
            del m
            torch.cuda.empty_cache()
        sv = np.load(d / "val" / slug(arm) / "sv.npy")
        np.save(root / "pt.npy", np.asarray(pt, dtype="float32"))
        np.save(root / "st.npy", np.asarray(st, dtype="float64"))
        np.save(root / "flags.npy", apply_threshold(st, flag_threshold(sv, FR)))
        dump(root / "access_done.json", {"arm": arm, "split": s})
        print(f"split {s} {arm} accessed", flush=True)
        del te
        gc.collect()
    del tr
    gc.collect()


def stage_run(out: Path) -> None:
    before = originals()
    assert before["artifacts\\guard\\test_access.jsonl" if os.name == "nt" else "artifacts/guard/test_access.jsonl"] == ORIG_GUARD_SHA
    dump(out / "originals_before.json", before)
    cfg0 = load_config("configs/train.yaml")
    hp = hps()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    for s in SPLITS:
        seed_everything(MODEL_SEED)
        fit_split(s, out, cfg0, hp, device)
        access_split(s, out, cfg0, hp, device)
    assert originals() == before
    print("RUN DONE, originals unchanged", flush=True)


def stage_report(out: Path) -> None:
    cfg = load_config("configs/train.yaml")
    eb = cfg["evaluation"]["ece_bins"]
    rows = []
    for s in SPLITS:
        d = out / f"split_{s}"
        vy, va = np.load(d / "val_y.npy"), np.load(d / "val_a.npy")
        for arm in ARMS:
            root = PREDS / f"split_{s}" / slug(arm)
            if not (root / "access_done.json").exists():
                rows.append({"split": s, "arm": arm, "set": "test", "status": "not accessed (EG gate)"})
                continue
            v = d / "val" / slug(arm)
            pv, sv = np.load(v / "pv.npy"), np.load(v / "sv.npy")
            y, a = np.load(root / "y.npy"), np.load(root / "a.npy")
            pt, st, fl = np.load(root / "pt.npy"), np.load(root / "st.npy"), np.load(root / "flags.npy").astype(bool)
            mv = compute_metrics(vy, pv, va, apply_threshold(sv, flag_threshold(sv, FR)), eb)
            mt = compute_metrics(y, pt, a, fl, eb)
            base = {"split": s, "arm": arm, "status": "ok", "eg_score_auc_pr": float(average_precision_score(y, st))}
            rows.append({**base, "set": "val", **{k: mv[k] for k in [*METRICS, "recall"]}})
            rows.append({**base, "set": "test", "flag_rate": float(fl.mean()), **{k: mt[k] for k in [*METRICS, "recall"]}})
            if arm == "REF":
                rows.append({**base, "arm": "REF-S1", "set": "val", **{k: mv[k] for k in [*METRICS, "recall"]}})
                rows.append({**base, "arm": "REF-S1", "set": "test", "flag_rate": float(fl.mean()), **{k: mt[k] for k in [*METRICS, "recall"]}})
    df = pd.DataFrame(rows)
    tabs = ROOT / "artifacts" / "tables"
    df.to_csv(tabs / "repeated_splits.csv", index=False)
    te = df[(df.set == "test") & (df.status == "ok")]
    summ = []
    for arm, g in te.groupby("arm", sort=False):
        r = {"arm": arm, "n_splits": len(g)}
        for m in [*METRICS, "recall"]:
            r |= {f"{m}_mean": g[m].mean(), f"{m}_sd": g[m].std(ddof=1), f"{m}_min": g[m].min(), f"{m}_max": g[m].max()}
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(tabs / "repeated_splits_summary.csv", index=False)
    rng = np.random.default_rng(0)
    pairs = []
    for arm, ref in (("JT chosen", "S2-EG"), ("REF-EG", "S2-EG"), ("JT chosen", "S1"), ("REF-EG", "REF")):
        a_, b_ = te[te.arm == arm].set_index("split"), te[te.arm == ref].set_index("split")
        ix = a_.index.intersection(b_.index)
        for m in [*METRICS, "recall"]:
            dd = (a_.loc[ix, m] - b_.loc[ix, m]).to_numpy()
            bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            pairs.append({"arm": arm, "minus": ref, "metric": m, "mean_diff": dd.mean(), "lo": lo, "hi": hi,
                          "n_positive": int((dd > 0).sum()), "n": len(dd), "per_split": " ".join(f"{x:+.4f}" for x in dd)})  # fmt: skip
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(tabs / "repeated_splits_paired.csv", index=False)
    vt = df[df.status == "ok"].pivot_table(index=["split", "arm"], columns="set", values="eod").reset_index()
    vt["test_minus_val"] = vt["test"] - vt["val"]
    opt = vt.groupby("arm", sort=False).agg(val_eod=("val", "mean"), test_eod=("test", "mean"),
                                             gap_mean=("test_minus_val", "mean"), gap_sd=("test_minus_val", "std"),
                                             n_test_higher=("test_minus_val", lambda x: int((x > 0).sum()))).reset_index()  # fmt: skip
    opt.to_csv(tabs / "repeated_splits_val_vs_test_eod.csv", index=False)
    main_arms = ["S2-EG", "JT chosen", "JT-A k5 la0.25", "REF-EG", "REF", "A2"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a_, m, lab in ((ax[0], "eod", "EOD (test, lower better)"), (ax[1], "auc_pr", "AUC-PR (test, higher better)")):
        for arm in main_arms:
            g = te[te.arm == arm].sort_values("split")
            a_.plot(g.split, g[m], "o-", label=arm, lw=1.2, ms=4)
        a_.set_xticks(SPLITS)
        a_.set_xlabel("split seed")
        a_.set_ylabel(lab)
    ax[0].legend(fontsize=8)
    fig.suptitle("Repeated splits, frozen settings from split 42: a robustness check, not a clean out-of-sample test", fontsize=10)
    plt.tight_layout()
    plt.savefig(ROOT / "artifacts" / "figures" / "repeated_splits.png", dpi=160)
    plt.close()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)
    print(summ[["arm", "n_splits"] + [f"{m}_{x}" for m in METRICS for x in ("mean", "sd", "min", "max")]].round(4).to_string(index=False))
    print(pairs.round(4).to_string(index=False))
    print(opt.round(4).to_string(index=False))
    print("guard sha ok:", sha_file(ROOT / "artifacts/guard/test_access.jsonl") == ORIG_GUARD_SHA)


def main() -> None:
    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    {"run": stage_run, "report": stage_report}[stage](out)


if __name__ == "__main__":
    main()
