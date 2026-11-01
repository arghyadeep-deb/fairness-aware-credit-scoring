"""Phase 6 final verification: the TEST set, once per arm (declared in RUN_LOG commit f6d3e3d).

python scripts/phase6_test.py access  <p2> <p4> <p4b> <p6> [arm ...]
python scripts/phase6_test.py analyze <p2> <p4> <p4b> <p6> [arm ...]

access : ALLOW_TEST=1 is set only around ONE load_arrays(test) per arm (purpose "phase6:<arm>", one guard-log
         entry per arm). Every seed's test predictions and everything else that depends on test rows are
         written to artifacts/test_preds/<arm>/seed_<s>/ IMMEDIATELY (pt, st, flags, val/test z for probe
         arms, matched-pair probability) and the test arrays are saved once to artifacts/test_preds/
         test_arrays.npz. The flag threshold is the 90th percentile of the arm's VALIDATION scores.
analyze: reads ONLY those files (never the test loader). Metrics, 1000 paired bootstrap resamples
         (n_jobs capped), z-probe after, SHAP probe, matched pairs -> <p6>/test/<arm>/seed_<s>/.
Resumable per arm in both modes. A second test access for an arm (after a crash) must be logged as an
exception in RUN_LOG: access_started.json without access_done.json marks that case.
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
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.tensors import Arrays, load_arrays  # noqa: E402
from src.eval.bootstrap import bootstrap_metrics  # noqa: E402
from src.eval.final import (  # noqa: E402
    delta_stats,
    load_backbone,
    load_joint,
    logits_and_z,
    matched_pairs,
    shap_probe,
    shap_vectors,
    z_probe_test,
)
from src.eval.metrics import compute_metrics  # noqa: E402
from src.eval.threshold import apply_threshold, flag_threshold  # noqa: E402
from src.posthoc.eg import fit_eg, pmf  # noqa: E402
from src.posthoc.phase5 import fairlearn_fit_predict, pergroup_fit  # noqa: E402
from src.posthoc.prior_correction import correct_logits  # noqa: E402
from src.run_arms import arm_cfg, save_json, sigmoid  # noqa: E402
from src.run_joint import load_setup, tag  # noqa: E402
from src.train import fit  # noqa: E402
from src.utils import git_hash, load_config  # noqa: E402

log = logging.getLogger("phase6")
S3, S5 = [0, 1, 2], [0, 1, 2, 3, 4]
JTA = [(k, la) for k in (3, 5) for la in (0.1, 0.25, 0.5)]
PROBE_ARMS = {"A2", "JT chosen", "JT tuned"} | {f"JT-A k{k} la{la}" for k, la in JTA}
SHAP_ARMS = set(PROBE_ARMS)
MP_ARMS = PROBE_ARMS | {"H1", "H2", "H3"}
PREDS = ROOT / "artifacts" / "test_preds"
NJOBS = 4


def arm_table(p2, p4, p4b, p6):
    """name -> dict(family, deployable, seeds, ...)."""
    t = {}
    for n in ("A0", "A1", "A1b", "A2", "A3", "A4", "A5", "G1"):
        t[n] = {"family": "nn", "src": n.lower(), "deployable": True, "seeds": S3}
    t["GI"] = {"family": "gi", "deployable": False, "seeds": S3}
    t["REF"] = {"family": "ref", "deployable": True, "seeds": S3}
    t["S1"] = {"family": "s1", "deployable": True, "seeds": S3}
    t["S2-EG"] = {"family": "eg", "deployable": True, "seeds": S3}
    t["S2-GT"] = {"family": "gt", "deployable": False, "seeds": S3}
    t["per-group Platt"] = {"family": "platt", "deployable": False, "seeds": S3}
    t["per-group isotonic"] = {"family": "isotonic", "deployable": False, "seeds": S3}
    t["ThresholdOptimizer"] = {"family": "to", "deployable": False, "seeds": S3}
    t["JT chosen"] = {
        "family": "jt",
        "dir": p4 / "jt" / "f0p05_c2_a0",
        "hp": "a2",
        "deployable": True,
        "seeds": S3,
    }
    t["JT tuned"] = {"family": "jt", "dir": p4b / "jt_tuned" / "f0p05_c2_a0", "hp": "tuned", "deployable": True, "seeds": S3}  # fmt: skip
    for i, lf in enumerate((0.05, 0.1, 0.25), 1):
        t[f"H{i}"] = {"family": "h", "dir": p4b / f"h{i}_raw" / tag((lf, 0.0, 0.0)), "hp": "tuned", "deployable": True, "seeds": S3}  # fmt: skip
    for k, la in JTA:
        t[f"JT-A k{k} la{la}"] = {"family": "jt", "dir": p6 / f"jta_k{k}" / tag((0.05, 2.0, la)), "hp": "a2", "deployable": True, "seeds": S5}  # fmt: skip
    return t


def slug(name: str) -> str:
    return name.replace(" ", "_")


def build_model(spec, name, s, cfg, ctx):
    """The trained (train-time) model for an arm/seed; no test data involved."""
    p2, hp_a2, hp_tuned, train, cards, device = (
        ctx[k] for k in ("p2", "hp_a2", "hp_tuned", "train", "cards", "device")
    )
    fam = spec["family"]
    if fam == "nn":
        hp = json.loads((p2 / spec["src"] / "tuning.json").read_text())["best_params"]
        return load_backbone(arm_cfg(spec["src"])["model"], train, cards, cfg, hp, torch.load(p2 / spec["src"] / f"seed_{s}" / "best.pt"), device)  # fmt: skip
    if fam in ("jt", "h"):
        hp = hp_a2 if spec["hp"] == "a2" else hp_tuned
        return load_joint(
            train, cards, cfg, hp, torch.load(spec["dir"] / f"seed_{s}" / "best.pt"), device
        )
    return load_backbone("tabtransformer", train, cards, cfg, hp_a2, torch.load(p2 / "a2" / f"seed_{s}" / "best.pt"), device)  # fmt: skip


def make_ctx(cfg, p2, p4b):
    device, train, val, cards, weights, _ = load_setup(cfg)
    return {"p2": p2, "device": device, "train": train, "val": val, "cards": cards, "weights": weights,
            "hp_a2": json.loads((p2 / "a2" / "tuning.json").read_text())["best_params"],
            "hp_tuned": json.loads((p4b / "jt_tuning.json").read_text())["best_params"]}  # fmt: skip


# ------------------------------------------------------------------------------------ access
def access(name, spec, cfg, ctx, p6):
    fr = cfg["evaluation"]["flag_rate"]
    train, val, cards, weights, device = (
        ctx[k] for k in ("train", "val", "cards", "weights", "device")
    )
    root = PREDS / slug(name)
    root.mkdir(parents=True, exist_ok=True)
    started = root / "access_started.json"
    exception = started.exists()  # started earlier but never finished: this is a SECOND access
    save_json(started, {"utc": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_hash(), "second_access_exception": exception})  # fmt: skip
    if exception:
        log.warning(
            "SECOND TEST ACCESS for %s (earlier attempt did not finish): log as exception", name
        )
    fam = spec["family"]
    # ---- THE single test access for this arm ----
    os.environ["ALLOW_TEST"] = "1"
    try:
        test, _ = load_arrays(cfg, "test", include_gender=fam == "gi", purpose=f"phase6:{name}")
    finally:
        os.environ.pop("ALLOW_TEST", None)
    if fam != "gi" and not (PREDS / "test_arrays.npz").exists():
        np.savez(
            PREDS / "test_arrays.npz",
            cont=test.cont.astype("float32"),
            cat=test.cat,
            y=test.y,
            a=test.a,
        )
    np.save(root / "y.npy", test.y)
    np.save(root / "a.npy", test.a)
    for s in spec["seeds"]:
        d = root / f"seed_{s}"
        d.mkdir(parents=True, exist_ok=True)
        if (d / "meta.json").exists():
            continue
        flags_override, extras = None, {}
        if fam in ("nn", "jt", "h", "s1", "eg", "platt", "isotonic", "to", "gt"):
            m = build_model(spec, name, s, cfg, ctx)
            lv, zv = logits_and_z(m, val, device)
            lt, zt = logits_and_z(m, test, device)
            if fam in ("h", "s1", "eg", "gt", "to"):
                lv, lt = correct_logits(lv, weights), correct_logits(lt, weights)
            pv, pt = sigmoid(lv), sigmoid(lt)
            sv, st = pv, pt
            if fam in ("platt", "isotonic"):
                ap = pergroup_fit(val.y, lv, val.a, fam)
                pv, pt = ap(lv, val.a), ap(lt, test.a)
                sv, st = pv, pt
            if fam == "eg":
                ztr = logits_and_z(m, train, device)[1]
                sc = StandardScaler().fit(ztr)
                w = json.loads((p6 / "s2_eg_v2" / f"seed_{s}" / "pred_meta.json").read_text())[
                    "w_pos"
                ]
                ec = cfg["exponentiated_gradient"]
                eg = fit_eg(
                    sc.transform(ztr),
                    train.y,
                    train.a,
                    w,
                    0.01,
                    ec["max_iter"],
                    ec["logistic_max_iter"],
                )
                sv = pmf(eg, sc.transform(zv)) + 1e-6 * pv
                st = pmf(eg, sc.transform(zt)) + 1e-6 * pt
                del ztr
            if fam == "gt":
                meta = json.loads(
                    (ctx["p2"] / "s2_gt" / f"seed_{s}" / "pred_meta.json").read_text()
                )
                flags_override = pt >= np.where(
                    test.a == 0, meta["threshold_f"], meta["threshold_m"]
                )
            if fam == "to":
                flags_override = fairlearn_fit_predict(val.y, pv, val.a, pt, test.a, seed=s)
            if name in PROBE_ARMS:
                np.save(d / "zv.npy", zv.astype("float32"))
                np.save(d / "zt.npy", zt.astype("float32"))
            if name in MP_ARMS and s == 0:
                np.save(d / "pq.npy", sigmoid(correct_logits(lt, weights)) if name == "A2" else pt)
            del m
        elif fam == "ref":
            pr = dict(cfg["reference"]["lightgbm"])
            stop = pr.pop("early_stopping_rounds")
            X, Xv, Xt = (
                np.concatenate([a.cont, a.cat], axis=1).astype("float32")
                for a in (train, val, test)
            )
            cats = list(range(train.cont.shape[1], X.shape[1]))
            mdl = lgb.LGBMClassifier(**pr, objective="binary", random_state=s, n_jobs=8, verbose=-1)
            mdl.fit(X, train.y, eval_set=[(Xv, val.y)], categorical_feature=cats,
                    callbacks=[lgb.early_stopping(stop, first_metric_only=True, verbose=False)])  # fmt: skip
            pv, pt = mdl.predict_proba(Xv)[:, 1], mdl.predict_proba(Xt)[:, 1]
            sv, st = pv, pt
            extras["retrain_max_abs_val_diff"] = float(np.abs(np.load(ctx["p2"] / "ref" / f"seed_{s}" / "val_prob.npy") - pv).max())  # fmt: skip
            del X, Xv, Xt, mdl
        else:  # gi
            train_g = load_arrays(cfg, "train", include_gender=True)[0]
            val_g = load_arrays(cfg, "val", include_gender=True)[0]
            src = json.loads((ctx["p2"] / "gi_source.json").read_text())
            armc = {**arm_cfg(src["source_arm"]), "name": "gi"}
            r = fit(armc, cfg, src["hp"], train_g, val_g, cards, weights, s, device)
            m = load_backbone(armc["model"], train_g, cards, cfg, src["hp"], r["state"], device)
            lv, _ = logits_and_z(m, val_g, device)
            lt, _ = logits_and_z(m, test, device)
            pv, pt = sigmoid(lv), sigmoid(lt)
            sv, st = pv, pt
            extras["retrain_max_abs_val_diff"] = float(np.abs(np.load(ctx["p2"] / "gi" / f"seed_{s}" / "val_prob.npy") - pv).max())  # fmt: skip
            del m, r, train_g, val_g
        thr = flag_threshold(sv, fr)
        flags = flags_override if flags_override is not None else apply_threshold(st, thr)
        np.save(d / "pt.npy", pt)
        np.save(d / "st.npy", st)
        np.save(d / "flags.npy", flags)
        save_json(d / "meta.json", {"threshold_from_val": float(thr), **extras})
        log.info("access %s s%d saved", name, s)
        gc.collect()
        torch.cuda.empty_cache()
    save_json(
        root / "access_done.json",
        {"arm": name, "git": git_hash(), "second_access_exception": exception},
    )
    del test
    gc.collect()


# ------------------------------------------------------------------------------------ analyze
def analyze(name, spec, cfg, ctx, p6):
    eb = cfg["evaluation"]["ece_bins"]
    train, val, device, weights = (ctx[k] for k in ("train", "val", "device", "weights"))
    root = PREDS / slug(name)
    out = p6 / "test" / slug(name)
    npz = np.load(PREDS / "test_arrays.npz")
    te = Arrays(
        npz["cont"],
        npz["cat"],
        npz["y"],
        npz["a"],
        train.cont_names,
        train.cat_names,
        train.binary_names,
    )
    y, a = np.load(root / "y.npy"), np.load(root / "a.npy")
    bg_ix = np.random.default_rng(0).choice(len(train.y), 100, replace=False)
    tr_ix = np.sort(np.random.default_rng(1).choice(len(train.y), 4000, replace=False))
    te_ix = np.sort(np.random.default_rng(2).choice(len(te.y), 4000, replace=False))
    t0 = time.time()
    for s in spec["seeds"]:
        sd, d = root / f"seed_{s}", out / f"seed_{s}"
        d.mkdir(parents=True, exist_ok=True)
        pt, flags = np.load(sd / "pt.npy"), np.load(sd / "flags.npy")
        meta = json.loads((sd / "meta.json").read_text())
        extras = {k: v for k, v in meta.items() if k != "threshold_from_val"}
        if name in PROBE_ARMS or (name in MP_ARMS and s == 0):
            m = build_model(spec, name, s, cfg, ctx)
        if name in PROBE_ARMS:
            ztr = logits_and_z(m, train, device)[1]
            extras["z_probe_after"] = z_probe_test(cfg, ztr, train.a, np.load(sd / "zv.npy"), val.a, np.load(sd / "zt.npy"), a, s)  # fmt: skip
        if name in MP_ARMS and s == 0:
            pq = np.load(sd / "pq.npy")
            pairs = matched_pairs(te)
            extras["matched_pair"] = delta_stats(pq, *pairs)
            np.save(d / "matched_delta.npy", np.abs(pq[pairs[0]] - pq[pairs[1]]))
        if name in SHAP_ARMS and s == 0:
            sv_tr, meth = shap_vectors(m, train.subset(bg_ix), train.subset(tr_ix), device)
            sv_te, _ = shap_vectors(m, train.subset(bg_ix), te.subset(te_ix), device)
            extras["shap_probe_test_auc"] = shap_probe(
                cfg, sv_tr, train.a[tr_ix], sv_te, a[te_ix], s
            )
            extras["shap_method"] = meth
            np.save(d / "shap_train.npy", sv_tr.astype("float32"))
            np.save(d / "shap_test.npy", sv_te.astype("float32"))
        m_ = compute_metrics(y, pt, a, flags, eb)
        boot = bootstrap_metrics(y, pt, a, flags, 1000, 0, eb, n_jobs=NJOBS)
        np.save(d / "test_prob.npy", pt)
        np.save(d / "test_flags.npy", flags)
        np.save(d / "boot.npy", boot)
        save_json(d / "metrics.json", {"metrics": m_, "threshold_from_val": meta["threshold_from_val"],
                                        "test_flag_rate": float(np.mean(flags)),
                                        "deployable": spec["deployable"], **extras})  # fmt: skip
        log.info("%s s%d: AUC-ROC %.4f AUC-PR %.4f EOD %.4f wECE %.4f flag %.3f", name, s, m_["auc_roc"], m_["auc_pr"], m_["eod"], m_["worst_group_ece"], np.mean(flags))  # fmt: skip
        for v in ("m", "ztr"):
            locals().pop(v, None)
        gc.collect()
        torch.cuda.empty_cache()
    save_json(out / "done.json", {"arm": name, "deployable": spec["deployable"], "git": git_hash(), "seconds": time.time() - t0})  # fmt: skip


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    mode = sys.argv[1]
    p2, p4, p4b, p6 = (Path(x) for x in sys.argv[2:6])
    only = sys.argv[6:]
    cfg = load_config("configs/train.yaml")
    ctx = make_ctx(cfg, p2, p4b)
    table = arm_table(p2, p4, p4b, p6)
    for name in only or list(table):
        spec = table[name]
        if mode == "access":
            if (PREDS / slug(name) / "access_done.json").exists():
                log.info("skip %s (access done)", name)
                continue
            access(name, spec, cfg, ctx, p6)
        else:
            if (p6 / "test" / slug(name) / "done.json").exists() or not (PREDS / slug(name) / "access_done.json").exists():  # fmt: skip
                log.info("skip %s", name)
                continue
            analyze(name, spec, cfg, ctx, p6)
    log.info("finished %s", mode)


if __name__ == "__main__":
    main()
