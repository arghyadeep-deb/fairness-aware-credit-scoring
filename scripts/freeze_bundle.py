"""Phase 7 step 1: freeze the serving bundle (artifacts/bundle/v1/) with a sha256 manifest.

python scripts/freeze_bundle.py <p2> <p4> <p4b> <p6>

Served model = the rule-selected JT chosen (0.05, 2, 0), untuned, the seed with the best VALIDATION
AUC-PR (chosen on validation, not on test). Reads NO Home Credit test loader: dashboard numbers, matched-pair
pool and examples come from artifacts/test_preds/ (saved arrays) and the Phase 6 report outputs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from api.bundle import Bundle, bundle_sha, load_bundle, sha256_file  # noqa: E402
from api.service import predict_arrays  # noqa: E402
from src.data.prep import Preprocessor  # noqa: E402
from src.data.tensors import Arrays, load_arrays  # noqa: E402
from src.eval.bootstrap import bootstrap_metrics  # noqa: E402
from src.eval.final import delta_stats, matched_pairs  # noqa: E402
from src.eval.metrics import METRIC_NAMES, compute_metrics  # noqa: E402
from src.utils import git_hash, load_config  # noqa: E402

VERSION = "v1"
JT_TAG = "f0p05_c2_a0"
LAMBDAS = (0.05, 2.0, 0.0)
SEEDS = [0, 1, 2]
OUT = ROOT / "artifacts" / "bundle" / VERSION


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def pick_seed(p4: Path) -> tuple[int, dict]:
    vals = {s: json.loads((p4 / "jt" / JT_TAG / f"seed_{s}" / "summary.json").read_text())["val_auc_pr"] for s in SEEDS}
    s = max(vals, key=vals.get)
    return s, vals


def inverse_example(pre: Preprocessor, cont: np.ndarray, cat: np.ndarray, spec: dict) -> dict | None:
    """Scaled arrays -> raw ApplicantRecord dict (no gender). Returns None if not representable."""
    names = spec["cont_names"]
    num = pre.groups["numeric"]
    x = pd.Series(cont, index=names)
    raw = dict(zip(num, pre.scaler.inverse_transform(x[num].to_numpy(dtype="float64")[None, :])[0]))
    rec = {
        "age_years": round(float(raw["AGE_YEARS"]), 1),
        "cnt_children": int(round(raw["CNT_CHILDREN"])),
        "cnt_fam_members": max(1, int(round(raw["CNT_FAM_MEMBERS"]))),
        "amt_income_total": round(float(raw["AMT_INCOME_TOTAL"]), 0),
        "amt_credit": round(float(raw["AMT_CREDIT"]), 0),
        "amt_annuity": round(float(raw["AMT_ANNUITY"]), 0),
        "amt_goods_price": round(float(raw["AMT_GOODS_PRICE"]), 0),
        "region_population_relative": round(float(raw["REGION_POPULATION_RELATIVE"]), 5),
        "years_employed": None if x["EMPLOYED_MISSING"] > 0.5 else round(max(0.0, float(raw["YEARS_EMPLOYED"])), 1),
        "flag_own_car": "Y" if x["FLAG_OWN_CAR"] > 0.5 else "N",
        "flag_own_realty": "Y" if x["FLAG_OWN_REALTY"] > 0.5 else "N",
    }
    for k in (1, 2, 3):
        rec[f"ext_source_{k}"] = None if x[f"EXT_SOURCE_{k}_MISSING"] > 0.5 else round(min(1.0, max(0.0, float(raw[f"EXT_SOURCE_{k}"]))), 4)  # fmt: skip
    mapping = {"NAME_FAMILY_STATUS": "name_family_status", "NAME_EDUCATION_TYPE": "name_education_type",
               "NAME_INCOME_TYPE": "name_income_type", "NAME_HOUSING_TYPE": "name_housing_type",
               "OCCUPATION_TYPE": "occupation_type", "ORGANIZATION_TYPE": "organization_type"}  # fmt: skip
    for c, v in zip(spec["cat_names"], cat):
        inv = {i: k for k, i in pre.encoders[c].items()}
        if v == 0:
            if c in ("OCCUPATION_TYPE", "ORGANIZATION_TYPE"):
                rec[mapping[c]] = None
            else:
                return None
        else:
            rec[mapping[c]] = inv[int(v)]
    ok = 18 <= rec["age_years"] <= 100 and rec["cnt_children"] <= 20 and rec["cnt_fam_members"] <= 25
    ok = ok and min(rec["amt_income_total"], rec["amt_credit"], rec["amt_annuity"], rec["amt_goods_price"]) > 0
    ok = ok and 0 <= rec["region_population_relative"] <= 1
    ok = ok and (rec["years_employed"] is None or rec["years_employed"] <= 60)
    return rec if ok else None


def main() -> None:
    p2, p4, p4b, p6 = (Path(x) for x in sys.argv[1:5])
    cfg = load_config("configs/train.yaml")
    data_cfg = load_config("configs/data.yaml")
    if OUT.exists():
        shutil.rmtree(OUT)  # only our own earlier bundle directory
    seed, vals = pick_seed(p4)
    hp = json.loads((p2 / "a2" / "tuning.json").read_text())["best_params"]
    prep_dir = ROOT / "artifacts" / "prep"
    order = json.loads((prep_dir / "feature_order.json").read_text())
    card = json.loads((prep_dir / "cardinalities.json").read_text())
    cont_names, cat_names = order["numeric"] + order["binary"], order["categorical"]
    # ---- model: the backbone of the JT checkpoint (Head B is never served) ----
    state = torch.load(p4 / "jt" / JT_TAG / f"seed_{seed}" / "best.pt", map_location="cpu")
    bb = {k[len("backbone."):]: v.float().cpu() for k, v in state.items() if k.startswith("backbone.")}
    (OUT / "model").mkdir(parents=True)
    torch.save(bb, OUT / "model" / "backbone.pt")
    spec = {"n_cont": len(cont_names), "cards": [card[c] for c in cat_names], "model_defaults": cfg["model_defaults"],
            "hp": hp, "cont_names": cont_names, "cat_names": cat_names, "lambdas": list(LAMBDAS), "seed": seed,
            "val_auc_pr_by_seed": vals, "head": "A (default probability); Head B never served",
            "probability": "sigmoid(logit), no post-hoc correction (learned calibration term lambda_c=2)"}  # fmt: skip
    write_json(OUT / "model" / "model_spec.json", spec)
    # ---- preprocessing artifacts (train-fitted) ----
    (OUT / "prep").mkdir()
    for n in ("scaler.joblib", "encoders.json", "cardinalities.json", "feature_order.json", "class_weights.json", "medians.json"):  # fmt: skip
        shutil.copy2(prep_dir / n, OUT / "prep" / n)
    (OUT / "prep" / "data_cfg.yaml").write_text(yaml.safe_dump(data_cfg), encoding="utf-8")
    # ---- SHAP background: 100 TRAIN rows ----
    train, _ = load_arrays(cfg, "train")
    ix = np.sort(np.random.default_rng(0).choice(len(train.y), 100, replace=False))
    np.savez(OUT / "shap_background.npz", cont=train.cont[ix].astype("float32"), cat=train.cat[ix].astype("int64"))
    # ---- matched-pair pool = saved test arrays (no test loader) ----
    tz = np.load(ROOT / "artifacts" / "test_preds" / "test_arrays.npz")
    pc, pk, pa, py = tz["cont"].astype("float32"), tz["cat"].astype("int64"), tz["a"].astype("int8"), tz["y"]
    np.savez(OUT / "matched_pool.npz", cont=pc, cat=pk, a=pa)
    # ---- served path used for every precomputed number (fp32 CPU, same as the API) ----
    tmp = Bundle(OUT, {"model_version": "tmp", "bundle_sha": "tmp", "risk_bands": {}}, None, None, spec,
                 None, None, pc, pk, pa)  # fmt: skip
    from api.bundle import build_model  # noqa: E402 (re-exported import)

    model = build_model("tabtransformer", spec["n_cont"], spec["cards"], spec["model_defaults"], spec["hp"])
    model.load_state_dict(bb)
    model.eval()
    tmp.model = model
    p_pool = np.concatenate([predict_arrays(tmp, pc[i : i + 8192], pk[i : i + 8192]) for i in range(0, len(pc), 8192)])
    saved = np.load(ROOT / "artifacts" / "test_preds" / "JT_chosen" / f"seed_{seed}" / "pt.npy")
    gap_bf16 = float(np.abs(saved - p_pool).max())  # Phase 6 scored with bf16 autocast on GPU
    # ---- fairness / calibration at the shared decision rule ----
    meta = json.loads((ROOT / "artifacts/test_preds/JT_chosen" / f"seed_{seed}" / "meta.json").read_text())
    thr = meta["threshold_from_val"]
    y, a = py, pa
    flags = (saved >= thr)  # the Phase 6 rule: val 90th percentile threshold applied to test scores
    m = compute_metrics(y, saved, a, flags, 15)
    boot = bootstrap_metrics(y, saved, a, flags, 1000, 0, 15, n_jobs=4, chunk_size=100)
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    ci = {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(METRIC_NAMES)}
    fair = {"threshold_rule": "flag applicants scoring above the 90th percentile of VALIDATION scores (shared rule)",
            "threshold": thr, "dpd": m["dpd"], "eod": m["eod"], "dir": m["dir"],
            "per_group": {"M": {"tpr": m["tpr_m"], "fpr": m["fpr_m"]}, "F": {"tpr": m["tpr_f"], "fpr": m["fpr_f"]}},
            "ci95": ci, "dataset": "home_credit_test", "served_seed": seed, "n_test": int(len(y)),
            "flag_rate_test": float(flags.mean()), "auc_roc": m["auc_roc"], "auc_pr": m["auc_pr"], "recall": m["recall"]}  # fmt: skip
    write_json(OUT / "json" / "fairness.json", fair)
    qs = np.quantile(saved, np.linspace(0, 1, 11))
    groups = {}
    for g, name in ((1, "M"), (0, "F")):
        sel = a == g
        bins = []
        which = np.clip(np.searchsorted(qs, saved[sel], side="right") - 1, 0, 9)
        for b_ in range(10):
            mk = which == b_
            if mk.any():
                bins.append({"mean_pred": float(saved[sel][mk].mean()), "frac_pos": float(y[sel][mk].mean()), "count": int(mk.sum())})  # fmt: skip
        groups[name] = {"bins": bins, "ece": m["ece_m"] if g == 1 else m["ece_f"]}
    write_json(OUT / "json" / "calibration.json", {"groups": groups, "brier": m["brier"], "worst_group_ece": m["worst_group_ece"],
                                                    "dataset": "home_credit_test", "bin_rule": "10 equal-mass bins of the pooled score"})  # fmt: skip
    # ---- matched-pair statistics for the served model ----
    pool_arr = Arrays(pc, pk, y, a, cont_names, cat_names, order["binary"])
    q, mm = matched_pairs(pool_arr)
    ds = delta_stats(p_pool, q, mm)
    write_json(OUT / "json" / "matched_stats.json", {"mean_abs_delta": ds["mean"], **ds, "n_pairs": int(len(q)),
                                                      "note": "nearest applicant of the other gender in the test pool; model never sees gender"})  # fmt: skip
    # ---- proxy audit (before from Phase 1; after from the Phase 6 report) ----
    before = json.loads((ROOT / "artifacts/leakage/before.json").read_text())
    top = json.loads((ROOT / "artifacts/leakage/top15_before.json").read_text())
    la = pd.read_csv(p6 / "report" / "leakage_after.csv").set_index("arm")
    write_json(OUT / "json" / "proxy_audit.json", {
        "before_auc_features": before["gender_from_features_auc"],
        "after_auc_z": {"A2": float(la.loc["A2", "probe_lgbm_test_auc"]), "JT": float(la.loc["JT chosen", "probe_lgbm_test_auc"])},
        "shap_probe_auc": {"A2": float(la.loc["A2", "shap_probe_test_auc_seed0"]), "JT": float(la.loc["JT chosen", "shap_probe_test_auc_seed0"])},
        "top_proxy_features": [{"feature": t["feature"], "gain": t["gain_share"]} for t in top],
        "note": "Gender stays recoverable from the learned representation z for both models (AUC well above 0.5)."})  # fmt: skip
    # ---- trade-off sweep (VALIDATION, mean of seeds) ----
    pts = []
    for sub in ("jt", "jt_fine"):
        for d in sorted((p4 / sub).glob("f*_c*_a*")):
            sm = [json.loads(f.read_text()) for f in sorted(d.glob("seed_*/summary.json"))]
            if not sm:
                continue
            lam = sm[0]["lambdas"]
            row = {"lambda_f": lam["f"], "lambda_c": lam["c"], "lambda_a": lam["a"], "n_seeds": len(sm)}
            for k in ("auc_roc", "auc_pr", "dir", "eod", "worst_group_ece"):
                row[k] = float(np.mean([s[f"val_{k}"] for s in sm]))
            pts.append(row)
    df = pd.DataFrame(pts).drop_duplicates(["lambda_f", "lambda_c", "lambda_a"], keep="last")
    front = []
    for i, r in df.iterrows():  # non-dominated in (auc_pr up, eod down)
        if not ((df.auc_pr >= r.auc_pr) & (df.eod <= r.eod) & ((df.auc_pr > r.auc_pr) | (df.eod < r.eod))).any():
            front.append({k: float(v) for k, v in r.items()})
    sel = df[(df.lambda_f == LAMBDAS[0]) & (df.lambda_c == LAMBDAS[1]) & (df.lambda_a == LAMBDAS[2])].iloc[0]
    write_json(OUT / "json" / "tradeoff.json", {"points": [{k: float(v) for k, v in r.items()} for _, r in df.iterrows()],
                                                 "frontier": front, "selected": {k: float(v) for k, v in sel.items()},
                                                 "source": "validation set, mean over seeds; rule-selected point (0.05, 2, 0)"})  # fmt: skip
    # ---- honest comparison table (Phase 6 test table) ----
    tt = pd.read_csv(p6 / "report" / "test_table.csv")
    keep = ["arm", "deployable", "n_seeds", "auc_roc", "auc_roc_lo", "auc_roc_hi", "auc_pr", "auc_pr_lo", "auc_pr_hi", "worst_group_ece", "worst_group_ece_lo", "worst_group_ece_hi", "eod", "eod_lo", "eod_hi", "dir", "recall", "test_flag_rate"]  # fmt: skip
    write_json(OUT / "json" / "comparison.json", {
        "rows": tt[keep].to_dict(orient="records"),
        "headline": "On this dataset the sequential baseline (class weights + prior correction + gender-blind fairness step) matched or beat the joint model.",
        "dataset": "home_credit_test, mean over seeds, 95% paired bootstrap CI", "served_arm": "JT chosen"})  # fmt: skip
    # ---- examples (8 from the test pool, no gender field, no identifiers) ----
    pre = Preprocessor(data_cfg)
    pre.groups = {k: order[k] for k in ("numeric", "binary", "categorical")}
    pre.scaler = joblib.load(prep_dir / "scaler.joblib")
    pre.encoders = json.loads((prep_dir / "encoders.json").read_text())
    rng = np.random.default_rng(7)
    exs, tried = [], 0
    for j in rng.permutation(len(pc)):
        tried += 1
        rec = inverse_example(pre, pc[j], pk[j], spec)
        if rec is not None:
            exs.append(rec)
        if len(exs) == 8 or tried > 5000:
            break
    write_json(OUT / "examples.json", {"examples": exs, "source": "held-out test rows (seed 7), identifiers and gender removed"})  # fmt: skip
    # ---- reproducibility files ----
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT / "requirements_freeze.txt").write_text(freeze, encoding="utf-8")
    cfgs = "".join(sha256_file(p) for p in sorted((ROOT / "configs").glob("*.yaml")))
    files = {p.relative_to(OUT).as_posix(): sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    rb = {"low_max": float(json.loads((prep_dir / "prep_report.json").read_text())["splits"]["train"]["default_rate"]), "medium_max": float(thr)}  # fmt: skip
    manifest = {"model_version": f"jt-chosen-0.05-2-0-seed{seed}-{VERSION}", "git_commit": git_hash(),
                "config_hash": hashlib.sha256(cfgs.encode()).hexdigest(), "seeds": SEEDS, "served_seed": seed,
                "lambdas": list(LAMBDAS), "risk_bands": rb,
                "risk_band_rule": "Low: below the training default rate; Medium: above it but below the flag threshold (validation 90th percentile); High: at or above the flag threshold",  # fmt: skip
                "parity_bf16_vs_fp32_max_abs_diff_on_test_pool": gap_bf16, "files": files, "bundle_sha": bundle_sha(files)}  # fmt: skip
    write_json(OUT / "manifest.json", manifest)
    load_bundle(OUT)  # verify round trip
    print(json.dumps({k: manifest[k] for k in ("model_version", "bundle_sha", "risk_bands", "parity_bf16_vs_fp32_max_abs_diff_on_test_pool")}, indent=1))  # fmt: skip
    print("examples:", len(exs), "seed", seed, "matched mean|delta|", ds["mean"])


if __name__ == "__main__":
    main()
