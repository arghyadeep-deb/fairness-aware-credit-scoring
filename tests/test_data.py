"""Stage 1 tests: unit tests on synthetic rows + integration checks on the real processed data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.checks import SPLITS, run_checks
from src.data.io import load_split
from src.data.prep import MISSING_LABEL, Preprocessor, class_weights, derive, split_indices
from src.utils import load_config, resolve

CFG = load_config("configs/data.yaml")


def make_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Small Home-Credit-shaped frame covering every column the Preprocessor reads."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "TARGET": r.binomial(1, 0.1, n),
            "CODE_GENDER": r.choice(["F", "M"], n),
            "CNT_CHILDREN": r.integers(0, 4, n),
            "CNT_FAM_MEMBERS": r.integers(1, 6, n).astype(float),
            "AMT_INCOME_TOTAL": r.lognormal(12, 0.5, n),
            "AMT_CREDIT": r.lognormal(13, 0.5, n),
            "AMT_ANNUITY": r.lognormal(10, 0.5, n),
            "AMT_GOODS_PRICE": r.lognormal(13, 0.5, n),
            "REGION_POPULATION_RELATIVE": r.uniform(0, 0.07, n),
            "DAYS_BIRTH": -r.integers(8000, 24000, n),
            "DAYS_EMPLOYED": -r.integers(0, 10000, n),
            "FLAG_OWN_CAR": r.choice(["Y", "N"], n),
            "FLAG_OWN_REALTY": r.choice(["Y", "N"], n),
            "EXT_SOURCE_1": r.uniform(0, 1, n),
            "EXT_SOURCE_2": r.uniform(0, 1, n),
            "EXT_SOURCE_3": r.uniform(0, 1, n),
            "NAME_FAMILY_STATUS": r.choice(["Married", "Single"], n),
            "NAME_EDUCATION_TYPE": r.choice(["Higher", "Secondary"], n),
            "NAME_INCOME_TYPE": r.choice(["Working", "Pensioner"], n),
            "NAME_HOUSING_TYPE": r.choice(["House", "Rented"], n),
            "OCCUPATION_TYPE": r.choice(["Laborers", "Core staff", None], n),
            "ORGANIZATION_TYPE": r.choice(["Business", "School", "XNA"], n),
        }
    )
    df.loc[:9, "DAYS_EMPLOYED"] = 365243
    df.loc[10:19, "EXT_SOURCE_1"] = np.nan
    return df


# ---------- unit tests (synthetic) ----------
def test_placeholder_becomes_nan_with_flag():
    raw = make_raw()
    d = derive(raw, CFG)
    assert (d.loc[:9, "EMPLOYED_MISSING"] == 1).all() and (
        d.loc[10:, "EMPLOYED_MISSING"] == 0
    ).all()
    assert d.loc[:9, "YEARS_EMPLOYED"].isna().all()
    assert d.loc[10:, "YEARS_EMPLOYED"].notna().all()
    assert (d["YEARS_EMPLOYED"].dropna() >= 0).all()  # placeholder would give -1000 years


def test_age_and_raw_day_columns_replaced():
    raw = make_raw()
    d = derive(raw, CFG)
    np.testing.assert_allclose(d["AGE_YEARS"], -raw["DAYS_BIRTH"] / 365.25)
    assert "DAYS_BIRTH" not in d.columns and "DAYS_EMPLOYED" not in d.columns


def test_gender_never_in_features_and_no_id_or_target():
    pre = Preprocessor(CFG).fit(make_raw())
    out = pre.transform(make_raw(seed=1))
    banned = {"CODE_GENDER", "SK_ID_CURR", "TARGET"}
    assert not banned & set(out.columns) and not banned & set(pre.feature_order)


def test_ratio_guard_zero_denominator():
    raw = make_raw()
    raw.loc[0, "AMT_INCOME_TOTAL"] = 0.0
    raw.loc[1, "AMT_ANNUITY"] = 0.0
    d = derive(raw, CFG)
    assert np.isnan(d.loc[0, "CREDIT_INCOME_RATIO"]) and np.isnan(d.loc[1, "CREDIT_TERM"])
    assert (
        not np.isinf(d[["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM"]]).any().any()
    )
    out = Preprocessor(CFG).fit(raw).transform(raw)
    assert not out.isna().any().any()


def test_ext_source_missing_indicator_and_median_fill():
    raw = make_raw()
    pre = Preprocessor(CFG).fit(raw)
    out = pre.transform(raw)
    assert (out.loc[10:19, "EXT_SOURCE_1_MISSING"] == 1).all()
    assert out["EXT_SOURCE_1_MISSING"].sum() == 10
    assert not out.isna().any().any()


def test_categorical_zero_is_missing_and_unseen():
    raw = make_raw()
    pre = Preprocessor(CFG).fit(raw)
    assert (pre.transform(raw).loc[raw["OCCUPATION_TYPE"].isna(), "OCCUPATION_TYPE"] == 0).all()
    new = make_raw(seed=2)
    new.loc[0, "NAME_FAMILY_STATUS"] = "Never-seen-category"
    out = pre.transform(new)
    assert out.loc[0, "NAME_FAMILY_STATUS"] == 0
    assert all(min(out[c]) >= 0 and max(out[c]) < pre.cardinalities[c] for c in CFG["categorical"])
    assert all(v >= 1 for m in pre.encoders.values() for v in m.values())
    assert MISSING_LABEL not in pre.encoders["OCCUPATION_TYPE"]


def test_fit_uses_train_only():
    train, other = make_raw(seed=0), make_raw(seed=5)
    other["AMT_INCOME_TOTAL"] *= 1000  # wildly different distribution
    pre = Preprocessor(CFG).fit(train)
    ref = pre.scaler.mean_.copy()
    pre.transform(other)
    np.testing.assert_array_equal(pre.scaler.mean_, ref)  # transform never refits
    refit = Preprocessor(CFG).fit(pd.concat([train, other]))
    assert not np.allclose(refit.scaler.mean_, ref)  # test would catch leakage-by-refit
    assert abs(pre.transform(train)["AMT_INCOME_TOTAL"].mean()) < 1e-3


def test_yn_rejects_garbage():
    raw = make_raw()
    raw.loc[0, "FLAG_OWN_CAR"] = "maybe"
    with pytest.raises(ValueError):
        derive(raw, CFG)


def test_split_is_stratified_and_reproducible():
    raw = make_raw(n=4000)
    y = raw["TARGET"].to_numpy()
    a = (raw["CODE_GENDER"] == "M").astype(int).to_numpy()
    s1 = split_indices(y, a, CFG["split"]["fractions"], 42)
    s2 = split_indices(y, a, CFG["split"]["fractions"], 42)
    assert all((s1[k] == s2[k]).all() for k in s1)
    assert len(s1["train"]) == 2800 and len(s1["val"]) == 600 and len(s1["test"]) == 600
    allidx = np.concatenate(list(s1.values()))
    assert len(set(allidx)) == len(allidx) == 4000
    for k in ("val", "test"):
        for t in (0, 1):
            for g in (0, 1):
                full = ((y == t) & (a == g)).mean()
                part = ((y[s1[k]] == t) & (a[s1[k]] == g)).mean()
                assert abs(full - part) < 0.005


def test_class_weights_formula():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y)
    assert w["0"] == pytest.approx(100 / 180) and w["1"] == pytest.approx(5.0)


# ---------- integration (real processed data; fail loudly if prep has not been run) ----------
@pytest.fixture(scope="module")
def prepared():
    proc, prep = resolve(CFG["paths"]["processed"]), resolve(CFG["paths"]["prep_artifacts"])
    assert (proc / "train.parquet").exists(), "run: python -m src.data.prep"
    return proc, prep


def test_real_data_all_stage1_checks_pass(prepared):
    proc, prep = prepared
    results = run_checks(proc, prep, CFG)
    failed = [(n, d) for n, ok, d in results if not ok]
    assert not failed, failed


def test_real_feature_files_have_no_gender_and_expected_artifacts(prepared):
    proc, prep = prepared
    for name in (
        "scaler.joblib",
        "encoders.json",
        "cardinalities.json",
        "feature_order.json",
        "class_weights.json",
        "medians.json",
    ):
        assert (prep / name).exists(), name
    order = json.loads((prep / "feature_order.json").read_text())["order"]
    assert not any("GENDER" in c.upper() for c in order)
    for s in SPLITS:
        cols = pd.read_parquet(proc / f"{s}.parquet").columns
        assert not any("GENDER" in c.upper() for c in cols) and "SK_ID_CURR" not in cols
        assert list(cols) == order + ["TARGET"]


def test_real_gender_aligned_and_no_xna(prepared):
    proc, _ = prepared
    for s in SPLITS:
        g = pd.read_parquet(proc / f"{s}_gender.parquet")
        x = pd.read_parquet(proc / f"{s}.parquet", columns=["TARGET"])
        assert len(g) == len(x) and set(g["gender"].unique()) == {0, 1}
    raw = pd.read_csv(resolve(CFG["paths"]["raw_csv"]), usecols=["SK_ID_CURR", "CODE_GENDER"])
    kept = pd.concat(pd.read_parquet(proc / f"{s}_gender.parquet") for s in SPLITS)
    assert len(kept) == (raw["CODE_GENDER"] != "XNA").sum() == 307507


def test_real_gender_alignment_matches_raw(prepared):
    proc, _ = prepared
    raw = pd.read_csv(
        resolve(CFG["paths"]["raw_csv"]), usecols=["SK_ID_CURR", "CODE_GENDER", "TARGET"]
    )
    raw = raw.set_index("SK_ID_CURR")
    g = pd.read_parquet(proc / "val_gender.parquet")
    x = pd.read_parquet(proc / "val.parquet", columns=["TARGET"])
    expect_gender = raw.loc[g["SK_ID_CURR"], "CODE_GENDER"].map({"F": 0, "M": 1}).to_numpy()
    assert (expect_gender == g["gender"].to_numpy()).all()
    assert (raw.loc[g["SK_ID_CURR"], "TARGET"].to_numpy() == x["TARGET"].to_numpy()).all()


def test_test_split_is_locked_without_permission(prepared, monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(PermissionError):
        load_split(CFG, "test", with_target=True)
    with pytest.raises(PermissionError):
        load_split(CFG, "test", with_target=False)  # no declared purpose
