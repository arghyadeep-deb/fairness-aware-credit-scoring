"""Stage 10 acceptance tests for the API (specs/03, specs/06). Needs the frozen bundle
(python scripts/freeze_bundle.py ...). Uses only saved arrays for test rows (no test loader)."""

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.bundle import DEFAULT_DIR
from api.main import create_app

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (DEFAULT_DIR / "manifest.json").exists(), reason="bundle not frozen yet"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(DEFAULT_DIR)) as c:
        yield c


@pytest.fixture(scope="module")
def example(client):
    return client.get("/examples").json()["examples"][0]


def test_health_and_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert (
        j["status"] == "ok"
        and len(j["bundle_sha"]) == 64
        and j["model_version"].startswith("jt-chosen")
    )


def test_predict_schema_and_determinism(client, example):
    a, b = client.post("/predict", json=example), client.post("/predict", json=example)
    assert a.status_code == 200
    j = a.json()
    assert set(j) >= {"probability", "risk_band", "calibrated", "model_version"}
    assert (
        0 < j["probability"] < 1
        and j["risk_band"] in ("Low", "Medium", "High")
        and j["calibrated"] is True
    )
    assert a.json() == b.json()


def test_explain_schema_sorted_and_deterministic(client, example):
    a, b = client.post("/explain", json=example), client.post("/explain", json=example)
    assert a.status_code == 200
    j = a.json()
    assert set(j) >= {"base_value", "contributions", "top_k", "model_version"} and j["top_k"] == 10
    assert len(j["contributions"]) == 10
    mags = [abs(c["shap"]) for c in j["contributions"]]
    assert mags == sorted(mags, reverse=True)
    assert all(set(c) == {"feature", "value", "shap"} for c in j["contributions"])
    assert a.json() == b.json()


def test_matched_pair_schema(client, example):
    r = client.post("/matched-pair", json={"record": example, "gender": "F"})
    assert r.status_code == 200
    j = r.json()
    assert set(j) >= {
        "own_probability",
        "matched_probability",
        "delta",
        "matched_features",
        "population_mean_abs_delta",
    }
    assert abs(j["delta"] - (j["matched_probability"] - j["own_probability"])) < 1e-9
    assert j["population_mean_abs_delta"] >= 0


def test_matched_pair_invalid_gender_is_400(client, example):
    r = client.post("/matched-pair", json={"record": example, "gender": "X"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "BAD_REQUEST"


def test_dashboard_endpoints_schemas(client):
    f = client.get("/fairness").json()
    assert set(f) >= {
        "threshold_rule",
        "threshold",
        "dpd",
        "eod",
        "dir",
        "per_group",
        "ci95",
        "dataset",
    }
    assert set(f["per_group"]) == {"M", "F"} and set(f["per_group"]["M"]) == {"tpr", "fpr"}
    c = client.get("/calibration").json()
    assert set(c["groups"]) == {"M", "F"} and {"bins", "ece"} <= set(c["groups"]["M"])
    assert (
        {"mean_pred", "frac_pos", "count"} <= set(c["groups"]["M"]["bins"][0])
        and "brier" in c
        and "worst_group_ece" in c
    )
    p = client.get("/proxy-audit").json()
    assert {"before_auc_features", "after_auc_z", "shap_probe_auc", "top_proxy_features"} <= set(p)
    assert {"A2", "JT"} <= set(p["after_auc_z"])
    cmp_ = client.get("/comparison").json()
    arms = {r["arm"]: r for r in cmp_["rows"]}
    assert (
        "S1" in arms
        and "S2-EG" in arms
        and "JT chosen" in arms
        and "sequential baseline" in cmp_["headline"]
    )
    assert all(isinstance(r["deployable"], bool) for r in arms.values())


def test_tradeoff_nearest_point(client):
    j = client.get("/tradeoff", params={"lambda_f": 0.05, "lambda_c": 2, "lambda_a": 0}).json()
    assert set(j) >= {"requested", "nearest", "metrics", "frontier"}
    assert j["nearest"] == {"lambda_f": 0.05, "lambda_c": 2.0, "lambda_a": 0.0}
    assert set(j["metrics"]) == {"auc_roc", "auc_pr", "dir", "eod", "worst_group_ece"}
    far = client.get("/tradeoff", params={"lambda_f": 0.5, "lambda_c": 0, "lambda_a": 0}).json()
    assert far["nearest"]["lambda_f"] == 0.5


def test_meta_options_built_from_bundle(client):
    m = client.get("/meta/options").json()
    enc = json.loads((DEFAULT_DIR / "prep/encoders.json").read_text())
    assert m["categorical"]["name_family_status"] == sorted(enc["NAME_FAMILY_STATUS"])
    assert m["numeric"]["age_years"]["min"] == 18 and m["numeric"]["age_years"]["max"] == 100
    assert set(m["risk_bands"]) == {"low_max", "medium_max"} and m["include_ext_source"] is True


def test_examples_have_no_gender_and_count(client):
    ex = client.get("/examples").json()["examples"]
    assert 5 <= len(ex) <= 10
    assert all(not any("gender" in k.lower() or "sex" in k.lower() for k in e) for e in ex)


@pytest.mark.parametrize(
    "field,value",
    [
        ("age_years", 5),
        ("age_years", 150),
        ("amt_income_total", 0),
        ("cnt_children", -1),
        ("region_population_relative", 2),
        ("ext_source_1", 1.5),
        ("flag_own_car", "maybe"),
    ],
)
def test_422_on_invalid_values(client, example, field, value):
    r = client.post("/predict", json={**example, field: value})
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "VALIDATION_ERROR" and any(d["field"].endswith(field) for d in e["details"])


def test_422_missing_and_gender_field_forbidden(client, example):
    miss = {k: v for k, v in example.items() if k != "amt_credit"}
    assert client.post("/predict", json=miss).status_code == 422
    assert (
        client.post("/predict", json={**example, "gender": "F"}).status_code == 422
    )  # gender is not an input


def test_malformed_json_400_and_unknown_route_404(client):
    r = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "BAD_REQUEST"
    n = client.get("/no-such-route")
    assert n.status_code == 404 and n.json()["error"]["code"] == "NOT_FOUND"


def test_unseen_category_warns_and_maps_to_missing(client, example):
    odd = {**example, "occupation_type": "Interplanetary pilot"}
    miss = {**example, "occupation_type": None}
    a, b = client.post("/predict", json=odd), client.post("/predict", json=miss)
    assert a.status_code == 200 and any("occupation_type" in w for w in a.json()["warnings"])
    assert (
        abs(a.json()["probability"] - b.json()["probability"]) < 1e-12
        and b.json()["warnings"] == []
    )


def test_503_on_tampered_and_missing_bundle(tmp_path):
    bad = tmp_path / "bundle"
    shutil.copytree(DEFAULT_DIR, bad)
    with (bad / "examples.json").open("ab") as f:
        f.write(b" ")  # one byte changed -> hash mismatch
    with TestClient(create_app(bad)) as c:
        for r in (
            c.get("/health"),
            c.post("/predict", json={"x": 1}),
            c.get("/fairness"),
            c.get("/examples"),
        ):
            assert r.status_code in (503, 422)
        r = c.get("/health")
        assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_READY"
        assert c.get("/fairness").status_code == 503
    with TestClient(create_app(tmp_path / "nowhere")) as c:
        assert c.get("/health").status_code == 503


def test_503_on_unlisted_extra_file(tmp_path):
    bad = tmp_path / "bundle2"
    shutil.copytree(DEFAULT_DIR, bad)
    (bad / "sneaky.txt").write_text("x")
    with TestClient(create_app(bad)) as c:
        assert c.get("/health").status_code == 503


def test_parity_100_saved_test_rows_api_array_path_vs_fp32_reference(client):
    """API model path vs an independent fp32 CPU reference built from the full training checkpoint."""
    import torch

    from api.service import predict_arrays
    from src.eval.final import load_joint
    from src.utils import load_config

    b = client.app.state.bundle
    cfg = load_config("configs/train.yaml")
    tz = np.load(ROOT / "artifacts/test_preds/test_arrays.npz")
    ix = np.sort(np.random.default_rng(0).choice(len(tz["y"]), 100, replace=False))
    cont, cat = tz["cont"][ix].astype("float32"), tz["cat"][ix].astype("int64")
    api_p = predict_arrays(b, cont, cat)
    p4 = Path((ROOT / "artifacts/phase4_rundir.txt").read_text().strip())
    ck = torch.load(
        p4 / "jt/f0p05_c2_a0" / f"seed_{b.spec['seed']}" / "best.pt", map_location="cpu"
    )
    from src.data.tensors import Arrays

    like = Arrays(
        cont, cat, tz["y"][ix], tz["a"][ix], b.spec["cont_names"], b.spec["cat_names"], []
    )
    m = load_joint(like, b.spec["cards"], cfg, b.spec["hp"], ck, torch.device("cpu"))
    with torch.no_grad():
        ref = (
            torch.sigmoid(m.predict(torch.from_numpy(cont), torch.from_numpy(cat)).float())
            .numpy()
            .astype("float64")
        )
    assert float(np.abs(api_p - ref).max()) < 1e-5
    # Phase 6 test predictions were scored with bf16 autocast on GPU: record the (expected, larger) gap
    saved = np.load(ROOT / "artifacts/test_preds/JT_chosen" / f"seed_{b.spec['seed']}" / "pt.npy")[
        ix
    ]
    assert float(np.abs(api_p - saved).max()) < 5e-3


def test_parity_raw_record_path_vs_offline_pipeline_on_validation_rows(client):
    """Raw ApplicantRecord -> API preprocessing -> probability vs the offline processed val features."""
    import torch

    from api.service import predict_arrays, to_arrays

    b = client.app.state.bundle
    ids = pd.read_parquet(ROOT / "data/processed/val_gender.parquet")["SK_ID_CURR"].to_numpy()[:400]
    feats = pd.read_parquet(ROOT / "data/processed/val.parquet").iloc[:400]
    cols = [
        "SK_ID_CURR",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "REGION_POPULATION_RELATIVE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "NAME_FAMILY_STATUS",
        "NAME_EDUCATION_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]
    raw = (
        pd.read_csv(ROOT / "data/raw/application_train.csv", usecols=cols)
        .set_index("SK_ID_CURR")
        .loc[ids]
    )
    raw = raw.reset_index()
    need = ["AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS"]
    keep = np.flatnonzero(raw[need].notna().all(axis=1).to_numpy())[:100]
    maxd = 0.0
    off_cont = feats[b.spec["cont_names"]].to_numpy("float32")
    off_cat = feats[b.spec["cat_names"]].to_numpy("int64")
    for i in keep:
        r = raw.iloc[i]
        emp = r["DAYS_EMPLOYED"]
        rec = {
            "age_years": -r["DAYS_BIRTH"] / 365.25,
            "cnt_children": int(r["CNT_CHILDREN"]),
            "cnt_fam_members": int(r["CNT_FAM_MEMBERS"]),
            "amt_income_total": float(r["AMT_INCOME_TOTAL"]),
            "amt_credit": float(r["AMT_CREDIT"]),
            "amt_annuity": float(r["AMT_ANNUITY"]),
            "amt_goods_price": float(r["AMT_GOODS_PRICE"]),
            "region_population_relative": float(r["REGION_POPULATION_RELATIVE"]),
            "years_employed": None if emp == 365243 else float(-emp / 365.25),
            "flag_own_car": r["FLAG_OWN_CAR"],
            "flag_own_realty": r["FLAG_OWN_REALTY"],
        }
        for k, c in (
            ("name_family_status", "NAME_FAMILY_STATUS"),
            ("name_education_type", "NAME_EDUCATION_TYPE"),
            ("name_income_type", "NAME_INCOME_TYPE"),
            ("name_housing_type", "NAME_HOUSING_TYPE"),
            ("occupation_type", "OCCUPATION_TYPE"),
            ("organization_type", "ORGANIZATION_TYPE"),
        ):
            rec[k] = None if pd.isna(r[c]) else r[c]
        for k in (1, 2, 3):
            v = r[f"EXT_SOURCE_{k}"]
            rec[f"ext_source_{k}"] = None if pd.isna(v) else float(v)
        cont, cat = to_arrays(b, rec)
        assert np.abs(cont[0] - off_cont[i]).max() < 1e-4 and (cat[0] == off_cat[i]).all()
        p_api = predict_arrays(b, cont, cat)[0]
        with torch.no_grad():
            p_off = float(
                torch.sigmoid(
                    b.model(
                        torch.from_numpy(off_cont[i : i + 1]), torch.from_numpy(off_cat[i : i + 1])
                    )[0]
                )
            )
        maxd = max(maxd, abs(p_api - p_off))
    assert len(keep) >= 90 and maxd < 1e-5


def test_latency_predict_p95_and_explain_budget(client, example):
    client.post("/predict", json=example)
    client.post("/explain", json=example)  # warm-up (imports, first allocations)
    lat = []
    for _ in range(40):
        t = time.perf_counter()
        assert client.post("/predict", json=example).status_code == 200
        lat.append(time.perf_counter() - t)
    assert np.percentile(lat, 95) < 0.3
    t = time.perf_counter()
    client.post("/predict", json=example)
    assert client.post("/explain", json=example).status_code == 200
    assert time.perf_counter() - t < 3.0


def test_no_applicant_data_written_or_logged(client, caplog):
    marker = {"amt_income_total": 987654.321, "amt_credit": 123456.789}
    rec = None

    def snap():
        out = {}
        for base in (
            ROOT / "api",
            ROOT / "web",
            ROOT / "artifacts" / "bundle",
            ROOT / "data" / "processed",
            Path.cwd(),
        ):
            for p in base.rglob("*") if base != Path.cwd() else base.glob("*"):
                if (
                    p.is_file()
                    and ".git" not in p.parts
                    and "__pycache__" not in p.parts
                    and ".pytest_cache" not in p.parts
                ):
                    out[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns)
        return out

    rec = {**client.get("/examples").json()["examples"][1], **marker}
    before = snap()
    with caplog.at_level(logging.DEBUG):
        for ep in ("/predict", "/explain"):
            client.post(ep, json=rec)
        client.post("/matched-pair", json={"record": rec, "gender": "M"})
        client.post("/predict", json={**rec, "age_years": 3})  # validation error path
    after = snap()
    new = {k for k in after if k not in before or after[k] != before[k]}
    assert not new, f"files written during requests: {sorted(new)[:5]}"
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "987654" not in text and "123456.789" not in text and "Interplanetary" not in text
    assert any(
        "/predict" in r.getMessage() for r in caplog.records
    )  # access log exists (id, path, status, latency)


def test_no_database_dependency_in_api():
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "api").glob("*.py"))
    for banned in ("sqlalchemy", "sqlite3", "pymysql", "psycopg", "mysql", "pymongo", "peewee"):
        assert banned not in src.lower()


def test_explain_display_merges_missing_flags_and_keeps_contributions(client, example):
    rec = dict(example, ext_source_3=None)
    j = client.post("/explain", json=rec).json()
    assert len(j["display"]) == j["top_k"] and all(not c["feature"].endswith("_MISSING") for c in j["display"])
    assert all(set(c) == {"feature", "value", "shap"} for c in j["contributions"])  # contract unchanged
    mags = [abs(c["shap"]) for c in j["display"]]
    assert mags == sorted(mags, reverse=True)


def test_merge_missing_is_additive_and_marks_missing():
    from api.explain import merge_missing

    items = [
        {"feature": "EXT_SOURCE_3", "value": 0.5, "shap": 0.2},
        {"feature": "EXT_SOURCE_3_MISSING", "value": 1, "shap": -0.5},
        {"feature": "EXT_SOURCE_1", "value": 0.4, "shap": 0.1},
        {"feature": "EXT_SOURCE_1_MISSING", "value": 0, "shap": 0.05},
        {"feature": "AMT_CREDIT", "value": 9, "shap": 0.01},
    ]
    out = {t["feature"]: t for t in merge_missing(items)}
    assert set(out) == {"EXT_SOURCE_3", "EXT_SOURCE_1", "AMT_CREDIT"}
    assert out["EXT_SOURCE_3"]["missing"] and out["EXT_SOURCE_3"]["value"] is None
    assert abs(out["EXT_SOURCE_3"]["shap"] + 0.3) < 1e-9
    assert not out["EXT_SOURCE_1"]["missing"] and out["EXT_SOURCE_1"]["value"] == 0.4
    assert abs(sum(t["shap"] for t in out.values()) - sum(t["shap"] for t in items)) < 1e-9


def test_posthoc_rows_stay_outside_the_frozen_comparison(client):
    assert "REF-EG" not in [r["arm"] for r in client.get("/comparison").json()["rows"]]
    import json as _json
    from pathlib import Path as _P

    data = _json.loads((_P(__file__).resolve().parents[1] / "web/data/posthoc_arms.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["arm"] == "REF-EG" and data["rows"][0]["posthoc"] is True


def test_nearest_other_gender_leaves_out_the_applicants_own_row():
    from api.matched import nearest_other_gender

    rng = np.random.default_rng(0)
    cont, cat = rng.normal(size=(200, 4)).astype("float32"), np.zeros((200, 2), dtype="int64")
    a = np.zeros(200, dtype="int64")
    q = cont[7] + 0.001  # a rounded copy of pool row 7
    j, excluded = nearest_other_gender(cont, cat, a, q, cat[7], own_gender_code=1)
    assert excluded and j != 7
    far = cont[7] + 5.0  # no pool row is this applicant
    assert nearest_other_gender(cont, cat, a, far, cat[7], own_gender_code=1)[1] is False


def test_sample_applicants_never_match_themselves(client):
    for rec in client.get("/examples").json()["examples"]:
        flags = []
        for g in ("M", "F"):
            j = client.post("/matched-pair", json={"record": rec, "gender": g}).json()
            assert abs(j["delta"]) > 1e-3
            assert isinstance(j["self_excluded"], bool)
            flags.append(j["self_excluded"])
        assert flags[0] == flags[1], "flag must not depend on the submitted gender"
