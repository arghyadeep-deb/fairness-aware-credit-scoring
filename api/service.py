"""Preprocess with the SAME transform as training (src.data.prep), predict, risk band."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from api.bundle import Bundle
from api.schemas import ApplicantRecord

DAYS_PER_YEAR = 365.25
PLACEHOLDER = 365243

# ApplicantRecord field -> raw Home Credit column
CAT_MAP = {
    "name_family_status": "NAME_FAMILY_STATUS",
    "name_education_type": "NAME_EDUCATION_TYPE",
    "name_income_type": "NAME_INCOME_TYPE",
    "name_housing_type": "NAME_HOUSING_TYPE",
    "occupation_type": "OCCUPATION_TYPE",
    "organization_type": "ORGANIZATION_TYPE",
}


def _as_dict(rec: ApplicantRecord | dict) -> dict:
    return rec.model_dump() if isinstance(rec, ApplicantRecord) else dict(rec)


def record_to_raw(rec: ApplicantRecord | dict) -> pd.DataFrame:
    """One-row frame with the raw application_train columns the Preprocessor derives from."""
    r = _as_dict(rec)
    emp = r.get("years_employed")
    row = {
        "CNT_CHILDREN": r["cnt_children"],
        "CNT_FAM_MEMBERS": r["cnt_fam_members"],
        "AMT_INCOME_TOTAL": r["amt_income_total"],
        "AMT_CREDIT": r["amt_credit"],
        "AMT_ANNUITY": r["amt_annuity"],
        "AMT_GOODS_PRICE": r["amt_goods_price"],
        "REGION_POPULATION_RELATIVE": r["region_population_relative"],
        "DAYS_BIRTH": -float(r["age_years"]) * DAYS_PER_YEAR,
        "DAYS_EMPLOYED": PLACEHOLDER if emp is None else -float(emp) * DAYS_PER_YEAR,
        "FLAG_OWN_CAR": r["flag_own_car"],
        "FLAG_OWN_REALTY": r["flag_own_realty"],
        "EXT_SOURCE_1": r.get("ext_source_1"),
        "EXT_SOURCE_2": r.get("ext_source_2"),
        "EXT_SOURCE_3": r.get("ext_source_3"),
    }
    for k, col in CAT_MAP.items():
        row[col] = r.get(k)
    df = pd.DataFrame([row])
    for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        df[c] = df[c].astype("float64")
    return df


def unseen_categories(b: Bundle, rec: ApplicantRecord | dict) -> list[str]:
    r = _as_dict(rec)
    out = []
    for k, col in CAT_MAP.items():
        v = r.get(k)
        if v is not None and v not in b.pre.encoders[col]:
            out.append(f"unseen category for {k}: treated as Missing")
    return out


def to_arrays(b: Bundle, rec: ApplicantRecord | dict) -> tuple[np.ndarray, np.ndarray]:
    x = b.pre.transform(record_to_raw(rec))
    cont = x[b.spec["cont_names"]].to_numpy(dtype="float32")
    cat = x[b.spec["cat_names"]].to_numpy(dtype="int64")
    return cont, cat


@torch.no_grad()
def predict_arrays(b: Bundle, cont: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """fp32 CPU probability = sigmoid(logit); Head B is never evaluated (served path = Head A)."""
    xc = torch.from_numpy(np.ascontiguousarray(cont, dtype="float32"))
    xk = torch.from_numpy(np.ascontiguousarray(cat, dtype="int64"))
    logit, _ = b.model(xc, xk)
    return torch.sigmoid(logit.float()).numpy().astype("float64")


def risk_band(b: Bundle, p: float) -> str:
    rb = b.manifest["risk_bands"]
    return "Low" if p < rb["low_max"] else ("Medium" if p < rb["medium_max"] else "High")


def predict(b: Bundle, rec: ApplicantRecord) -> dict:
    cont, cat = to_arrays(b, rec)
    p = float(predict_arrays(b, cont, cat)[0])
    return {
        "probability": p,
        "risk_band": risk_band(b, p),
        "calibrated": True,
        "model_version": b.version,
        "warnings": unseen_categories(b, rec),
    }
