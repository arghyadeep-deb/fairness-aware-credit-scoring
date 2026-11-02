"""Pydantic models for the API (specs/03). Gender is NOT part of ApplicantRecord."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApplicantRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_years: float = Field(ge=18, le=100)
    cnt_children: int = Field(ge=0, le=20)
    cnt_fam_members: int = Field(ge=1, le=25)
    amt_income_total: float = Field(gt=0)
    amt_credit: float = Field(gt=0)
    amt_annuity: float = Field(gt=0)
    amt_goods_price: float = Field(gt=0)
    region_population_relative: float = Field(ge=0, le=1)
    years_employed: float | None = Field(default=None, ge=0, le=60)
    name_family_status: str = Field(max_length=64)
    name_education_type: str = Field(max_length=64)
    name_income_type: str = Field(max_length=64)
    name_housing_type: str = Field(max_length=64)
    occupation_type: str | None = Field(default=None, max_length=64)
    organization_type: str | None = Field(default=None, max_length=64)
    flag_own_car: Literal["Y", "N"]
    flag_own_realty: Literal["Y", "N"]
    ext_source_1: float | None = Field(default=None, ge=0, le=1)
    ext_source_2: float | None = Field(default=None, ge=0, le=1)
    ext_source_3: float | None = Field(default=None, ge=0, le=1)


class PredictResponse(BaseModel):
    probability: float
    risk_band: Literal["Low", "Medium", "High"]
    calibrated: bool = True
    model_version: str
    warnings: list[str] = []


class Contribution(BaseModel):
    feature: str
    value: Any
    shap: float


class DisplayContribution(Contribution):
    missing: bool = False


class ExplainResponse(BaseModel):
    base_value: float
    contributions: list[Contribution]
    display: list[DisplayContribution] = []
    top_k: int
    model_version: str
    warnings: list[str] = []


class MatchedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record: ApplicantRecord
    gender: Literal["M", "F"]


class MatchedResponse(BaseModel):
    own_probability: float
    matched_probability: float
    delta: float
    self_excluded: bool = False
    matched_features: dict[str, Any]
    population_mean_abs_delta: float
    model_version: str


class ErrorDetail(BaseModel):
    field: str
    issue: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = []


class ErrorResponse(BaseModel):
    error: ErrorBody
