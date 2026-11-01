# Spec 03: API Contracts (FastAPI, JSON, prefix none, port 8000)

All requests/responses are JSON. Validation by pydantic. CORS enabled for the local frontend origin only.
Every response includes `model_version` (from bundle manifest).

## Shared schema: ApplicantRecord
```
{
  "age_years": float,            // 18..100
  "cnt_children": int,           // 0..20
  "cnt_fam_members": int,        // 1..25
  "amt_income_total": float,     // > 0
  "amt_credit": float,           // > 0
  "amt_annuity": float,          // > 0
  "amt_goods_price": float,      // > 0
  "region_population_relative": float,  // 0..1
  "years_employed": float|null,  // 0..60, null = unemployed/pensioner placeholder
  "name_family_status": str, "name_education_type": str, "name_income_type": str,
  "name_housing_type": str, "occupation_type": str|null, "organization_type": str|null,
  "flag_own_car": "Y"|"N", "flag_own_realty": "Y"|"N",
  "ext_source_1": float|null, "ext_source_2": float|null, "ext_source_3": float|null  // present only if include_ext_source=true
}
```
Gender is NOT part of ApplicantRecord. Unknown category strings map to index 0 (Missing), not an error.

## POST /predict
Req: `ApplicantRecord`
Res 200: `{"probability": float, "risk_band": "Low"|"Medium"|"High", "calibrated": true, "model_version": str}`

## POST /explain
Req: `ApplicantRecord`
Res 200: `{"base_value": float, "contributions": [{"feature": str, "value": any, "shap": float}], "top_k": 10, "model_version": str}`
Sorted by |shap| descending. Must complete within 3 s (NFR-01 covers predict + explain).

## POST /matched-pair
Req: `{"record": ApplicantRecord, "gender": "M"|"F"}`  (gender used ONLY to select the match pool; never fed to the model)
Res 200: `{"own_probability": float, "matched_probability": float, "delta": float, "matched_features": {...}, "population_mean_abs_delta": float, "model_version": str}`
(Replaces the SRS `/counterfactual` gender-flip endpoint: flipping gender is a no-op because the model never sees gender.)

## GET /fairness
Res 200: `{"threshold_rule": str, "threshold": float, "dpd": float, "eod": float, "dir": float, "per_group": {"M": {"tpr": float, "fpr": float}, "F": {...}}, "ci95": {...}, "dataset": "home_credit_test"}`

## GET /calibration
Res 200: `{"groups": {"M": {"bins": [{"mean_pred": float, "frac_pos": float, "count": int}], "ece": float}, "F": {...}}, "brier": float, "worst_group_ece": float}`

## GET /proxy-audit
Res 200: `{"before_auc_features": float, "after_auc_z": {"A2": float, "JT": float}, "shap_probe_auc": float, "top_proxy_features": [{"feature": str, "gain": float}]}`

## GET /tradeoff?lambda_f=&lambda_c=&lambda_a=
Returns the nearest precomputed sweep point.
Res 200: `{"requested": {...}, "nearest": {"lambda_f": float, "lambda_c": float, "lambda_a": float}, "metrics": {"auc_roc": float, "auc_pr": float, "dir": float, "eod": float, "worst_group_ece": float}, "frontier": [ {...} ]}`

## GET /meta/options
Res 200: `{"categorical": {"name_family_status": ["Married", "..."], "...": []}, "numeric": {"age_years": {"min": 18, "max": 100}, "...": {}}, "risk_bands": {"low_max": 0.08, "medium_max": 0.20}, "include_ext_source": true, "model_version": str}`
Built from the bundle (encoders.json, ranges, config). The frontend builds its form from this; nothing is hard-coded in JS.

## GET /examples
Res 200: `{"examples": [ApplicantRecord, ...]}` (5 to 10 rows sampled from the held-out TEST set with identifiers removed and no gender field; fixed seed, saved in the bundle).

## GET /health
Res 200: `{"status": "ok", "model_version": str, "bundle_sha": str}`

## Error format (all endpoints)
```
{"error": {"code": str, "message": str, "details": [ {"field": str, "issue": str} ]}}
```
| HTTP | code | when |
|---|---|---|
| 422 | VALIDATION_ERROR | out-of-range / missing / wrong type fields |
| 400 | BAD_REQUEST | malformed JSON, gender value invalid |
| 404 | NOT_FOUND | unknown route |
| 503 | MODEL_NOT_READY | bundle failed to load or manifest hash mismatch |
| 500 | INTERNAL_ERROR | unexpected; logged with request id, no stack trace to client |

## Non-functional contract
- p95 latency: /predict < 300 ms, /predict+/explain < 3 s (single applicant).
- Deterministic: same input -> same output.
- No request body is logged with raw values; log request id, endpoint, latency only.

## Stateless and no-database contract
- No database, no file writes, no session store on the server. Handlers are pure functions of (request, frozen bundle).
- The browser may keep the last result in sessionStorage to render result.html; the server never sees it again.
- Static frontend is served by the same app (StaticFiles at "/"), so the demo runs with one command.
