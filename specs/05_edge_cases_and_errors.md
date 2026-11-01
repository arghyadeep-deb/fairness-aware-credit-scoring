# Spec 05: Edge Cases and Error Handling

## Data
| Case | Handling |
|---|---|
| CODE_GENDER == "XNA" (4 rows) | drop; log count |
| DAYS_EMPLOYED == 365243 | NaN + EMPLOYED_MISSING flag; impute median on train |
| Missing OCCUPATION_TYPE / ORGANIZATION_TYPE | category "Missing" (index 0) |
| EXT_SOURCE_* missing (common) | median impute + missing-indicator column (config) |
| Unseen category at val/test/serve | index 0, never raise |
| Constant or near-constant columns after split | warn in checks; do not silently drop |
| Duplicate SK_ID_CURR | assert none |
| Group too small in a batch / split cell | stratified sampler guarantees min per cell; checks fail if any TARGETxGENDER cell < 50 in a split |
| Outliers (AMT_INCOME_TOTAL extreme) | keep raw values; clip at configured percentile only in scaler stage, documented |

## Training
| Symptom | Cause | Handling |
|---|---|---|
| Loss NaN | log(0), high LR | clip probs to [1e-6, 1-1e-6]; grad-clip 1.0; abort run and record failure |
| AUC collapses ~0.5 after adversary starts | lambda_a too high / ramp too fast | longer warm-up, smoother ramp; sweep must record AUC floor |
| Constant predictions | lambda_f too high | lower lambda_f; sweep excludes configs with AUC-ROC < 0.60 on val from the frontier |
| Fairness oscillates | adversary/trunk instability | lower adversary LR, larger stratified batch |
| Batch lacks a group/class | sampler bug | raise error in sampler test; fairness/cal terms return 0 and log warning only if a group is absent |
| CUDA unavailable | env | env_check fails loudly; allow CPU only with `--allow-cpu` flag |
| Run interrupted | | checkpoint every epoch; resume supported; never overwrite finished runs |

## Evaluation
| Case | Handling |
|---|---|
| ECE with empty bins | skip empty bins, report bin counts |
| Metric undefined (e.g. no positives in a group bootstrap sample) | resample again; log skips; if > 1% skipped, flag |
| Different thresholds across arms | forbidden; shared `threshold.py`, test asserts identical rule |
| Test set accessed before stage 8 | guard: `load_test()` requires env `ALLOW_TEST=1` and logs the access |
| DIR when favourable-rate is 0 for a group | return NaN, report, do not coerce to 1 |

## Leakage audit
| Case | Handling |
|---|---|
| Probe AUC < 0.60 | print loud warning: premise weak, discuss with mentor; still save results |
| Shuffled-gender sanity probe not ~0.5 (+/-0.02) | test fails -> leakage bug |
| Fairness improves but new proxy appears | SHAP-vector probe + feature ranking reported |

## Serving
| Case | Handling |
|---|---|
| Missing/invalid field | 422 with field-level details |
| Unknown category string | map to 0, add `warnings: ["unseen category: X"]` to response |
| Bundle missing or hash mismatch | /health 503; all model endpoints 503 MODEL_NOT_READY |
| SHAP slow | cap background size; return 503 with code TIMEOUT if > 3 s, never hang |
| Matched-pair pool has no neighbour of other gender | return 404-style error body with code NO_MATCH |
| Train/serve skew | parity test: 100 test rows offline vs API, max abs diff < 1e-5 |
| Concurrent requests | demo is single-user sequential (NFR-08); no shared mutable state in handlers |
| Large/garbage payload | body size limit, JSON schema validation first |

## Frontend
| Case | Handling |
|---|---|
| API down | friendly message, retry button |
| Slow response | spinner, 10 s timeout |
| Lambda slider between sweep points | show nearest point and label it as nearest |
