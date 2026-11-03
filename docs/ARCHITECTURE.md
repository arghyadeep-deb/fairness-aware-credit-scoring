# Architecture

## Pipeline (offline first, then serve)
```
[0 ENV] -> [1 DATA PREP] -> [2 LEAKAGE AUDIT "before"] -> [3 BASELINES A0-A5 + seq. pipelines + GBT ref]
   -> [4 JOINT MODEL + unit tests] -> [5 LAMBDA SWEEP (val, 3 seeds)] -> [6 OPERATING POINT (pre-committed rule)]
   -> [7 POST-HOC ARMS] -> [8 VERIFICATION (test once, probe on z, matched-pair, UCI Taiwan retrain)]
   -> [9 FREEZE BUNDLE + manifest] -> [10 FASTAPI + FRONTEND] -> [11 FIGURES + REPORT from bundle]
```
Each stage reads the previous stage's artifacts and writes its own. No stage reads raw data except stage 1.

## Repo layout
```
fairness-credit/
  CLAUDE.md
  docs/            context, architecture, rules, style, libs, commands
  specs/           01..06 spec files
  configs/         data.yaml, model_*.yaml, arm_*.yaml, sweep.yaml, selection_rule.yaml
  data/raw/        application_train.csv (git-ignored)
  data/processed/  train|val|test .parquet, gender_*.parquet (git-ignored)
  src/
    data/          prep.py, splits.py, checks.py
    models/        tabtransformer.py, cnn1d.py, mlp.py, heads.py
    losses/        weighted_ce.py, focal.py, fairness.py, calibration.py, grl.py, joint.py
    train.py       ONE trainer for all arms (model + loss chosen by config)
    eval/          metrics.py, threshold.py, leakage.py, matched_pair.py, bootstrap.py
    posthoc/       prior_correction.py, recalibration.py, threshold_optimizer.py
    sweep.py       lambda grid, Pareto frontier
    select.py      applies selection_rule.yaml
    freeze.py      builds artifact bundle + manifest.json
  api/             main.py, schemas.py, service.py, explain.py
  web/             index.html, result.html, dashboard.html, css/, js/
  tests/           test_data.py, test_leakage.py, test_grl.py, test_losses.py, test_metrics.py, test_parity.py, test_api.py
  artifacts/       prep/, leakage/, runs/<timestamp>/, bundle/<version>/
  notebooks/       EDA only, never imported by src/
```

## Model
- Shared trunk: TabTransformer. Categoricals -> learned embeddings -> transformer encoder (contextual embeddings);
  numerics -> LayerNorm/scaled -> concatenated with flattened contextual embeddings -> MLP -> representation z.
- Head A: z -> P(default). Served output.
- Head B: z -> GRL -> P(gender). TRAINING ONLY. Never used at inference.
- GRL: identity forward, gradient multiplied by -lambda_a backward.

## Joint loss
`L = L_wCE + lambda_f * L_fair + lambda_c * L_cal + lambda_a * L_adv`
- L_wCE: cross-entropy with class weights from the train label distribution.
- L_fair: differentiable soft surrogate. Config `fairness_type: eo | dp`. Main run: equalized-odds surrogate (soft TPR/FPR gap
  across gender). DP is an ablation.
- L_cal: soft-binned per-group ECE (differentiable). Probabilities clipped to [1e-6, 1-1e-6].
- L_adv: cross-entropy of Head B on gender, through GRL.
- Training schedule: warm-up epochs on L_wCE only; then ramp lambda_a (and lambda_f, lambda_c) linearly to targets.
- Batches: stratified sampler so every batch contains both genders and both classes.
- Sanity identity: with all lambdas = 0 the joint loss value on a fixed batch equals A2's loss value.

## Arms (all via the same trainer and the same metric battery)
| Arm | Model | Treatment |
|---|---|---|
| A0 | TabTransformer | plain CE |
| A1 | TabTransformer | SMOTE-NC (train only) |
| A2 | TabTransformer | weighted CE |
| A3 | 1D CNN | weighted CE |
| A4 | MLP | weighted CE |
| A5 | TabTransformer | focal loss |
| S1 | TabTransformer | A2 + analytic prior correction |
| S2 | TabTransformer | S1 + fairlearn ThresholdOptimizer or ExponentiatedGradient (sequential patching) |
| G1 | TabTransformer | weighted CE + group-wise calibration loss ("Towards Fair and Calibrated Models") |
| AB-adv / AB-fair / AB-cal | TabTransformer | one joint term at a time (ablations); pairs too |
| JT | TabTransformer | full joint objective at the pre-committed operating point |
| REF | LightGBM | scale_pos_weight, reference ceiling, not an arm |
| GI | best arm | gender INCLUDED as a feature (reference for the 1-6% finding) |

Equal Optuna trial budget per tuned arm. 3 seeds per arm. Bootstrap CIs on test metrics.

## Metric battery (fixed, identical for every arm)
- Discrimination: AUC-ROC, AUC-PR, KS.
- Fairness (at the shared threshold): demographic parity diff, equalized-odds diff, disparate impact ratio, per-group TPR/FPR.
- Calibration: per-group ECE, worst-group ECE, calibration slope/intercept, Brier.
- Proxy leakage: gender-from-features AUC (before), gender-from-z AUC (after), gender-from-SHAP AUC, mean matched-pair delta.

## Leakage measurement (important)
- "Before": LightGBM predicts gender from raw features X. Baseline AUC.
- "After": raw X is unchanged by training, so probe the learned representation z: train a fresh LightGBM/logistic probe on
  FROZEN z (from a trained arm) to predict gender. Compare gender-from-z AUC for A2 (no adversary) vs JT (adversary).
- SHAP probe: classifier predicting gender from per-row SHAP vectors.

## Matched-pair check (replaces gender flipping)
For each test applicant, find the nearest neighbour (on scaled features, excluding gender) of the OTHER gender; compare model
scores. Report mean and distribution of |delta|. The model never sees gender, so flipping a gender column would always give delta = 0.

## Artifact bundle (frozen, versioned, loaded by API at startup)
weights, scaler.joblib, encoders.json, cardinalities.json, feature_order.json, class_weights.json, shap_background.parquet,
prior_correction.json, lambda_sweep.json, metrics_test.json, calibration_curves.json, matched_pair_index (fitted neighbour
structure + reference pool), manifest.json (sha256 of every file, git commit, config hash, seeds, timestamp).

## Serving (backend + frontend, built in Phase 7 from the frozen bundle)
**No database anywhere in the app.** The API is stateless: it loads the frozen bundle (files) once at startup, scores each request in memory,
returns the result, and stores nothing about applicants. The only database in the project is the SQLite file MLflow uses during training.
No SQLAlchemy, MySQL, PostgreSQL or any DB driver in api/.

### Backend (FastAPI)
```
api/
  main.py       app factory, lifespan (load bundle once), CORS (local origins only), exception handlers, routes, StaticFiles mount of web/ at "/"
  schemas.py    pydantic models: ApplicantRecord, responses, error body (specs/03)
  bundle.py     load artifacts/bundle/<version>/, verify manifest sha256 of every file, expose handles; failure => /health 503
  service.py    preprocess with the SAME transform functions as training (src/data), predict, apply prior correction, risk band
  explain.py    SHAP (GradientExplainer, 100-row background from the bundle), top-k contributions, time budget
  matched.py    matched-pair lookup against the saved reference pool (test set), population mean |delta|
```
- One process, one command: `uvicorn api.main:app --port 8000` serves both the API and the frontend (open http://127.0.0.1:8000/).
- Read-only data: precomputed JSON for /fairness, /calibration, /proxy-audit, /tradeoff; the model object for /predict, /explain, /matched-pair.
- Logging: request id, endpoint, status, latency only. Never log applicant values or gender.
- CPU inference is enough for the demo (GPU optional).

### Frontend (plain HTML/CSS/JS, no framework, Chart.js from CDN, no build step)
```
web/
  index.html      Page 1: applicant form
  result.html     Page 2: result
  dashboard.html  Page 3: fairness / calibration / proxy audit / trade-off
  css/style.css   one shared stylesheet, responsive, light theme, accessible contrast
  js/api.js       fetch wrapper (base URL, timeouts, error body parsing)
  js/form.js, js/result.js, js/dashboard.js, js/charts.js
```
- Page 1: fields from ApplicantRecord; dropdown options and numeric ranges come from `GET /meta/options` (never hard-coded); "Load example applicant"
  button using `GET /examples`; client-side validation mirrors the API ranges; submit calls /predict, /explain, /matched-pair, stores the response in
  sessionStorage (browser only, never sent to a server store) and goes to result.html. No gender field in the form (the gender choice for /matched-pair is
  a separate, clearly labelled optional dropdown on the result page: "compare with a similar applicant of the other gender").
- Page 2: probability and risk band (with the band thresholds shown in plain words), SHAP contribution bars (top 10), matched-pair panel (own score vs matched
  score, delta, population mean |delta|), a short note on what the numbers mean.
- Page 3: fairness bars (DPD, EOD, DIR, per-group TPR/FPR with CI), reliability diagrams per gender group with ECE, proxy-audit table (leakage AUC before/after,
  top proxy features), lambda sliders that call /tradeoff and redraw the Pareto/trade-off chart, with the nearest sweep point labelled.
- The top disclaimer banner was removed at the owner's request (2026-11-03). The result page's "What the numbers mean" note still says the score is a research output, not an approval decision.
- States: loading spinner, API-down message with retry, 422 field errors shown next to the field, empty/NO_MATCH message for matched-pair.
- Works offline except Chart.js CDN (pin the version; optionally vendor a local copy in web/js/vendor/ so the demo works without internet).
