# Spec 02: Functional Requirements

Format: ID | requirement | input | output | stage

## Offline pipeline
| ID | Requirement | Input | Output | Stage |
|---|---|---|---|---|
| P-01 | Load raw CSV, drop XNA gender rows, fix DAYS_EMPLOYED placeholder + flag, derive AGE_YEARS, YEARS_EMPLOYED | application_train.csv | clean DataFrame | 1 |
| P-02 | Impute (median / "Missing"), label-encode categoricals (0 = Missing), binary Y/N -> 0/1 | clean DF | encoded DF | 1 |
| P-03 | Joint stratified split on TARGET x CODE_GENDER (70/15/15, seed 42) | encoded DF | train/val/test + aligned gender arrays | 1 |
| P-04 | Fit StandardScaler on train numerics; save encoders, cardinalities, feature order, class weights | train | artifacts/prep/* | 1 |
| P-05 | Run data checks (pandera + asserts) and print GATE report | splits | pass/fail + table | 1 |
| L-01 | Proxy leakage "before": LightGBM (+ logistic probe) predicts gender from X; report AUC val/test, top-15 gain features | X, A | artifacts/leakage/before.json + PNG | 2 |
| B-01 | One shared trainer: model + loss chosen by config; early stopping on val; seeds; MLflow logging | config, data | run folder with weights, val/test predictions | 3 |
| B-02 | Models: TabTransformer, 1D CNN, MLP | features | logits (+ representation z for TabTransformer) | 3 |
| B-03 | Losses: CE, weighted CE, focal; SMOTE-NC applied to train only | | | 3 |
| B-04 | Post-hoc: analytic prior correction (S1); fairlearn ThresholdOptimizer / ExponentiatedGradient (S2); per-group Platt/isotonic (analysis only) | val predictions | corrected predictions | 3/7 |
| B-05 | Group-wise calibration loss baseline (G1) | | | 3 |
| B-06 | LightGBM reference ceiling and gender-included reference (GI) | | | 3 |
| M-01 | Metric battery (see ARCHITECTURE) with one shared threshold rule chosen on val | y, p, A | metrics dict | 3 |
| M-02 | Bootstrap 95% CIs over test set; mean over 3 seeds | | table CSV | 3/8 |
| J-01 | GRL layer (identity fwd, -lambda bwd) | | | 4 |
| J-02 | Soft fairness loss (eo / dp), soft-binned per-group ECE loss, adversarial loss | | | 4 |
| J-03 | Joint loss with warm-up and lambda ramp; stratified batch sampler | | | 4 |
| J-04 | lambda=0 equivalence test vs A2 loss on fixed batch | | pass/fail | 4 |
| S-01 | Lambda sweep: coarse grid then refine near frontier; full metrics per point on val, 3 seeds | sweep.yaml | sweep results, Pareto frontier plot | 5 |
| S-02 | Apply pre-committed selection rule to val results | selection_rule.yaml | chosen config | 6 |
| V-01 | Final test evaluation of all arms and JT (once) | frozen configs | final tables | 8 |
| V-02 | Leakage "after": fresh probe on frozen z (A2 vs JT) + SHAP-vector probe | z, SHAP | AUC table | 8 |
| V-03 | Matched-pair check (nearest neighbour of other gender) | test X, model | delta distribution | 8 |
| V-04 | UCI Taiwan retrain with identical hyperparameters and lambdas | UCI Taiwan data | metrics table | 8 |
| F-01 | Freeze bundle with manifest.json (sha256, commit, config hash) | run | artifacts/bundle/<version>/ | 9 |

## Serving (from SRS FR-01..FR-09)
| ID | Requirement | Input | Output |
|---|---|---|---|
| A-01 | Collect and validate applicant form (FR-01) | form fields | validated record |
| A-02 | Preprocess with the training transforms (FR-02) | record | model-ready vector; gender never used |
| A-03 | Predict default probability + risk band Low/Medium/High (FR-03) | vector | prob in [0,1], band |
| A-04 | SHAP explanation, ranked factors (FR-04) | vector | feature contributions |
| A-05 | Matched-pair check for this applicant (FR-05) | vector (+ optional gender label for matching only) | matched applicant score, delta, global mean delta |
| A-06 | Fairness metrics on held-out test (FR-06) | | DPD, EOD, DIR, per-group TPR/FPR |
| A-07 | Reliability per group (FR-07) | | curve points, ECE, Brier |
| A-08 | Proxy leakage audit (FR-08) | | AUC before/after, ranked proxy features |
| A-09 | Final report page combining A-03..A-07 (FR-09) | | rendered page |
| A-11 | Form options and ranges from the bundle (`/meta/options`), example applicants from the test set (`/examples`) | | options, example records |
| A-12 | Serve the static frontend from the same app; no database; no persistence of applicant data | | |
| A-10 | Trade-off explorer: precomputed lambda sweep surface (slider) | lambda | metrics at nearest sweep point |

## Frontend pages
1. Applicant form. 2. Result: score, band, SHAP waterfall, matched-pair panel. 3. Dashboard: fairness bars, reliability
diagrams, lambda slider redrawing the trade-off chart (Chart.js).

## Risk band thresholds
Defined in configs (default: Low < 0.08, Medium 0.08 to 0.20, High > 0.20 on the calibrated probability). Must be shown to user in plain language.
