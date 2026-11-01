# Spec 09: Resolved Decisions (defaults; do not re-ask, change only via config)

These replace every "e.g." / "TBD" elsewhere. If a value turns out unsuitable (e.g. loss terms on very different scales), adjust it in
the config, log the change in RUN_LOG.md, and apply it to ALL arms equally.

## Evaluation rules
- **Shared threshold rule**: flag the top 10% highest-risk applicants on the VALIDATION set of each arm (flag_rate = 0.10, from config).
  Same rule for every arm and for the post-hoc arms; the numeric threshold may differ per arm, the flag rate is identical. Applied unchanged to test.
- **Favourable outcome** = approved (not flagged). Approval rate per group = 1 - flag rate in group.
  `DIR = min(approval_M/approval_F, approval_F/approval_M)`. `DPD = |approval_M - approval_F|`.
  `EOD = max(|TPR_M - TPR_F|, |FPR_M - FPR_F|)` where "positive" = flagged, evaluated against TARGET.
- **ECE**: 15 equal-frequency bins, per group; worst-group ECE = max over groups. Also report calibration slope/intercept (logistic recalibration) and Brier.
- **Bootstrap**: 1,000 resamples of the test set, stratified by TARGET x gender, 95% percentile CI. Seeds: 0, 1, 2; report mean across seeds with pooled CI.
- Fairness type in loss: `eo` (main), `dp` (ablation).

## Model defaults (TabTransformer)
embedding dim 32, 4 heads, 3 transformer layers, attention dropout 0.1, FFN multiplier 2, categorical column dropout none;
numeric branch: LayerNorm then concat with flattened contextual embeddings; MLP head [128, 64], ReLU, dropout 0.1;
representation z = last hidden layer (64-d). Head B: Linear(64, 32) -> ReLU -> Linear(32, 1) behind the GRL.
1D CNN (A3): numeric+embedded vector treated as a sequence, 2 conv layers (32, 64 filters, kernel 3), global max pool, MLP [64].
MLP (A4): [256, 128, 64], ReLU, dropout 0.1.
Optimiser AdamW, lr 1e-3 (tuned in range 3e-4..3e-3), weight decay 1e-4, batch 2048, max 30 epochs, early stopping patience 4 on val AUC-PR, bf16 autocast.
Optuna search space (same SIZE for all arms): lr, weight decay, dropout; plus embedding dim for transformer arms; plus focal gamma (1..3) for A5.

## Imbalance
Class weights = n_total / (2 * n_class) from train. Focal loss gamma tuned in [1, 3], alpha from class prior.
Prior correction (S1): logit shift `-log(w_pos / w_neg)` applied on logits (analytic), checked on val. SMOTE-NC at sampling ratio 0.5, categoricals marked.

## Joint loss
- Warm-up: 3 epochs with lambdas = 0 (weighted CE only). Then linear ramp over 5 epochs to target lambdas.
- Lambda grid (REVISED in Phase 4 after the smoke check, before any sweep ran): main sweep lambda_f in {0, 0.05, 0.1, 0.25, 0.5}, lambda_c in {0, 0.5, 1, 2}, lambda_a = 0 (20 points x 3 seeds). Original grid was lambda_f {0, 0.5, 2} x lambda_c {0, 0.5, 2} x lambda_a {0, 0.5, 1} (27 points).
  Reason: smoke runs showed lambda_f = 2 (normalised weight about 15) cost 0.10 to 0.2 AUC-ROC, and the GRL adversary at lambda_a 0.5 or 1 was unstable (AUC-ROC 0.46 to 0.62, adversary loss up to 8 to 16). This is NOT based on any frontier or sweep result (none existed). lambda_a is handled by a separate time-boxed rescue protocol (see RUN_LOG Phase 4): GRL with small lambda_a, then slower reversal, then a bounded confusion loss (a deviation from the GRL, labelled as such).
  If a loss term is on a very different scale, rescale the TERM (normalise by its initial magnitude), not the grid; log it.
- Soft TPR/FPR: sigmoid with temperature 0.1 around the flag-rate threshold (batch quantile). Soft ECE: 10 soft bins, Gaussian kernel.
- Stratified sampler: every batch has >= 64 samples per TARGET x gender cell.
- Discard sweep points with val AUC-ROC < 0.60 from the frontier.

## Operating-point selection rule (committed as configs/selection_rule.yaml before any sweep exists)
Revised after Phase 2: at a 10% flag rate DIR is >= 0.93 for every baseline, so it cannot discriminate. The binding fairness constraint is now EOD.
Among sweep points (mean over 3 seeds, VALIDATION only):
1. `EOD <= 0.04` (less than half the baseline A2 value of 0.086). `DIR >= 0.80` is kept only as a safety check.
2. `AUC-ROC >= AUC-ROC(A2) - 0.02`
3. `worst_group_ECE <= max(1.5 * worst_group_ECE(S1), 0.02)`. The 0.02 floor keeps the bound above the bootstrap noise level (S1 is about 0.008).
4. Choose the point with maximum AUC-PR.
If none satisfies all constraints, relax in this order and record: (3) multiplier 1.5 -> 2.0, then (2) margin 0.02 -> 0.03.
If still none, pick the max-AUC-PR point with DIR >= 0.80 and set a soft flag.

## Matched-pair check
For each test applicant of gender g, match to the nearest applicant of the other gender in the same split (k=1), distance =
Euclidean on scaled numeric features + 0.5 per mismatching categorical feature (Gower-style). Reference pool for the API = test set. Report mean, median, p95 of |delta_prob|.

## Explainability
SHAP via `shap.GradientExplainer` on the torch model (fallback `KernelExplainer` with 50 background rows). Background = 100 train rows saved in the bundle.
Must meet predict + explain < 3 s on CPU; if not, reduce background to 50, then switch to top-k features only; record in RUN_LOG.

## Risk bands (placeholder for demo)
Low < 0.08, Medium 0.08 to 0.20, High > 0.20 on the final probability (after prior correction).

## Experiment tracking (MLflow 3.x)
MLflow 3.17 disables the plain `file:` tracking backend by default. Decision: use SQLite.
- Tracking URI: `sqlite:///artifacts/mlflow.db` (set in configs/base.yaml as `mlflow.tracking_uri`). Artifact root: `artifacts/mlflow_artifacts/`.
- Do NOT use `file:./mlruns` and do NOT set MLFLOW_ALLOW_FILE_STORE.
- Open the UI with: `.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///artifacts/mlflow.db`
- Phase 1 environment check must log one tiny run to this URI and read it back (PASS/FAIL).
- Never log gender values, raw rows or the datasets to MLflow; only params, metrics and small artifacts (plots, JSON).

## Decisions made during Phase 1 (accepted by the owner)
- Numeric outlier clipping at quantiles [0.001, 0.999], fitted on train only (configs/data.yaml `clip_quantiles`). Keep for all arms and both datasets.
- Python 3.13 is the project interpreter (docs saying 3.11 are superseded).
- Phase 1 baseline leakage: gender-from-X AUC 0.90 (LightGBM, test). Top proxies: OCCUPATION_TYPE, ORGANIZATION_TYPE, FLAG_OWN_CAR, EXT_SOURCE_1, AGE_YEARS. Default rate women 7.00%, men 10.14%.


## Decisions made after the Phase 2 review (owner)
- A1 (SMOTE-NC, ratio 0.5) is kept as run. Gate result unchanged: it shifts the prior ~5.7x vs ~11.4x for class weights, so it is not a like-for-like comparison with A2. Added A1b: SMOTE-NC at ratio 1.0 (matched prior shift), same tuning budget, 3 seeds.
- REF (LightGBM) is a reference ceiling. Fit on the full train set with early stopping on validation; its row is replaced if the tuned fit is better. No TabTransformer-vs-trees statement until resolved.
- S2 is split in two. S2-GT = S1 + group thresholds (analysis only, uses gender at inference, not deployable, validation-optimistic). S2-EG = S1-corrected scores + fairlearn ExponentiatedGradient (no gender at inference), the deployable sequential baseline.
- Optuna: `n_startup_trials = 4`, fixed sampler seed, so TPE drives trials 5 to 10.
