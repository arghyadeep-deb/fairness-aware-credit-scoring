# RUN_LOG

Soft flags (top of file, newest first): none raised so far.
Open gate item: Phase 2 Stage 3 "A1 calibrates worse than A2" FAILED (A1 ECE 0.152 < A2 ECE 0.342). Reported, not weakened. Needs the owner's decision.

## Phase 0: setup check (2026-11-01) PASS
- Installed in .venv: jupyterlab, ipykernel, nbformat, nbconvert, nbclient (already listed in requirements.txt). Kernel `fairness-credit` registered.
- `python src/env_check.py`: torch 2.11.0+cu128, CUDA True, RTX 5070 Ti Laptop, 4096x4096 matmul OK, lightgbm 4.7.0, sklearn 1.9.1,
  MLflow SQLite round-trip PASS (`sqlite:///artifacts/mlflow.db`), Home Credit 307,511 rows, UCI Taiwan 30,000 rows.
- Earlier readiness check failed MLflow only because it used the `file:` backend; SQLite per specs/09 works.
- git init, commit `phase0: ...`. Never pushed.
- Note: environment is Python 3.13.13; docs say 3.11 (3.10 to 3.13 accepted by preflight). Kept 3.13.

## Phase 1: data prep + leakage "before" (2026-11-01) PASS
Gate (specs/06 Stages 0 to 2):
- Stage 0 env_check: PASS
- Stage 1 data prep: PASS (all checks in `src/data/checks.py`; 4 XNA rows dropped, 307,507 rows kept)
  - rows train/val/test = 215,254 / 46,126 / 46,127
  - default rate = 8.07% in every split; female share = 65.84% / 65.83% / 65.84%
  - default rate by gender: F 7.00%, M 10.14%; smallest TARGET x gender cell per split = 1,598 rows
- Stage 2 leakage before: PASS
  - gender-from-features AUC (LightGBM): val 0.8991, test 0.9003 (best iteration 610)
  - logistic probe: val 0.8745, test 0.8769
  - shuffled-gender control: test 0.5030 (within 0.5 +/- 0.02)
  - top proxies by gain: OCCUPATION_TYPE 0.393, ORGANIZATION_TYPE 0.125, FLAG_OWN_CAR 0.109, EXT_SOURCE_1 0.063, AGE_YEARS 0.054
- pytest: 19 passed (tests/test_data.py, tests/test_leakage.py)
- Notebook: notebooks/01_data_and_leakage_before.ipynb, executed with nbconvert from a fresh kernel, 7 code cells, 0 errors.
- Soft flags: none (AUC 0.90 >= 0.60).
- Artifacts: data/processed/*.parquet, artifacts/prep/*, artifacts/leakage/{before.json, before_<ts>.json, top15_before.json, top15_proxy_features_before.png}, artifacts/guard/test_access.jsonl.

Decisions and notes to remember:
- Outlier clipping: specs/05 allows clipping "only in scaler stage at a configured percentile". configs/data.yaml sets
  `clip_quantiles: [0.001, 0.999]` (fit on train, applied to all numerics incl. derived). Set to null to keep raw values. This was my choice, not in specs/09; it applies to all arms equally.
- Test-set access: specs/08 requires test AUC for the leakage probe. The probe predicts gender and never loads TARGET. Access goes through
  `src/data/io.load_split` which logs it (`artifacts/guard/test_access.jsonl`, 1 entry, purpose `leakage_probe_gender_only`).
  Specs/07 lists "guard log not empty" as a hard stop before Phase 6, so Phase 6 must treat this single, gender-only entry as expected.
  No model was evaluated or tuned on test. Data checks read descriptive split statistics only.
- Compute: smoke run (5% rows) passed first, outputs in artifacts/smoke/. Full prep ~1 min, leakage ~1 min. About 5 GPU-free minutes used in total.
- .gitignore: added `!artifacts/RUN_LOG.md` so the run log is tracked and sessions can resume from git.
- Added pyproject.toml (black/ruff, line length 100).
- Next: Phase 2 (shared trainer, models, losses, baselines A0-A5, S1, S2, G1, REF, GI; 3 seeds). Projected: the longest compute so far; measure the first epoch before committing to the full run.

## Phase 2: baselines (2026-11-01) 1 GATE ITEM FAILED
Run folder: artifacts/runs/20261101T052529Z_phase2 (git hash at run time f00d266). All numbers on VALIDATION (test locked; guard log still 1 entry).
Gate (specs/06 Stage 3):
- A0-A5, S1, S2, G1, REF (+GI) run with 3 seeds through one trainer, equal tuning budget (10 trials x 8 epochs x 30% subsample each): PASS
- Every arm has a mean +/- 95% CI row (results.csv, 1,000 paired stratified bootstrap resamples): PASS
- A0 AUC-ROC: 0.746 with EXT_SOURCE (expected "higher with it"): PASS
- A2 higher recall and worse ECE than A0: PASS. ECE 0.342 vs 0.004. Recall at p>0.5 is 0.678 vs 0.003; at the shared 10% flag rate 0.320 vs 0.316 (equal within noise). S1 repairs ECE to 0.0035: PASS
- A1 calibrates worse than A2: **FAIL**. A1 ECE 0.152 < A2 0.342. SMOTE-NC at ratio 0.5 raises the prior ~5.7x vs ~11.4x for class weights. A1 is worse on discrimination (AUC-ROC 0.686). Not changed to pass.
- Group-wise calibration of A0 reported: ECE female 0.0059, male 0.0092: PASS
- pytest: 46 passed. Notebook: notebooks/02_baselines.ipynb executed, 9 code cells, 0 errors.
Key numbers (AUC-ROC / AUC-PR / ECE / DIR / EOD): A0 .746/.221/.004/.942/.088; A2 .746/.220/.342/.943/.086; S1 .746/.220/.0035/.943/.086;
S2 .746/.220/.0035/.988/.0035; A4 .747/.222/.345/.936/.099; A3 .714/.198; G1 .743/.218/.178/.971/.036; GI .749/.224/.341/.909/.163; REF .699/.169.
Soft flags: none (A2 does reproduce the imbalance-calibration damage). Observation: REF (LightGBM) is below the neural arms; untuned, not investigated.
Decisions and notes:
- Validation-only evaluation: specs/09 says bootstrap on test, but test is locked until Phase 6 (hard stop). Phase 6 repeats the table on test.
- S2 uses group thresholds that keep the shared 10% flag rate (equalized-odds gap minimised on val); fairlearn ThresholdOptimizer sets its own operating point (~33% flagged), so it is a separate row S2_fairlearn. S2 thresholds were fit and scored on val, so its fairness numbers are optimistic until Phase 6.
- G1 uses the soft-binned group ECE loss (10 bins, sigma 0.05), fixed lambda_c = 1.0 (not tuned). The loss is in src/losses/calibration.py and is reused in Phase 3.
- Optuna: 10 trials are all TPE random start-up trials, same sampler seed for every arm, so A0/A2/A5/G1 picked the same configuration. Search-space size differs only as specified in specs/09 (emb_dim for transformer arms, gamma for A5).
- GI = A4 configuration (best mean val AUC-PR among A0-A5) with gender added as a continuous input. Reference only.
- Bootstrap ECE is biased upward for well-calibrated arms: percentile CI of A0/S1 ECE sits slightly above the point estimate.
- Compute: first real TabTransformer epoch 1.1 s (MLP 0.3 s, CNN 0.7 s, SMOTE-NC 22 s). Tuning 4 min, final training 7 min, evaluation 1 min (about 13 min total, far under the 10 h cap; no reductions).
- Files: artifacts/runs/20261101T052529Z_phase2/{a0..g1,s1,s2,s2_fairlearn,ref,gi}/, results.csv, results_per_seed.json; MLflow experiment phase2_baselines (SQLite).
- Next: Phase 3 (GRL, soft fairness loss, adversarial head, joint loss + unit tests). Short compute, mostly unit tests.
