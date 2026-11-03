# Reproducing the study

Fairness-aware credit scoring: an **empirical study** of how imbalance correction, a fairness intervention and calibration interact, plus a joint objective `L = weighted_CE + lambda_f*fairness + lambda_c*calibration + lambda_a*adversarial` that makes the trade-off explicit. It is not a new method. Data: Home Credit Default Risk (Kaggle; **not Indian data**) and, as a secondary check, UCI Default of Credit Card Clients (Taiwan).

This file is the public reproduction guide. It summarises the protocol, gives the run order, points to the resolved settings, and lists every known deviation and correction.

## 1. What is in this repository and what is not

| In git | Not in git |
|---|---|
| Code (`src/`, `scripts/`, `api/`, `web/`), configs (`configs/`), tests | Raw data (licence: download it yourself, section 2) |
| Served model bundle `artifacts/bundle/v1/` with sha256 manifest | Run folders `artifacts/runs/` (checkpoints, per-seed predictions) |
| Result tables `artifacts/tables/`, figures `artifacts/figures/` | Saved test predictions `artifacts/test_preds*/`, guard logs `artifacts/guard/` |
| Resolved settings `reproduce/resolved_settings.json` | Internal planning notes |
| Executed notebooks `notebooks/01` to `08` | |

The bundle's `matched_pool.npz` holds the 46,127 held-out Home Credit rows (scaled features and gender) used for the matched-applicant search.

## 2. Data

- **Home Credit:** `application_train.csv` from the Kaggle competition "Home Credit Default Risk" (accept the competition rules first). Put it at `data/raw/application_train.csv`. Only this table is used (no bureau or previous-application tables).
- **Taiwan:** UCI "Default of Credit Card Clients" (30,000 rows, CC BY 4.0), in `data/raw/uci_taiwan/`.
- `scripts/download_data.ps1` downloads both (needs a Kaggle API token).

## 3. Environment

- Python 3.13; PyTorch with CUDA 12.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`, then `pip install -r requirements.txt`.
- `requirements.txt` is unpinned. **The exact versions used are in `artifacts/bundle/v1/requirements_freeze.txt`**; install from that file for a faithful rerun.
- Runs used float32 weights, at most 4 CPU threads, one process at a time. Phase 6 test scoring used bf16 autocast on GPU (section 6, item 5).
- `python src/env_check.py` checks the setup; `pytest -q` runs the test suite.

## 4. Protocol (fixed before results were seen)

1. **Gender is never a model input.** `CODE_GENDER` is held in a separate aligned array, used only by fairness terms, the adversary head, fairness post-processing fits and the evaluation. (Arm GI deliberately includes it, as an analysis-only reference.)
2. **Split:** stratified 70/15/15 on TARGET, seed 42. Clipping, medians, encoders, scaler and class weights are fitted on train only.
3. **One decision rule for every arm:** flag the top 10% of applicants, using the 90th percentile of VALIDATION scores as the threshold.
4. **Operating-point selection rule** (`configs/selection_rule.yaml`) was committed to git before the lambda sweep was run.
5. **Test set:** each arm is scored once; every access is written to a guard log before predictions are saved, and metrics are computed from the saved predictions. Arms added after the test run are labelled post-hoc.
6. **Metrics:** AUC-ROC, AUC-PR (average precision), KS, Brier, ECE with 15 equal-frequency bins and the worst of the two gender groups, calibration slope and intercept; at the decision rule: EOD = max(|TPR gap|, |FPR gap|), DPD and DIR on approval rates, recall, precision.
7. **Uncertainty:** 1,000 paired bootstrap resamples of the evaluation rows, models and thresholds held fixed (section 6, item 11). Seeds 0, 1, 2 (JT-A rows: 0 to 4).

## 5. Run order

Each stage writes a new timestamped folder under `artifacts/runs/` and never overwrites earlier results. Later stages take earlier folders as arguments; several scripts also read the pointer files `artifacts/*_rundir.txt` (write the folder path into them after each stage).

| Phase | Command |
|---|---|
| 1 Data and leakage before | `python -m src.data.prep --config configs/data.yaml`; `python -m src.eval.leakage --config configs/data.yaml` |
| 2 Baselines | `python -m src.run_arms --arms a0 a1 a2 a3 a4 a5 g1 s1 s2 ref gi --seeds 0 1 2`; `python -m src.eval.report --run-dir <p2>` |
| 3 Joint loss | `python -m src.run_joint smoke` |
| 4 Stabilise and sweep | `python -m src.run_joint sweep --run-dir <p4> --points coarse --seeds 0 1 2`; `scripts/phase4_stability.py`, `phase4_adv_rescue.py`, `phase4_select.py`, `phase4_finalize.py` |
| 4b Adversary rescue, H arms, tuned JT | `scripts/phase4b_grl_rescue.py`, `phase4b_h_arms.py`, `phase4b_tune_jt.py <out> 0.05,2,0`, `phase4b_select.py` |
| 5 Post-hoc comparison (validation) | `python scripts/phase5_compare.py <p2> <p4> <p4b>` |
| 6 Test (once per arm) and Taiwan | `python scripts/phase6_test.py access <p2> <p4> <p4b> <p6>`; `phase6b_jta.py`; `phase6_report.py <p2> <p6>`; `phase6_taiwan.py fit`, then `access`, then `analyze` (each with `<p2> <p6>`) |
| 7 Freeze and serve | `python scripts/freeze_bundle.py <p2> <p4> <p4b> <p6>`; `uvicorn api.main:app --port 8000` |
| 8 Figures and tables | `python scripts/make_all_figures_tables.py` |
| Post-hoc (declared after the test run) | `scripts/phaseE_diagnostics.py`, `phaseE_three_way.py`, `posthoc_ref_eg.py`, `posthoc_tuning.py`, `repeated_splits.py` |
| Corrections | `scripts/observed_diffs.py`, `scripts/ece_interval_check.py`, `scripts/make_synthetic_examples.py`, `scripts/export_resolved_settings.py` |

Each script's first lines give its exact arguments. Notebooks are built by `scripts/build_nb01.py` to `build_nb08.py` and only display saved results.

**Known obstacle:** `scripts/repeated_splits.py` checks the Home Credit guard log against a hash recorded before that run (`ORIG_GUARD_SHA`). Later post-hoc accesses changed the log, so a rerun needs that constant updated to the current log's hash.

## 6. Known deviations and corrections

1. **Effect sizes.** The original difference tables (`paired_diffs_test.csv`, `diag_paired_diffs.csv`, `three_way_diffs.csv`) used the mean of the bootstrap differences as the point estimate. The observed difference of seed means is in the matching `*_observed.csv` tables; use those. Example: JT chosen minus S2-EG, EOD +0.0213 (the original table shows +0.0177).
2. **Worst-group ECE comparisons are descriptive only.** In 8 of 347 difference rows the observed ECE difference lies outside its own percentile interval, and in one the sign reverses (observed -0.0003, interval +0.0063 to +0.0119). A synthetic check (`artifacts/tables/ece_interval_coverage.csv`) found no interval method reliable across scenarios: the percentile interval covered the true difference in 100%, 96% and 10% of replicates (no gap, small gap, large gap). No directional significance claim is made for ECE differences.
3. **KS with tied scores** was computed inside tie blocks before 2026-11-03 (fixed in `src/eval/metrics.py`). On saved test predictions the change is at most 0.0023; saved tables keep the old values.
4. **Checkpoint timing.** With warm-up 3 and ramp 5 epochs, checkpoint selection starts at epoch 7, which ends at 99.8% of full strength. 48 of 274 saved runs chose epoch 7. The served JT chosen seeds chose epoch 8 or later. Whether starting at epoch 8 in every candidate run could change the selected lambda point was not tested.
5. **Precision.** Test scoring used bf16 on GPU; serving uses fp32 on CPU. Maximum difference 0.00184 (the protocol's parity tolerance was 1e-5); this changes the flag for 37 and the risk band for 100 of 46,127 held-out applicants.
6. **Guard-log scope.** The log records model evaluations, not every read. Taiwan preparation (`src/data/taiwan.py`) builds the held-out arrays before dropping them, and Home Credit's descriptive checks (`src/data/checks.py`) read all splits. No use of test labels for tuning was found, but the log alone cannot certify that.
7. **Taiwan preprocessing** applies signed log1p and train-fitted scaling but not the clipping quantiles the protocol lists for both datasets. Taiwan results are as run; a corrected run would be a separate, declared experiment.
8. **REF configuration label.** `configs/arm_ref.yaml` describes REF as using `scale_pos_weight`. REF actually uses no class weights (so REF-S1 equals REF). The file is unchanged because the bundle manifest records a hash of the configs.
9. **Main table seeds:** 3 seeds per arm, 5 for the JT-A rows.
10. **Matched-pair comparator** labelled "A2" is A2 after prior correction, not A2's raw probabilities.
11. **What the intervals mean.** Bootstrap intervals resample evaluation rows with the fitted models and validation thresholds fixed. They do not include retraining-seed or threshold-selection variation; seed variation is reported separately (`_sd` columns, repeated splits).
12. **Repeated splits are exploratory.** Five new splits reuse settings chosen on split 42, and about 85% of each new test set was in split 42's train or validation data, so they are a robustness check, not an independent test.
13. **Post-hoc arms** (REF-S1, REF-EG, REF-T, ENS, REF-T-EG, ENS-EG) were declared after the original test results were known.

## 7. Demo

`uvicorn api.main:app --port 8000`, then open http://127.0.0.1:8000/. The "Load example applicant" button uses 8 synthetic applicants (`api/data/synthetic_examples.json`, built by `scripts/make_synthetic_examples.py` from training-row fields; each is farther from every real applicant than real applicants typically are from each other). For hosting, see the keep-alive notes in `api/keepalive.py`.
