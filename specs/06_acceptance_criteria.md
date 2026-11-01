# Spec 06: Acceptance Criteria (stage gates)

A stage is DONE only when every box is true, tests pass, and a GATE report is printed and saved.
Stages map to phases in specs/07_phased_build_plan.md (Phase 1 = Stages 0-2, Phase 2 = Stage 3, Phase 3-4 = Stage 4, Phase 5 = Stages 5-6, Phase 6 = Stages 7-8, Phase 7 = Stages 9-10, Phase 8 = Stage 11).

## Stage 0: Environment
- [ ] `python src/env_check.py` prints torch >= 2.7, CUDA True, GPU name, successful GPU matmul, lightgbm/sklearn versions.

## Stage 1: Data prep
- [ ] Gender absent from X; separate aligned gender parquet per split.
- [ ] No NaN after imputation; no ID overlap across splits; categorical indices within cardinality.
- [ ] Default rate and gender share per split within 0.5 percentage points of each other.
- [ ] Scaler fit on train only; train numeric mean ~0, std ~1.
- [ ] GATE prints: rows per split, default rate per split, % female per split, default rate per gender.
- [ ] `pytest tests/test_data.py` passes.

## Stage 2: Leakage "before"
- [ ] LightGBM + logistic probes trained on X only (never TARGET); val and test AUC reported.
- [ ] Shuffled-gender probe AUC within 0.5 +/- 0.02.
- [ ] Top-15 proxy features saved with chart. before.json saved with config + timestamp.
- [ ] Gender-from-features AUC >= 0.60, otherwise record a soft flag (top of SUMMARY.md, discuss with mentor) and continue.

## Stage 3: Baselines
- [ ] A0-A5, S1, S2, G1, REF run with 3 seeds each through the one trainer, equal tuning budget.
- [ ] Every arm has a full row (mean +/- 95% CI) in results CSV under the shared threshold rule.
- [ ] A0 AUC-ROC around 0.65-0.70 without EXT_SOURCE (higher with it).
- [ ] A2 shows higher recall than A0 and worse ECE (confirms the imbalance-calibration damage); S1 repairs most of that ECE.
- [ ] A1 calibrates worse than A2.
- [ ] Group-wise calibration of A0 reported (tests Liu et al. "calibration as byproduct").

## Stage 4: Joint model
- [ ] GRL test: forward identity; backward multiplies gradient by -lambda.
- [ ] Fairness loss ~0 when both groups receive identical predictions.
- [ ] Calibration loss ~0 on synthetic perfectly calibrated data.
- [ ] All lambdas = 0: joint loss equals A2 loss on a fixed batch with fixed weights.
- [ ] Overfits a 200-row batch to near-zero loss.
- [ ] Adversary inactive during warm-up (test).
- [ ] Gender-head AUC trends toward 0.5 (not 0, not 1) in a 3-epoch smoke run.

## Stage 5: Lambda sweep
- [ ] Coarse grid then refine; full metric battery per point on VALIDATION, 3 seeds.
- [ ] Pareto frontier plotted; frontier shows a real trade-off (not flat).
- [ ] No test data touched for modelling or tuning. The guard log may contain only the whitelisted entry `leakage_probe_gender_only` (Phase 1, with_target=false); any other entry is a failure.

## Stage 6: Operating point
- [ ] configs/selection_rule.yaml committed (git hash earlier than sweep results).
- [ ] Rule (specs/09): maximise AUC-PR subject to EOD <= 0.04, AUC-ROC >= A2 - 0.02, worst-group ECE <= max(1.5 x S1, 0.02); DIR >= 0.80 is only a safety check; one config chosen.

## Stage 7: Post-hoc comparison
- [ ] Prior correction, per-group Platt/isotonic (labelled analysis-only), ThresholdOptimizer compared with JT at matched fairness.

## Stage 8: Verification
- [ ] Test set evaluated once for all arms + JT (access log shows one entry per arm).
- [ ] Leakage "after": probe on frozen z for A2 vs JT; JT AUC lower than A2 (report even if not).
- [ ] SHAP-vector probe AUC reported; new-proxy check reported.
- [ ] Matched-pair delta distribution reported (mean, p95).
- [ ] UCI Taiwan retrained with identical hyperparameters/lambdas; metrics table produced.

## Stage 9: Freeze
- [ ] Bundle with manifest.json (sha256 of every file, git commit, config hash, seeds, requirements freeze).
- [ ] Bundle loads on a clean machine/venv.

## Stage 10: Serve
- [ ] All endpoints in spec 03 return documented schemas; error format matches.
- [ ] Parity test: 100 test rows offline vs API, max abs diff < 1e-5.
- [ ] /predict p95 < 300 ms; /predict + /explain < 3 s.
- [ ] NFR-03: deployed model DIR >= 0.8 on test (or reported honestly if the rule cannot be met).
- [ ] Unseen categories handled; 422 on invalid ranges; 503 when bundle missing.
- [ ] Three frontend pages work end to end, lambda slider redraws chart; form built from /meta/options; disclaimer visible on every page.
- [ ] No database used and no applicant data written to disk or logs (test scans the filesystem and logs after a batch of requests).
- [ ] One command (`uvicorn api.main:app`) serves both API and frontend.

## Stage 11: Paper readiness
- [ ] Every figure/table regenerated from the frozen bundle by one script.
- [ ] Claims reviewed against CRITICAL_RULES 19-22.
- [ ] Literature search completed (closest work: fair+calibrated training; credit proxy-leakage; imbalance-calibration).
- [ ] Threats to validity section written (one dataset family, one protected attribute, non-Indian data, TabTransformer vs GBT).
