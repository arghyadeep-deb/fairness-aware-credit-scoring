# RUN_LOG

Soft flags (top of file, newest first): none raised so far.
Recorded gate result (accepted by owner, unchanged): Phase 2 Stage 3 "A1 calibrates worse than A2" FAILED (first run A1 ECE 0.152 < A2 0.342; re-run 0.139 < 0.333). A1 at ratio 0.5 shifts the prior ~5.7x vs ~11.4x for class weights, so it is not a like-for-like comparison. A1b (ratio 1.0) ECE 0.237 is also below A2.

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

## Phase 2 review (2026-11-01): decisions and re-run
Owner decisions applied; commits f8189b2 (selection rule) and the Phase 3 commit below. New run folder: artifacts/runs/20261101T055729Z_phase2 (the first run 20261101T052529Z_phase2 is kept untouched).
1. A1 accepted as is. A1b added (SMOTE-NC ratio 1.0, same tuning budget, 3 seeds). ECE: A1 0.139, A1b 0.237, A2 0.333 (S1 0.006, A0 0.007). AUC-ROC: A1 0.686, A1b 0.668, A2 0.746.
2. REF investigated. Root cause was NOT the tuning budget or subsampling (the first REF already used the full train set): LightGBM also tracked the default logloss, which worsens immediately under scale_pos_weight, so early stopping kept boosting round 1 (best_iteration = 1). Fixed: metric auc only, first_metric_only early stopping with patience 200, lr 0.02, 31 leaves, no scale_pos_weight (config picked from 4 settings tried on validation: 0.758 to 0.765). New REF: AUC-ROC 0.7646 [0.757, 0.773], AUC-PR 0.241, 1,222 rounds, ECE 0.004. This replaces the REF row.
   Result now resolved, stated honestly: on validation LightGBM is ahead of the TabTransformer (A2 AUC-ROC 0.746, AUC-PR 0.221), +0.019 AUC-ROC with non-overlapping CIs; the MLP (A4 0.747) is no better than the transformer. REF EOD 0.097 is the largest of the unmitigated arms.
   Side note: the Phase 1 gender probe uses the same LightGBM early-stopping pattern (auc plus default logloss). It kept 610 rounds and reached AUC 0.900, so it was not stopped at round 1; if logloss stopped it early, the true leakage AUC can only be higher. Not changed (Phase 1 numbers stand).
3. Selection rule committed in f8189b2 (configs/selection_rule.yaml, specs/06, 07, 09 updated) before any sweep exists: EOD <= 0.04, AUC-ROC >= A2 - 0.02, worst-group ECE <= max(1.5 x S1, 0.02), max AUC-PR, DIR >= 0.80 safety only. Notes for Phase 5: in the re-run A2 EOD is 0.076 (0.086 in the first run), so 0.04 is 0.52x the baseline, not below half; and S1 worst-group ECE is 0.0123, so 1.5x = 0.0185 and the 0.02 floor is the binding bound. The file is not edited (rule 9).
4. S2 split. S2-GT (group thresholds): analysis only, uses gender, not deployable, validation-optimistic: EOD 0.005, DIR 0.987. S2-EG (fairlearn ExponentiatedGradient, equalized odds, base = logistic regression on frozen A2 representation z, fit on TRAIN, no gender at inference): EOD 0.033 [0.017, 0.059], DIR 0.973, recall 0.316; meets EOD <= 0.04. First attempt (base classifier flagged ~35%, class_weight balanced) collapsed to S1 under the shared rule (EOD 0.073): kept in s2_eg_v1_unmatched/ and results_v1_s2eg_unmatched.csv. Final version picks the positive class weight on train so the base classifier flags ~10% (weight ~4.3). S2-EG AUC/ECE columns are S1's (only the flags change); 1e-6 x S1 probability breaks ties in the mixture score. fairlearn ThresholdOptimizer kept as S2_fairlearn (own operating point, ~34% flagged, not rate-matched).
5. Optuna n_startup_trials = 4 (TPE for trials 5 to 10, seed 42). Tuning for all 8 tuned arms took 5.6 min (< 15 min limit), so all arms were re-tuned and re-trained in the new run. Hyperparameters changed (e.g. A2: lr 0.0026, dropout 0.25), A0/A2 AUC-ROC 0.744/0.746 (first run 0.746/0.746). Arms no longer all share one configuration.
6. notebooks/02_baselines.ipynb rebuilt for the new run (A1b, S2-GT, S2-EG, new REF and a first-vs-second REF table), executed, 10 code cells, 0 errors.
Re-run gate status: all Stage 3 items as before; "A1 worse than A2" still FAILED (recorded above). Key numbers (AUC-ROC / AUC-PR / ECE / DIR / EOD): A0 .744/.220/.007/.940/.083; A2 .746/.221/.333/.946/.076; S1 .746/.221/.006/.946/.076; S2-GT .746/.221/.006/.987/.005; S2-EG .746/.221/.006/.973/.033; G1 .744/.219/.187/.966/.045; REF .765/.241/.004/.935/.097; GI .749/.223/.345/.906/.165.
Compute: re-run about 20 min wall (tuning 5.6 min). Guard log unchanged (1 whitelisted entry).

## Phase 3: joint loss building blocks (2026-11-01) PASS (Stage 4 unit tests)
Gate (specs/06 Stage 4, unit-test part; the 3-epoch gender-head smoke run belongs to Phase 4 per specs/07):
- GRL forward identity, backward x (-lambda): PASS
- Fairness loss ~0 when both groups get identical predictions: PASS (exactly 0 for constant scores; <0.03 for group-independent scores; soft dp matches hard flag-rate gap within 0.003)
- Calibration loss ~0 on perfectly calibrated data: PASS (0.002; matches hard ECE 0.046 to 0.32 for logit shifts 0.25 to 2)
- All lambdas = 0: joint loss and gradients equal the A2 loss on a fixed batch, at warm-up, mid-ramp and fully ramped: PASS (difference exactly 0.0)
- Overfits a 200-row batch: PASS (joint loss 1.08 to 0.002 with fairness and calibration terms on, lambda_a = 0)
- Adversary inactive during warm-up: PASS (terms not computed, head receives no gradient, trunk gradient equals A2's)
- Not in Phase 3: gender-head AUC trending to 0.5 (Phase 4 smoke).
pytest: 73 passed (21 in tests/test_joint.py, 6 in tests/test_phase2_review.py). Notebook: notebooks/03_joint_model.ipynb executed, 10 code cells, 0 errors.
Design decisions:
- Joint loss = wCE + lambda_f*L_fair + lambda_c*L_cal + lambda_a*L_adv with the lambda on the loss value. The adversary head applies a GRL with coefficient 1 (pure reversal) so that lambda_a = 0 gives exactly the A2 loss and no influence on the trunk; trunk gradient from the adversary is -lambda_a x the plain-head gradient. Terms with effective lambda 0 are not computed.
- Soft flags: sigmoid((logit - batch quantile) / (T x batch std)), T = 0.1, quantile at 1 - flag_rate (0.10), threshold and std detached. The std scaling makes T scale-free (weighted CE inflates logits). eo = |soft TPR gap| + |soft FPR gap| (sum, slightly stricter than the reported max-based EOD); dp = |soft flag-rate gap|. A missing group or cell contributes 0 with a warning.
- Schedule: 3 warm-up epochs with all lambdas 0, then linear ramp to targets over 5 epochs (epoch may be fractional). Config: configs/train.yaml `joint:`.
- Not yet normalised by initial term magnitude (Phase 4 stabilisation, specs/09).
- Files: src/losses/{grl,fairness,joint}.py, src/models/heads.py, tests/test_joint.py, notebooks/03_joint_model.ipynb.
- Compute: unit tests plus demos only (seconds). Total project compute so far about 0.6 h of the 10 h cap.
- Next: Phase 4 (stabilise: stratified batch sampler, warm-up/ramp in the trainer, adversary stability, AUC floor, short joint runs on validation, gender-head AUC smoke).

## Phase 4: joint training, stabilisation + smoke (2026-11-01) STOPPED AT SMOKE, NO SWEEP RUN
Stop rule from the owner: if the default AUC collapses with the adversary at full strength, stop and report instead of sweeping. It triggered. The sweep, Pareto plot, selection rule, ablations, JT-vs-S2-EG comparison and notebook 04 were NOT done. configs/selection_rule.yaml is untouched and no sweep result exists.
Built (all tested, 84 pytest passed): per-term normalisation by initial magnitude (mean over the last warm-up epoch, frozen when the ramp starts; configs joint.normalize: initial; lambda = 0 still equals A2 exactly); stratified batch sampler (>= 64 per TARGET x gender cell, natural mix kept: only the smallest cell is lifted, 1.7% to 3.1%); joint trainer src/train_joint.py (A2 hyper-parameters, no re-tuning, early stopping only once lambdas are at full strength, lr-halving retry on NaN, AUC-floor flag); runner src/run_joint.py (smoke / resumable sweep); diagnostics scripts/phase4_stability.py. Normalisation references seen: fair 0.137, cal 0.350, adv 0.691.
Compute: about 6 min of GPU for all smoke and diagnostic runs (epoch about 1.3 s); project total about 0.7 h of 10 h.
Smoke (3 epochs, compressed schedule 1 warm-up + 1 ramp, seed 0, validation), AUC-ROC / EOD / worst-group ECE / adversary-head AUC / gender-from-z probe AUC at the last epoch:
- control (0,0,0), same as A2: 0.743 / 0.091 / 0.366 / 0.58 / 0.76
- adversary only (0,0,1): 0.517 / 0.044 / 0.086 / 0.59 / 0.79  <- AUC collapse
- full (2,2,1): 0.637 / 0.033 / 0.101 / 0.55 / 0.67
Real 3+5 schedule, 14 epochs (artifacts/smoke/joint_stability/*.json):
- adversary only, lambda_a 0.5: AUC-ROC 0.46 to 0.73, mostly 0.46 to 0.62 after the ramp; lambda_a 1.0: 0.46 to 0.54. Adversary-head AUC swings between 0.34 and 0.65. The adversary loss rises from 0.7 to 4.3 and 8.2 (confidently wrong), i.e. the trunk maximises an unbounded loss. The gender-from-z probe does not fall (0.67 to 0.81 against 0.75 at the start): gender is still recoverable, the head is only being fooled.
- A competent head did not help (diagnostic, flag joint.adv_pretrain, default off, not adopted: Head B fitted on detached z during warm-up, head AUC 0.67 to 0.70 at ramp start): AUC still falls to 0.48 to 0.52 at once, adversary loss peaks at 16.
- fairness only, lambda_f = 2 (normalised weight 2/0.137 = 14.6): AUC-ROC 0.53 to 0.66, EOD 0.006 to 0.04. Strong trade-off, EOD can be cut at a large AUC cost.
- calibration only, lambda_c = 2: AUC-ROC 0.743 to 0.745 (unchanged), worst-group ECE 0.016 to 0.027 (A2 0.36, S1 0.012), EOD 0.02 to 0.05. The calibration term works without costing ranking quality.
- fairness + calibration (2,2,0): AUC-ROC 0.58 to 0.68, EOD 0.006 to 0.04, worst-group ECE 0.02 to 0.10. Adding the adversary (2,2,1) changes this little (0.57 to 0.64): the AUC loss there comes from lambda_f, not the adversary.
Gender-head AUC trend, as asked: it does not start near 0.9 (the head sees z; its starting AUC is 0.58, a gender probe on z gives 0.75, a probe on the raw features gave 0.90 in Phase 1). With the adversary on it goes to about 0.5 or BELOW 0.5 (down to 0.34), which is the head being inverted, not information removed (probe on z unchanged). So it neither stays high nor cleanly converges.
Open decision for the owner (options in the Phase 4 report): (1) keep a GRL adversary but with a much smaller lambda_a grid; (2) a bounded adversarial objective (confusion loss) instead of unbounded reversal, which departs from the GRL in the spec; (3) run the sweep with lambda_a = 0 and report the unstable adversary as a negative result.

## Phase 4 continuation (owner decision after the smoke stop)
Grid change, committed BEFORE any sweep: lambda_f {0, 0.05, 0.1, 0.25, 0.5}, lambda_c {0, 0.5, 1, 2}, lambda_a = 0 for the main sweep (20 points x 3 seeds). Reason: in the smoke runs lambda_f = 2 cost 0.10 to 0.2 AUC-ROC and the GRL adversary was unstable. Not because of any frontier result: no sweep result existed. The adversary gets a separate time-boxed rescue (about 1 h compute) judged by a fresh probe on frozen z, ordered: (a) small lambda_a, (b) slower reversal, (c) bounded confusion loss (deviation from the spec GRL).
