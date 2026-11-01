# Spec 07: Phased Build Plan (8 phases, executed day by day)

The original 8-week plan is kept as 8 PHASES. A phase is a unit of work with its own code, tests, runs, gate and commit. It is NOT tied to a
calendar week. Each phase normally takes about one working day of Claude Code time (long compute runs in the background). Phase N must pass
its gate before Phase N+1 starts. Ignore any "Week N" wording in older documents: Week N = Phase N.

## Execution mode
- DEFAULT: phase by phase. Finish a phase, print its PHASE GATE report, commit, append to artifacts/RUN_LOG.md, then STOP and wait for the
  user to say "next phase". The user may close the session between phases; everything is resumable from git + RUN_LOG.md + artifacts.
- If the user says "run all phases" or "continue autonomously": go phase to phase without stopping, still obeying hard stops below.
- On every new session: read CLAUDE.md, docs/, specs/, then artifacts/RUN_LOG.md and `git log`, report which phase is current, and continue from there.
- Every phase has a --smoke check first (5% of rows, 2 epochs, 1 seed, tiny sweep; outputs to artifacts/smoke/, never used as results), then the
  full run. Smoke must be green before any long compute.

## Phase map (original week -> phase)
| Phase | Original week | Work | Gate (specs/06) |
|---|---|---|---|
| 1 | Week 1 | Env check, configs, skeleton; data prep + checks + tests (specs/08); leakage "before" audit | Stage 0, 1, 2 |
| 2 | Week 2 | Shared trainer; TabTransformer / 1D CNN / MLP; CE, weighted CE, focal, SMOTE-NC; S1 prior correction, S2 fairlearn, G1 group-calibration, REF, GI; metric battery, shared threshold rule, bootstrap CIs; run all baseline arms, 3 seeds | Stage 3 |
| 3 | Week 3 | GRL, soft fairness loss (eo, dp), soft-binned group ECE loss, adversarial head, joint loss; unit tests incl. lambda=0 equals A2 loss | Stage 4 (unit tests) |
| 4 | Week 4 | Stabilise joint training: stratified batch sampler, warm-up, lambda ramp, adversary stability, AUC floor; short joint runs on val | Stage 4 (smoke + stability) |
| 5 | Week 5 | Commit configs/selection_rule.yaml FIRST; lambda sweep (coarse then refine, 3 seeds on val); Pareto frontier; apply selection rule | Stage 5, 6 |
| 6 | Week 6 | Post-hoc comparison at matched fairness; verification: test set once, leakage probe on z, SHAP-vector probe, matched-pair check; UCI Taiwan retrain (same hyperparameters/lambdas) | Stage 7, 8 |
| 7 | Week 7 | Freeze bundle + manifest; FastAPI backend; 3-page frontend; parity test on the real bundle | Stage 9, 10 |
| 8 | Week 8 | Figures + tables from the frozen bundle; artifacts/SUMMARY.md; paper-readiness checklist (writing the paper itself is a separate task) | Stage 11 |

### Phase 7 task detail (backend + frontend, no database)
1. Freeze bundle: weights, scaler/encoders/feature_order, prior_correction, SHAP background (100 rows), matched-pair reference pool, precomputed JSON (fairness, calibration, proxy audit, sweep/frontier), examples.json, manifest.json with sha256.
2. api/bundle.py: load + verify hashes; fail closed (503) on mismatch.
3. api/schemas.py + api/service.py: ApplicantRecord validation, same preprocessing as training, calibrated probability, risk band.
4. api/explain.py, api/matched.py; endpoints per specs/03 incl. /meta/options and /examples; error format and handlers.
5. api/main.py: lifespan load, CORS (local only), StaticFiles mount of web/ at "/".
6. web/: three pages + css + js modules as in docs/ARCHITECTURE.md; form built from /meta/options; Chart.js pinned (vendor local copy if possible); disclaimer on every page.
7. tests/test_api.py (TestClient): every endpoint schema, 422 on invalid ranges, unseen category warning, 503 on tampered bundle, parity test (100 test rows, max abs diff < 1e-5), latency checks (/predict p95 < 300 ms, predict+explain < 3 s), proof that no applicant data is written to disk (scan for new files after requests).
8. Manual end-to-end check with Chrome or the built-in browser: fill form, see result, open dashboard, move lambda slider; save 3 screenshots to artifacts/ui/.
9. PHASE GATE as in specs/06 Stage 9 and 10.

Code for a later phase is written in that phase, not earlier (e.g. the API is built in Phase 7 from the frozen bundle). Shared helpers may be
written earlier only if a current phase needs them.

## Notebook per phase
Each phase (1 to 8) also produces its executed .ipynb in notebooks/ (names in docs/CODE_STYLE.md) and lists it in the PHASE GATE report. A phase is not done if its notebook errors or has no saved outputs.

## PHASE GATE report (print at the end of every phase)
1. Phase number and name. 2. Gate items from specs/06 with PASS/FAIL each. 3. pytest summary. 4. Key numbers. 5. Soft flags raised.
6. Files/artifacts written. 7. Time used and projected time for the next phase. 8. Git commit hash. Then append the same to artifacts/RUN_LOG.md.
If any gate item fails: do not advance; fix, or stop and report (hard stop rules below).

## Compute budget (full runs). Rough estimates, unverified; measure the first real epoch and adapt within the caps
- TabTransformer config in specs/09. Batch 2048, mixed precision (bf16) on GPU.
- Phase 2 baselines: 9 trained arms x 3 seeds, max 30 epochs, early stopping patience 4.
- Optuna: 10 trials per tuned arm, each trial = 8 epochs on a 30% stratified subsample of train, identical search-space size for every arm.
- Phase 5 sweep: coarse 27 points x 1 seed; refine the 8 points nearest the Pareto frontier x 3 seeds.
- Hard cap for ALL full runs combined across phases: 10 hours of compute. If the projection after the first measured epoch exceeds the cap,
  reduce in this order and RECORD each reduction in RUN_LOG.md: (1) Optuna trials 10 -> 6, (2) coarse grid 27 -> 18, (3) seeds 3 -> 2 for
  non-JT ablations. Never reduce: A0-A5, S1, S2, JT, leakage before/after, bootstrap CIs.
- Run long jobs with run_in_background, write progress to artifacts/RUN_LOG.md, checkpoint every epoch, make every runner resumable
  (skip completed (arm, seed) pairs).

## Hard stops (stop and report to the user; do not continue past these)
- Any rule in docs/CRITICAL_RULES.md would have to be broken to proceed.
- The test set was accessed before Phase 6 for anything other than the whitelisted `leakage_probe_gender_only` entry (with_target=false) already logged in Phase 1.
- configs/selection_rule.yaml was not committed before the first sweep run.
- NaN loss that persists after clip + grad-clip + LR halved once.
- CUDA unavailable (unless the user passed --allow-cpu).
- A unit test in specs/06 Stage 4 fails and cannot be fixed without weakening it.
- Disk full or the same command fails twice with the same error.

## Soft flags (continue, but write to the TOP of SUMMARY.md and RUN_LOG.md)
- Gender-from-features AUC < 0.60 (premise weak).
- A2 does not show higher recall / worse ECE than A0 (hypothesis not reproduced).
- JT does not beat S2 at matched fairness (report honestly).
- JT leakage AUC on z not lower than A2.
- UCI Taiwan pending or results contradict Home Credit.
- DIR >= 0.8 cannot be met at any sweep point with AUC-ROC >= A2 - 0.02.

## Second dataset
UCI Taiwan (Phase 6): file at data/raw/uci_taiwan/default of credit card clients.xls. If missing, write the loader + config, mark it PENDING in
SUMMARY.md, and continue. Never download from unofficial mirrors; ask the user to place the file.

## Progress and reporting
- Task list: one task per phase item; mark done only when its gate passes.
- Git commit at the end of each phase: `phase<N>: <what>`. Never push.
- No long chat recaps. Phase-end message: the PHASE GATE report, then "ready for Phase N+1".

## Final deliverable (end of Phase 8)
artifacts/SUMMARY.md: results table (arms x metrics, mean +/- CI), Pareto plot path, leakage before/after table, matched-pair stats, UCI Taiwan table,
all soft flags, reductions made to stay in budget, exact commands to reproduce.
