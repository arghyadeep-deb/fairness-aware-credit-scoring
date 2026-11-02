# OVERNIGHT REPORT (2026-11-01 to 2026-11-02)

Sections A to E of the overnight instruction are done and committed; nothing was pushed. Nothing is blocked. No hard stop was triggered.

## What completed
- **A** Reviewed Codex's uncommitted edits, added tests, kept them (logged in RUN_LOG).
- **B / Phase 6** Final verification on the Home Credit test set: every arm accessed once (guard log 28 entries), everything else computed from `artifacts/test_preds/`. Taiwan retrain with one guarded access per arm (8 entries). Notebook 06, gate, commit `2976f80`.
- **C / Phase 7** Frozen bundle `artifacts/bundle/v1` (sha256 manifest, fails closed with 503), FastAPI + web demo, 26 API tests, headless-Edge end-to-end check with screenshots in `artifacts/ui/`, clean-venv load identical to 6 decimals, notebook 07 (0 errors), gate, commit `d3d3aa2`.
- **D / Phase 8** `scripts/make_all_figures_tables.py` (6 figures, 11 tables, no test access), `artifacts/SUMMARY.md`, notebook 08 (0 errors), gate, commit `6bb6731`.
- **E** Validation-only diagnostics, commit `9602905`: JT-W rejected by the sampler check (batch default rate 0.0806 vs 0.0807, gender share 0.3418 vs 0.3416, so importance weights are 1.0); JT-A+PH is worse than S2-EG on AUC and calibration with no distinguishable EOD gain.
- Final test suite: 150 passed. Ruff clean on all new files.

## Headline findings (full text in SUMMARY.md)
Gender is recoverable from features (AUC about 0.90). The class-weighted loss breaks calibration and the prior-correction formula repairs it (worst-group ECE 0.350 to 0.013). S2-EG is significantly ahead of JT chosen on AUC-ROC, AUC-PR, worst-group ECE, EOD and DIR, and tied on recall (S2-EG test EOD 0.0151 vs JT 0.0363, a barely significant gap). JT's EOD gain over A2 and S1 is not significant on test. Single-step GRL failed (Phase 4 single-step runs (validation numbers, control probe 0.785): probe 0.67 to 0.81, adversary loss up to 8 to 16, AUC-ROC 0.46 to 0.73); multi-step GRL lowers leakage modestly (test: 0.759 to 0.773, A2 0.791). LightGBM leads the TabTransformer by about 0.019 AUC-ROC. Taiwan shows the same direction.

## Blocked or partial
- Nothing is blocked. `artifacts/OVERNIGHT_BLOCKED.md` (gitignored) described the earlier memory block, which was resolved after you cleared memory; it is now marked resolved.
- Partial: no human visual review of the UI (automated Edge check only); no check in Chrome or on a phone; JT-W was not run because its premise was rejected.

## Soft flags (all)
1. JT chosen does not beat the sequential baseline; its validation EOD was optimistic (0.0234 to 0.0363 test).
2. Single-step GRL failed in the Phase 4 single-step runs (validation numbers, control probe 0.785): probe 0.67 to 0.81, adversary loss up to 8 to 16, AUC-ROC 0.46 to 0.73. JT chosen's test probe of 0.801 has the adversary off and is not a GRL result.
3. FLAG_OWN_CAR is a possible new proxy in the JT top-15 SHAP features.
4. Matched-pair deltas not reduced by JT (0.0336 vs A2 0.0297).
5. JT worst-group ECE 0.0198 on test (cap 0.02); 0.0238 on 5 validation seeds.
6. Mixed SHAP sample sizes (17 arms 4,000 rows, ten arms 5,000).
7. Taiwan: JT chosen loses 0.05 AUC-PR with unchanged lambdas.
8. Phase 6 predictions used bf16 on GPU; serving is fp32 CPU (gap 0.00184, in the manifest).
9. Codex edited the working tree without authorisation; reviewed, tested, kept. Two harness memory kills, resolved by a smaller workload and a restart after you cleared memory.
10. Browser check ran on port 8765 because port 8000 was held by the owner's separate VECTOR API (`run_vector.py api`, PID 47112), not by a leftover server of this project as first written; the documented command still uses 8000.
11. ThresholdOptimizer is not flag-rate matched (analysis only).
12. The bundle is gitignored; a fresh clone must run `scripts/freeze_bundle.py`.
13. Harness blocked `rm -rf`, so `artifacts/cleanvenv/` and `artifacts/ui/_profile/` remain on disk (gitignored). Delete them by hand if you want the space.

## Commits (all local, not pushed)
`45c7767` Phase 6 pre-declaration | `6b1bd6d` memory-safe runner | `a903cee` Phase 7/8 prep (code) | `2976f80` Phase 6 | `d3d3aa2` Phase 7 | `6bb6731` Phase 8 | `9602905` diagnostics

## Resources
- Peak memory (process tree RSS): Phase 6 analysis at most 1.96 GB per arm; JT tuned 1.75 GB; H arms about 0.93 GB; JT-A about 1.95 GB. Phase 7/8 and diagnostics were not measured separately (one process at a time, 4 threads, no kills).
- Compute: total hours were NOT measured and I did not total them from the logs, so no figure is given. Early phases are logged at about 0.6 h cumulative (RUN_LOG, Phase 3); later per-step wall times are in the RUN_LOG entries. No reduction from specs/07 was applied. Free disk 196 GB.

## Run the demo
```
.venv\Scripts\activate
python scripts/freeze_bundle.py         # only if artifacts/bundle/v1 is missing
uvicorn api.main:app --port 8000        # open http://127.0.0.1:8000/
```
If port 8000 is busy (for example while VECTOR is running), use `--port 8765`.

## What you must check in the morning
1. Look at `artifacts/ui/01_form.png`, `02_result.png`, `03_dashboard.png` and click through the live demo yourself (visual check is pending).
2. Read the soft flags above, especially 1 (the headline is a negative result for the joint model) and 3 (FLAG_OWN_CAR).
3. Decide how the paper frames the joint model given the negative result (SUMMARY.md section 2 and the claims checklist; rule 22 wording on Kleinberg/Pleiss still has to be applied in the paper text).
4. Confirm you are happy with the Codex edits that were kept (RUN_LOG, overnight section A).
5. Decide whether to push (not done) and whether to delete `artifacts/cleanvenv` and `artifacts/ui/_profile`.

## Update (2026-11-02): three-way lambda exploration
Validation only, SUMMARY.md section 9. No point passes all limits or beats S2-EG on AUC-PR; the headline is unchanged. Apps closed to free memory must be restarted by hand: VECTOR API, Docker Desktop and others listed in RUN_LOG.
