# RUN_LOG

Soft flags (top of file, newest first): none raised so far.

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
