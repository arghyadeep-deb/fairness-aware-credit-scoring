# Spec 04: Constraints

## Technical
- Python 3.11; PyTorch >= 2.7 with CUDA 12.8 (RTX 5070 Ti). Must also run inference on CPU for the demo.
- Training offline only, on the dev machine. The API never trains.
- Serving: any machine with Python 3.10+, no OS-specific dependencies (NFR-06). No internet at inference.
- Memory: processed data fits in RAM (~307K rows x ~25 features). Use float32.
- Latency: predict + explain within 3 s (NFR-01). Use SHAP with 50-100 row background or GradientExplainer; time it in stage 4.
- Reproducibility: fixed seeds, pinned requirements (pip freeze at freeze stage), config hash in manifest.

## Methodological
- Single protected attribute (gender, binary after dropping XNA) in v1. Age groups optional extension.
- Fairness definition in the loss: equalized-odds surrogate (main), demographic-parity surrogate (ablation). Selection rule keeps DIR >= 0.8 (NFR-03).
- Calibration is evaluated per gender group; per-group post-hoc calibration is analysis only (needs gender at inference).
- Same preprocessing for all arms. Same splits. Same threshold rule. Same metric battery.
- Equal Optuna budget per tuned arm (default 30 trials, from config).
- 3 seeds per arm, bootstrap 1,000 resamples on test for CIs.

## Application architecture decisions
- No database in the application (no MySQL, PostgreSQL, SQLite-for-app, SQLAlchemy). File-based frozen bundle only; stateless API. MLflow's SQLite file is training-time only.
- Frontend: plain HTML/CSS/JS, Chart.js via CDN, no framework, no build step, no Node required.
- Backend and frontend run as one process (`uvicorn api.main:app`).

## Data / legal / ethical
- Home Credit is not Indian data; no claim of Indian validity.
- Gender never used as a predictor in deployed model. Health, hobbies, religion, caste, ethnicity, name, address, phone excluded.
- No personal data in the demo beyond what is needed (NFR-07). Demo uses synthetic or test-set rows.
- Licences: Kaggle competition rules for Home Credit (research use); UCI Taiwan is CC BY 4.0 (cite UCI repository + Yeh & Lien 2009). PAKDD is not used. Do not redistribute raw data.

## Schedule
8 phases executed day by day, see specs/07_phased_build_plan.md (smoke check first in every phase, then background full runs, hard cap 10 h of
total compute). If the budget is tight, cut order: optional age attribute -> extra dataset -> GI reference -> G1 -> fewer seeds for ablations.
Never cut: A0-A5, S1, S2, JT, leakage before/after, bootstrap CIs.

## Known expected outcomes (do not treat as bugs)
- TabTransformer likely below LightGBM on AUC.
- Without EXT_SOURCE features AUC-ROC is ~0.63-0.68; with them ~0.74-0.77. Config flag `include_ext_source` (default true).
- Gender-from-features AUC may be only ~0.60-0.70. If < 0.60, record a soft flag at the top of SUMMARY.md (premise weak, discuss with the mentor) and continue.
