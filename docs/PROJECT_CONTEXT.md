# Project Context

## Who and what
- Owner: Arghyadeep Deb (B.Tech CSE AI/ML, KIIT). Team: Swastika Khatua, Shivangi Saha, Shristi Saha, Simran Sharma. Mentor: Dr. Samaresh Mishra.
- Deliverables: (1) trained model + experiments, (2) FastAPI backend + HTML/CSS/JS demo, (3) conference-style paper (workshop / student / applied-AI conference), (4) project report + SRS (already written).
- Dev machine: Lenovo Legion 5, RTX 5070 Ti. Needs PyTorch >= 2.7 with CUDA 12.8 (cu128).
- Build mode: 8 phases executed day by day with Claude Code, starting 2026-10-31 (specs/07_phased_build_plan.md). Paper writing is a separate, later task.

## The three problems
1. Class imbalance: ~8% defaults. Predicting "no default" for everyone gives 92% accuracy and is useless.
2. Proxy leakage: removing gender does not remove its effect; other features re-encode it.
3. Calibration: a "30% risk" score must mean ~30% observed default, per gender group, not only overall.

## The twist
Each standard fix hurts another axis: weighted loss / SMOTE inflate probabilities (calibration); resampling shifts group base
rates (fairness metrics); Pleiss et al. (NeurIPS 2017) show calibration is compatible with only one error-rate constraint
(equal FNR), and Kleinberg et al. (2016) show the inherent conflict when base rates differ.

## What the project claims (and does not)
CLAIMS (defensible):
- An empirical, multi-seed study, with confidence intervals, of how imbalance fixes, fairness interventions and calibration
  interact in credit scoring on Home Credit (+ UCI Taiwan retrain).
- A single differentiable joint objective with tunable lambdas that makes the trade-off surface visible (Pareto frontier).
- A proxy-leakage probe on the learned representation z and on SHAP vectors, before vs after.
- A matched-pair check for individual-level gender sensitivity.

DOES NOT CLAIM:
- A new technique. GRL / adversarial debiasing (Ganin 2015; Zhang 2018; Edwards & Storkey 2016), calibration losses (MMCE),
  fair+calibrated training ("Towards Fair and Calibrated Models", 2023), TabTransformer (Huang 2020) all exist.
- That "no prior work" measures the interaction. Unverified. Only write "to our knowledge" AFTER a proper literature search.
- Validation on Indian borrowers. Home Credit is not Indian. India (RBI ECL, Ind AS 109, from 1 April 2027) is motivation only.
- That TabTransformer beats gradient-boosted trees. It probably will not. Report honestly; the objective transfers to any backbone.

## Known prior work to position against (read before writing the paper)
- Pleiss et al. 2017, On Fairness and Calibration (NeurIPS).
- Kleinberg, Mullainathan, Raghavan 2016, Inherent trade-offs in fair determination of risk scores.
- Liu, Simchowitz, Hardt 2018, Group calibration is a byproduct of unconstrained learning.
- "Towards Fair and Calibrated Models" (arXiv 2310.10399): train-time group-wise calibration losses, Pareto analysis.
- Hartomo et al. 2025 (IEEE Access): weighted-loss TabTransformer + SHAP (Reference Paper 1).
- Nwafor & Nwafor 2023 (Finance Research Letters): ensembles + 1D CNN, fairness by unawareness (Reference Paper 2).
- Structural Gender Bias in Credit Scoring: Proxy Leakage (arXiv 2601.18342): gender AUC ~0.65 from non-sensitive features (Taiwan data).
- Subgroup Threshold Optimization (arXiv 2403.10652): on Home Credit, excluding gender raises false rejection of creditworthy women 1-6%.
- Resampling and calibration: Dal Pozzolo et al.; Piccininni et al. 2024; Andersen et al. 2026 (clinical): resampling degrades
  calibration, gains reproducible by threshold adjustment.

## Why this matters for baselines
Imbalance damage to calibration is cheap to repair (analytic prior correction, isotonic). So the joint model must beat
"weighted CE -> prior correction -> fairness method", not only plain weighted CE. This sequential pipeline is a REQUIRED baseline.

## Datasets
- Primary: Home Credit Default Risk, application_train.csv (~307,511 x 122), TARGET (1=default), protected attribute CODE_GENDER.
- Secondary: UCI "Default of Credit Card Clients" (Taiwan, Yeh & Lien 2009), 30,000 rows, 23 features, ~22% default, SEX column (1=male, 2=female),
  licence CC BY 4.0, file `data/raw/uci_taiwan/default of credit card clients.xls` (already downloaded). Different features, so it is a RETRAIN of the same pipeline
  with the same hyperparameters and lambdas (no re-tuning), not a reuse of frozen weights. Cite the UCI repository and the original paper.
- Dropped: PAKDD 2010/2009 Brazilian data. No official download (organiser server unreachable), no stated licence, gender column unconfirmed. Do not use mirrors.
- Rejected: German Credit (1,000 rows, saturated), unverified "Indian CIBIL" Kaggle sets (no provenance).
