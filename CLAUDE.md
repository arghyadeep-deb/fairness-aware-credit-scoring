# Fairness-Aware Credit Scoring (B.Tech major project, KIIT)

Read this file first every session. Details live in docs/ and specs/. Do not duplicate them here.

## One-paragraph summary
Credit default model (TabTransformer) trained with ONE joint loss:
`L = weighted_CE + lambda_f*fairness + lambda_c*calibration + lambda_a*adversarial`
(adversarial = gender head behind a gradient reversal layer). The paper claim is an EMPIRICAL STUDY of how
imbalance correction, fairness intervention and calibration interact in credit scoring, plus a joint
objective that exposes the trade-off. It is NOT a new method. Never write "novel method" or "no prior work".
Data: Home Credit Default Risk (Kaggle, NOT Indian data). Secondary: UCI Default of Credit Card Clients (Taiwan, 30,000 rows, CC BY 4.0, has SEX): retrain, same hyperparameters, no re-tuning.
PAKDD 2010 was DROPPED (no official download, no stated licence): do not look for it.

## Read order for context
1. docs/PROJECT_CONTEXT.md  (why, claims, scope)
2. docs/ARCHITECTURE.md     (pipeline, modules, data flow)
3. docs/CRITICAL_RULES.md   (never violate)
4. docs/CODE_STYLE.md, docs/LIBRARIES.md, docs/COMMANDS.md
5. specs/01..09 (spec-driven development: problem, requirements, API, constraints, edge cases, acceptance, phased build plan, exact data spec, resolved decisions)

## Top 8 rules (full list in docs/CRITICAL_RULES.md)
1. CODE_GENDER is never in the feature matrix X. Separate aligned array A only.
2. Fit scaler/imputer/encoders/SMOTE on TRAIN only.
3. Test set is touched once, in final verification. Never tune on test.
4. One decision-threshold rule for all arms, chosen on validation.
5. Operating-point selection rule is committed to git BEFORE viewing sweep results.
6. Fairness-style "gender flip" is invalid (model never sees gender). Use the matched-pair check.
7. Never overwrite results. Every run gets a new timestamped folder / MLflow run.
8. A stage is done only when its tests pass and its GATE report prints.

## Notebooks
Every phase ends with an executed .ipynb report in notebooks/ (docs/CODE_STYLE.md, rule 31). Logic stays in src/; notebooks only load results and show tables/charts.

## Working method
- Spec-driven: before coding a stage, read its spec in specs/, state the plan in 5 lines, then implement.
- After each stage: run pytest, print the GATE report, summarise in 3-5 lines. No long recaps.
- If a gate fails, stop and report. Do not weaken a gate to make it pass.
- Defaults for every previously vague choice are fixed in specs/09_resolved_decisions.md. Use them; do not re-ask.
- Only stop for the "hard stops" in specs/07_phased_build_plan.md (and at the end of each phase, see below). Soft flags: continue and record them.

## Build mode: 8 phases, executed day by day
Follow specs/07_phased_build_plan.md. Phase N = original Week N (1 data+leakage, 2 baselines, 3 joint loss, 4 stabilise, 5 lambda sweep,
6 verification+second dataset, 7 freeze+API+frontend, 8 figures+summary). Each phase: smoke check, full run, PHASE GATE report, git commit,
RUN_LOG.md entry, then STOP and wait for "next phase" (unless the user says "run all phases"). Exact data columns: specs/08. Gates: specs/06.
Defaults: specs/09. Ignore any "Week N" wording in older documents (it means Phase N, not a calendar week).
On every new session first read artifacts/RUN_LOG.md and `git log` to find the current phase and resume.

## Current status
Fresh start. Machine setup (.venv, PyTorch cu128, preflight tools) and datasets in data/raw/ are already in place. No project code, configs,
artifacts, processed data or git repo exist yet. artifacts/RUN_LOG.md does not exist yet: create it in Phase 1. Begin at Phase 0 (verify setup, git init), then Phase 1.
