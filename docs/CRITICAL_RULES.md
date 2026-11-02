# Critical Rules (violating any of these invalidates results)

## Data
1. CODE_GENDER is never in X. It is saved as separate aligned array A, used only for: fairness metrics, calibration-by-group,
   leakage probes, adversarial head target, matched-pair grouping. The only exception is the explicit reference arm GI
   (gender included), which must be labelled as such and never deployed.
2. Drop rows with CODE_GENDER == "XNA" (4 rows). SK_ID_CURR and TARGET are never features.
3. DAYS_EMPLOYED == 365243 is a placeholder: set NaN, add flag EMPLOYED_MISSING, then impute.
4. Imputers, scalers, encoders, SMOTE, class weights, prior-correction constants: fit on TRAIN only.
5. Split is stratified jointly on TARGET x CODE_GENDER, seed 42, 70/15/15. Never re-split mid-project.
6. Categorical encoding: integer labels, index 0 reserved for Missing/unseen.

## Evaluation integrity
7. The test set is used once, in stage 8. Hyperparameters, thresholds, lambdas, operating point: validation only.
8. One decision-threshold rule for ALL arms (chosen on val, e.g. fixed flag rate). Different thresholds per arm are forbidden.
9. selection_rule.yaml is committed to git BEFORE the lambda sweep results are viewed. Do not edit it afterwards.
10. Equal tuning budget (same number of Optuna trials, same search-space size) for every tuned arm.
11. 3 seeds minimum per arm. Report mean +/- bootstrap 95% CI. A single-seed fairness difference is not a result.
12. SMOTE-NC is applied inside the training data only, after the split. Never to val/test.
13. Never overwrite results. New timestamped run folder / MLflow run each time.

## Method validity
14. The gender-flip counterfactual is invalid. Use the matched-pair check.
15. "After" leakage is measured on representation z, not on raw X.
16. Per-group Platt/isotonic needs gender at inference. It is an ANALYSIS arm only, labelled non-deployable.
17. UCI Taiwan (second dataset) is a retrain with identical hyperparameters and lambdas. No re-tuning.
18. With all lambdas = 0, the joint loss must equal A2's loss on a fixed batch (unit test). Adversarial term must not be active
    before warm-up ends.

## Claims and wording (paper, report, UI)
19. Never write "novel method", "first", "no prior work", "solves fairness". Allowed: "empirical study", "to our knowledge"
    (only after literature search), "makes the trade-off explicit".
20. Never imply validation on Indian borrowers. Home Credit is not Indian data.
21. Report negative results honestly (e.g. GBT beating TabTransformer, joint not dominating sequential patching).
22. Do not describe Kleinberg as proving equalized-odds vs calibration conflict. Cite Kleinberg et al. 2016 AND Pleiss et al. 2017
    precisely.

## Engineering
23. All settings in configs/*.yaml. No magic numbers in code.
24. Fixed seeds (python, numpy, torch, cuda), logged with every run.
25. Probabilities clipped to [1e-6, 1-1e-6] inside every log/ECE computation.
26. Train and serve share the same preprocessing functions. Parity test must pass (|offline - API| < 1e-5).
27. Do not weaken a test or gate to make it pass. If a gate fails, stop and report the numbers.
28. No secrets or raw data in git. data/ and artifacts/ are git-ignored except manifest.json and selection_rule.yaml.
29. No database in the application and no persistence of applicant data (inputs, scores, gender). The API is stateless; never log applicant values.
30. The frontend must build its form from `/meta/options`; never hard-code category lists or ranges in JS.
31. Notebooks are .ipynb files in notebooks/, one per phase (names in docs/CODE_STYLE.md). They are thin report notebooks: import from src/, no core logic, no copy-pasted training code. Every notebook is executed top to bottom (nbclient/nbconvert --execute) so outputs and charts are saved inside the .ipynb, and re-runs cleanly from a fresh kernel. Never commit a notebook with errors or empty outputs.
32. Never stop, kill or force-close any process, application or service outside this project (for example Ollama, VECTOR, Docker, browsers, chat apps). If memory is short, stop only processes this session started, reduce the workload, and tell the owner what to close.
