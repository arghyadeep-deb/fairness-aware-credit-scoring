# Spec 01: Problem Statement

## Problem
A lender must decide who gets a loan and estimate default risk. Automated models fail in three coupled ways:
1. **Imbalance**: ~8% positives. Naive models ignore defaulters.
2. **Proxy discrimination**: gender is excluded, yet other features reconstruct it (reported gender-from-features AUC ~0.65 on
   other credit data) and the model may still treat groups differently.
3. **Miscalibration**: scores do not equal real default rates, overall or per gender group.

Standard fixes interact: imbalance correction distorts probabilities and group base rates; error-rate fairness constraints
conflict with within-group calibration when base rates differ (Pleiss et al. 2017; Kleinberg et al. 2016).

## Goal
Build and evaluate, on Home Credit Default Risk, a TabTransformer trained with a joint objective
`L_wCE + lambda_f L_fair + lambda_c L_cal + lambda_a L_adv`, and produce an empirical study of the
imbalance x fairness x calibration trade-off, compared against single-fix and sequential-patch baselines.

## Research questions
- RQ1: How much do imbalance corrections (weighted CE, SMOTE-NC, focal) change group-wise calibration and fairness metrics vs plain CE?
- RQ2: Does a joint objective reach a better accuracy-fairness-calibration operating point than sequential patching
  (weighted CE -> prior correction -> fairness post-processing) at matched fairness?
- RQ3: Does the gradient-reversal adversary reduce gender information in the representation z (probe AUC drop), and do new proxies appear (SHAP probe)?
- RQ4: Do conclusions hold when the identical pipeline is retrained on UCI Default of Credit Card Clients (Taiwan)?

## Success definition
Rigorous, reproducible results with CIs for RQ1-RQ4, an honest report of wins and losses, a working demo, and a conference-style paper.
A "win" for JT is not required; a clean empirical answer is.

## Out of scope
New backbone architecture; Indian bureau data; production deployment; causal fairness claims; multiple protected attributes
(age groups are an optional extension).

## Stakeholders
Student team, mentor (evaluation), reviewers of the conference paper.
