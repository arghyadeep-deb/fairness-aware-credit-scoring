# Spec 08: Home Credit Data Spec (exact)

Source file: `data/raw/application_train.csv`. Do not use other Home Credit tables in v1.

## Columns
Identifier (never a feature): `SK_ID_CURR`. Label: `TARGET` (1 = default). Protected attribute (never in X): `CODE_GENDER`.

Numeric (raw): `CNT_CHILDREN, CNT_FAM_MEMBERS, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
REGION_POPULATION_RELATIVE, DAYS_BIRTH, DAYS_EMPLOYED`
Optional numeric (config `include_ext_source`, default true): `EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3`
Categorical: `NAME_FAMILY_STATUS, NAME_EDUCATION_TYPE, NAME_INCOME_TYPE, NAME_HOUSING_TYPE, OCCUPATION_TYPE, ORGANIZATION_TYPE`
Binary (Y/N -> 1/0): `FLAG_OWN_CAR, FLAG_OWN_REALTY`

## Derived
- `AGE_YEARS = -DAYS_BIRTH / 365.25`
- `YEARS_EMPLOYED = -DAYS_EMPLOYED / 365.25` after fixing the 365243 placeholder (set NaN), plus flag `EMPLOYED_MISSING` (1 if placeholder).
- Drop the raw DAYS_BIRTH and DAYS_EMPLOYED after deriving.
- If EXT_SOURCE_k missing: median impute and add `EXT_SOURCE_k_MISSING` indicator.
- Optional engineered ratios (config `add_ratios`, default true): `CREDIT_INCOME_RATIO = AMT_CREDIT/AMT_INCOME_TOTAL`,
  `ANNUITY_INCOME_RATIO = AMT_ANNUITY/AMT_INCOME_TOTAL`, `CREDIT_TERM = AMT_CREDIT/AMT_ANNUITY` (guard divide by zero -> NaN -> impute).

## Cleaning
1. Drop rows with `CODE_GENDER == "XNA"`. Keep F and M only; encode gender array as F=0, M=1 (A).
2. Median imputation for numerics (train medians). "Missing" category (index 0) for categoricals.
3. Categorical label-encoding: index 0 = Missing/unseen; categories seen in train get 1..K. Save mapping.
4. StandardScaler on numeric columns including derived ones, fit on train only. Binary and flag columns are not scaled.

## Split
70/15/15, seed 42, stratified jointly on `TARGET x CODE_GENDER` (4 cells). Same split for all arms, forever.

## Outputs
`data/processed/{train,val,test}.parquet` (features + TARGET, no gender, no ID), `data/processed/{train,val,test}_gender.parquet`
(aligned, includes SK_ID_CURR for audit), `artifacts/prep/{scaler.joblib, encoders.json, cardinalities.json, feature_order.json, class_weights.json, medians.json}`.

## Checks (pandera + asserts, `pytest tests/test_data.py`)
- Gender column not in any feature file or feature_order.
- No NaN in features; no ID overlap between splits; categorical indices < cardinality.
- Default rate and gender share per split within 0.5 percentage points of each other; every TARGET x GENDER cell >= 50 rows per split.
- Train numeric mean ~0 (abs < 0.05), std ~1 (within 0.05).
- GATE report prints: rows per split, default rate per split, % female per split, default rate per gender.

## Leakage "before" (Stage 2)
- LightGBM classifier predicts gender from FEATURES ONLY (never TARGET); fit train, early stopping on val; report ROC-AUC on val and test.
- Logistic regression probe as reference. Shuffled-gender sanity probe must give AUC 0.5 +/- 0.02 (test).
- Top-15 features by gain saved as JSON + PNG. Save `artifacts/leakage/before.json` with config, seed, git hash, timestamp.
- If AUC < 0.60 -> soft flag (see specs/07), continue.

## Second dataset: UCI Default of Credit Card Clients (Taiwan), Phase 6 of specs/07
Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients (CC BY 4.0). File: `data/raw/uci_taiwan/default of credit card clients.xls`
(needs `xlrd` or `openpyxl`). 30,000 rows, no missing values (per UCI). Config: `configs/uci_taiwan.yaml`.
- Read quirk (verify on load): the sheet has a descriptive first row; real column names are on the second row (`header=1`). Target column is
  `default payment next month` (1 = default). Identifier `ID` is never a feature.
- Protected attribute: `SEX` (1 = male, 2 = female). Never in X. Gender array A: F=0, M=1.
- Categorical: `EDUCATION`, `MARRIAGE` (undocumented codes such as 0, 5, 6 -> map to index 0 "Missing/other"; log the counts).
- Numeric: `LIMIT_BAL`, `AGE`, `PAY_0`, `PAY_2..PAY_6`, `BILL_AMT1..6`, `PAY_AMT1..6` (scaled; PAY_* are ordinal repayment status, treat as numeric).
- Same code path as Home Credit: same split rule (70/15/15, seed 42, stratified TARGET x SEX), same scaler/encoding rules, same models,
  same hyperparameters, same lambdas, same threshold rule (flag_rate 0.10) and metric battery. NO re-tuning.
- Extra reported number only for this dataset: metrics also at flag_rate = train default rate (~22%), because 10% is far below its base rate.
- Expect: default ~22%, ~60% female, AUC-ROC ~0.75-0.78; gender-from-features AUC may differ from Home Credit. Report as found.
- Cite: UCI repository and Yeh, I-C., Lien, C-h. (2009), Expert Systems with Applications 36(2).
