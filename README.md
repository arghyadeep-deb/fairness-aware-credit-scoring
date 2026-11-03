# Fairness-Aware Credit Scoring

An **empirical study** of how imbalance correction, a fairness intervention and calibration interact in credit-default prediction, with a joint training objective that makes the trade-off explicit, and a small web demo of the served model. B.Tech major project, KIIT.

> Research prototype. The models are trained on Home Credit Default Risk (Kaggle), which is **not Indian data**, and must not be used for real credit decisions. This is not a new method; it compares known techniques under one protocol.

## What it does

A TabTransformer is trained with one joint loss:

```
L = weighted_CE + lambda_f * fairness + lambda_c * calibration + lambda_a * adversarial
```

- **weighted_CE** corrects for class imbalance (about 8% of applicants default).
- **fairness** is a soft equalized-odds penalty between gender groups.
- **calibration** pulls predicted probabilities toward observed default rates.
- **adversarial** is a gender-prediction head behind a gradient-reversal layer.

Gender is **never a model input**. It is held in a separate array and used only by the fairness terms, the adversary and the evaluation.

The joint model is compared with sequential pipelines (class weights, then prior correction, then a gender-blind fairness step), gender-aware references and LightGBM, all under one decision rule: flag the 10% riskiest applicants using a threshold chosen on the validation set.

## Main results (held-out test set, Home Credit)

Mean of 3 seeds. Higher is better for AUC-ROC, AUC-PR and DIR; lower is better for worst-group ECE and EOD.

| Method | AUC-ROC | AUC-PR | Worst-group ECE | EOD | DIR |
|---|---:|---:|---:|---:|---:|
| A2: class-weighted TabTransformer | .7463 | .2209 | .3496 | .0601 | .9511 |
| S1: A2 + prior correction | .7463 | .2209 | .0131 | .0601 | .9511 |
| **S2-EG: S1 + gender-blind fairness step** | .7463 | .2209 | .0131 | .0151 | .9893 |
| **JT chosen: joint loss (0.05, 2, 0), served in the demo** | .7433 | .2145 | .0198 | .0363 | .9818 |
| REF: LightGBM | .7661 | .2513 | .0085 | .0849 | .9423 |
| REF-EG: LightGBM + fairness step (post-hoc) | .7661 | .2513 | .0085 | .0168 | .9805 |

**Findings**

- The sequential pipeline **matched or beat the joint model**. S2-EG is significantly ahead of JT chosen on AUC-ROC, AUC-PR, EOD and DIR.
- Prior correction alone repairs calibration (worst-group ECE .35 to .013) without changing ranking.
- Gender stays recoverable from the model's internal representation even though gender is never an input: a probe predicts it with AUC 0.79 (A2) and 0.80 (JT chosen, adversary off). Turning the gradient-reversal adversary on lowered this only to about 0.76.
- LightGBM ranks applicants better than every neural arm; adding the same fairness step brings its EOD down to S2-EG's level.
- Validation fairness was optimistic: JT chosen's EOD rose from .0234 on validation to .0363 on test.

Worst-group ECE differences are reported as descriptive only, and some arms were added after the test results were known (labelled post-hoc). Both points, and every other known deviation, are listed in [REPRODUCE.md](REPRODUCE.md).

A secondary check retrains on the UCI Taiwan credit-card dataset with the same settings.

## Demo

A FastAPI backend serves the frozen model and a three-page frontend.

```bash
pip install -r requirements.txt
uvicorn api.main:app --port 8000
# open http://127.0.0.1:8000/
```

- **Applicant:** enter an application (or load a synthetic example) and get a default probability and risk band.
- **Result:** the features that pushed the score up or down (SHAP), and a comparison with the most similar applicant of the other gender.
- **Dashboard:** test-set fairness, calibration by gender, the proxy audit, a lambda trade-off explorer, and the method comparison.

The model bundle in `artifacts/bundle/v1/` is checked against a sha256 manifest at start-up; the API refuses to serve if any file has changed. For hosting on Render, the start command is `uvicorn api.main:app --host 0.0.0.0 --port $PORT`; a keep-alive ping (`api/keepalive.py`) stops the free tier from sleeping.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Data preparation, models, losses, training, evaluation |
| `scripts/` | Phase runners, reports, figure and notebook builders |
| `configs/` | Data, training, arm and selection-rule configuration |
| `api/`, `web/` | Serving backend and frontend |
| `artifacts/bundle/v1/` | Frozen served model with manifest |
| `artifacts/tables/`, `artifacts/figures/` | Result tables and figures |
| `notebooks/` | Executed report notebooks, one per phase (01 to 08) |
| `tests/` | Test suite (`pytest -q`) |
| `reproduce/` | Resolved hyperparameters and settings |

## Reproducing

See **[REPRODUCE.md](REPRODUCE.md)** for data download, the pinned environment, the protocol, the run order and all known deviations and corrections.

## Data

- [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle; `application_train.csv` only). Not redistributed here beyond the model bundle's held-out reference pool.
- [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Taiwan, CC BY 4.0).
