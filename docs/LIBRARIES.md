# Preferred Libraries (use these; ask before adding anything else)

| Purpose | Library | Notes |
|---|---|---|
| Deep learning | PyTorch >= 2.7 (cu128) | RTX 50-series needs CUDA 12.8 build |
| TabTransformer | custom nn.Module (optionally lucidrains `tab-transformer-pytorch` as reference) | custom keeps loss/GRL hooks simple |
| Data | pandas, numpy, pyarrow | parquet for processed data |
| Classical ML | scikit-learn | scaler, splits, logistic probe, metrics, isotonic/Platt |
| Trees | lightgbm (leakage probe, reference ceiling), xgboost optional | |
| Resampling | imbalanced-learn (SMOTENC) | train split only |
| Fairness | fairlearn (MetricFrame, ThresholdOptimizer, ExponentiatedGradient) | for S2 baseline + metric cross-checks |
| Explainability | shap | background 50-100 rows to meet latency |
| Tuning | optuna | same trial budget per arm |
| Tracking | mlflow 3.x with SQLite backend (`sqlite:///artifacts/mlflow.db`, see specs/09) | every run; file: backend is disabled in MLflow 3 |
| Validation | pandera, pydantic | dataframe schemas; API schemas |
| Config | PyYAML | |
| Serving | fastapi, uvicorn, httpx (tests) | |
| Frontend | vanilla HTML/CSS/JS, Chart.js via CDN | no framework |
| Testing | pytest | |
| Plotting | matplotlib | reliability diagrams, Pareto plots, importances |
| Style | black, ruff | |

App rule: no database drivers or ORMs in api/ (no SQLAlchemy, psycopg, mysql, sqlite3 for app data). Frontend: no npm/Node build, no React/Vue.

Not allowed without approval: TensorFlow/Keras, large frontend frameworks, any paid API, any library needing internet at inference.

Notebooks: jupyterlab, ipykernel, nbformat, nbconvert, nbclient (allowed, in requirements.txt).
