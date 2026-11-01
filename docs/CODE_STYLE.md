# Code Style

- Python 3.11. Type hints on all public functions. Docstrings state WHY, not what.
- Small functions (< ~40 lines), one responsibility. No global mutable state.
- Formatting: black (line length 100), ruff for lint, isort via ruff. Run before every commit.
- Tracking: MLflow via SQLite URI from configs/base.yaml (never the file: backend).
- Config over constants: read from configs/*.yaml via one `load_config(path)` helper; dataclasses or pydantic models for configs.
- Randomness: a single `seed_everything(seed)` in src/utils.py (python, numpy, torch, cuda, cudnn deterministic flags).
- Paths: pathlib.Path only. No hard-coded absolute paths. Root derived from the repo.
- Logging: `logging` module, not print, except the final GATE reports which are printed in a fixed tabular format.
- Tensors: always state shape in a comment at function boundaries, e.g. `# (B, F)`.
- PyTorch: models are nn.Module with a pure `forward`; losses are nn.Module or functions taking (logits/probs, y, a); no data
  loading inside models. Move to device in the trainer only.
- Metrics: pure numpy/sklearn functions that take arrays (y_true, y_prob, group) and return floats/dicts. Fully unit-tested.
- Errors: raise specific exceptions (ValueError with a clear message) for bad inputs; never silently coerce data.
- Tests: pytest, one test file per module group. Deterministic (fixed seeds), fast (< 30 s each, small synthetic data where possible).
- Git: small commits, message format `stage<N>: <what>`. Never commit data/, artifacts/ (except manifest.json), .venv/.
- Naming: snake_case functions/vars, PascalCase classes, UPPER_CASE constants only for column names in configs.
- Comments: explain non-obvious decisions (why clip probabilities, why warm-up). No commented-out code.
- Output to the user after each task: 3-5 lines on what changed and the key numbers. No long recaps.

## Notebooks (.ipynb)
- Location: notebooks/. Files: 01_data_and_leakage_before.ipynb, 02_baselines.ipynb, 03_joint_model.ipynb, 04_lambda_sweep_and_selection.ipynb, 05_posthoc_comparison.ipynb, 06_final_verification_and_taiwan.ipynb, 07_serving_demo_check.ipynb, 08_figures_and_summary.ipynb (number = phase; Phase 0 needs none).
- Each notebook: title cell, goal in 2 lines, imports from src/, load results from artifacts/ or data/processed/, tables and charts, a short "what this shows" markdown cell. No heavy training inside a notebook; training runs from scripts, notebooks only load and display results (small smoke demos allowed).
- Create with nbformat, execute with `python -m nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`. Kernel: the project .venv (register with `python -m ipykernel install --user --name fairness-credit`).
- Fixed seeds; no absolute paths; no gender values or raw applicant rows printed beyond aggregates.
