# Commands (run from repo root, venv activated)

## Setup
```
python -m venv .venv
.venv\Scripts\activate                      # Windows   (Mac/Linux: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python src/env_check.py                      # must print CUDA True + GPU name
```

## Pipeline stages
```
python -m src.data.prep --config configs/data.yaml            # stage 1
python -m src.eval.leakage --config configs/data.yaml --stage before   # stage 2
python -m src.train --config configs/arm_a0.yaml --seed 0      # stage 3, one arm/one seed
python -m src.run_arms --arms a0 a1 a2 a3 a4 a5 s1 s2 g1 --seeds 0 1 2   # stage 3, all
python -m src.sweep --config configs/sweep.yaml                 # stage 5
python -m src.select --rule configs/selection_rule.yaml         # stage 6
python -m src.verify --config configs/verify.yaml               # stage 8 (touches test set ONCE)
python -m src.freeze --run artifacts/runs/<timestamp>           # stage 9
```

## Serving
```
uvicorn api.main:app --reload --port 8000
# one command serves API + frontend: open http://127.0.0.1:8000/  (web/ is mounted by the app, no separate static server)
```

## Quality
```
pytest -q                       # all tests
pytest tests/test_data.py -q    # one module
ruff check . && black --check .
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

## Git
```
git add -A && git commit -m "stage1: data prep + checks"
```

## Hosting on Render (keep-alive)
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Render's free tier sleeps a web service after 15 minutes without traffic. `api/keepalive.py` pings `<public url>/health` every 10 minutes so it stays awake.
- On Render it turns on by itself (Render sets `RENDER_EXTERNAL_URL`). Elsewhere set `KEEPALIVE_URL=https://your-host`. Change the interval with `KEEPALIVE_SECONDS` (minimum 60). Turn it off with `KEEPALIVE_DISABLED=1`. Locally it stays off.
