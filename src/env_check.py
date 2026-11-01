"""Stage 0 environment check: GPU, libraries, MLflow (SQLite) round-trip, both datasets."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, mlflow_tracking_uri, resolve  # noqa: E402

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))


def check_torch(cfg: dict) -> None:
    import torch

    major_minor = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    need = tuple(int(x) for x in cfg["env"]["min_torch"].split("."))
    record("torch version", major_minor >= need, torch.__version__)
    cuda = torch.cuda.is_available()
    record("CUDA available", cuda, str(cuda))
    if not cuda:
        return
    record("GPU", True, torch.cuda.get_device_name(0))
    a = torch.randn(4096, 4096, device="cuda")
    t0 = time.time()
    b = a @ a
    torch.cuda.synchronize()
    record("GPU matmul 4096x4096", bool(torch.isfinite(b).all()), f"{time.time() - t0:.3f}s")


def check_libs() -> None:
    import lightgbm
    import sklearn

    record("lightgbm", True, lightgbm.__version__)
    record("scikit-learn", True, sklearn.__version__)


def check_mlflow(cfg: dict) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = mlflow_tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    art_root = resolve(cfg["mlflow"]["artifact_root"])
    art_root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    exp = client.get_experiment_by_name("env_check")
    exp_id = (
        exp.experiment_id
        if exp
        else client.create_experiment("env_check", artifact_location=art_root.as_uri())
    )
    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_param("probe", "env_check")
        mlflow.log_metric("value", 1.0)
        run_id = run.info.run_id
    back = client.get_run(run_id)
    ok = back.data.metrics.get("value") == 1.0 and back.data.params.get("probe") == "env_check"
    record("MLflow SQLite round-trip", ok, cfg["mlflow"]["tracking_uri"])


def check_data(cfg: dict) -> None:
    import pandas as pd

    ids = pd.read_csv(resolve(cfg["paths"]["raw_csv"]), usecols=["SK_ID_CURR"])
    record("Home Credit CSV", len(ids) == 307511, f"{len(ids)} rows")
    uci = pd.read_excel(resolve(cfg["paths"]["uci_xls"]), header=1)
    ok = len(uci) == 30000 and "default payment next month" in uci.columns and "SEX" in uci.columns
    record("UCI Taiwan XLS", ok, f"{len(uci)} rows x {uci.shape[1]} cols")


def main() -> int:
    cfg = load_config("configs/base.yaml")
    steps = [(check_torch, (cfg,)), (check_libs, ()), (check_mlflow, (cfg,)), (check_data, (cfg,))]
    for fn, args in steps:
        try:
            fn(*args)
        except Exception as exc:  # report every failure instead of stopping at the first
            record(fn.__name__, False, repr(exc)[:200])
    print(f"{'CHECK':28} {'RESULT':6} DETAIL")
    for name, ok, detail in results:
        print(f"{name:28} {'PASS' if ok else 'FAIL':6} {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print("STAGE 0 GATE:", "PASS" if not failed else f"FAIL ({', '.join(failed)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
