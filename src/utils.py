"""Shared helpers: config loading, seeding, repo paths, git hash, MLflow URI."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and merge its `base_config` (if any) underneath it.

    Why: stage configs stay small while paths/seeds live in one place.
    """
    with open(resolve(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("base_config")
    if base:
        merged = load_config(base)
        merged.update({k: v for k, v in cfg.items() if k != "base_config"})
        return merged
    return cfg


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (if installed) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_hash() -> str:
    """Current commit hash, or 'no-git' before the first commit."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def mlflow_tracking_uri(cfg: dict[str, Any]) -> str:
    """Absolute SQLite URI built from the repo-relative one in the config."""
    uri = cfg["mlflow"]["tracking_uri"]
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        raise ValueError(f"MLflow URI must be sqlite:///..., got {uri}")
    db = resolve(uri[len(prefix) :])
    db.parent.mkdir(parents=True, exist_ok=True)
    return prefix + db.as_posix()


def mlflow_experiment(cfg: dict[str, Any], name: str) -> str:
    """Point MLflow at the SQLite store and return the experiment id (created if needed)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri(cfg))
    root = resolve(cfg["mlflow"]["artifact_root"])
    root.mkdir(parents=True, exist_ok=True)
    exp = MlflowClient().get_experiment_by_name(name)
    return (
        exp.experiment_id
        if exp
        else MlflowClient().create_experiment(name, artifact_location=root.as_uri())
    )
