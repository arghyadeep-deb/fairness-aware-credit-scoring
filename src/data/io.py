"""Split loaders with a test-set access log (specs/05: guard for the test split)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.utils import resolve

# Purposes that never touch TARGET and are mandated by specs/08 (leakage probe predicts gender).
TEST_GENDER_ONLY_PURPOSES = {"leakage_probe_gender_only"}


def _log_test_access(cfg: dict[str, Any], purpose: str, with_target: bool) -> None:
    path = resolve(cfg["paths"]["guard_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "with_target": with_target,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_split(
    cfg: dict[str, Any], split: str, with_target: bool = True, purpose: str = "", processed=None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X[, with TARGET column], gender array A) for a split.

    The test split needs ALLOW_TEST=1, or a gender-only purpose that never loads TARGET.
    Every test access is appended to the guard log so Phase 6 can audit it.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split}")
    if split == "test":
        allowed = os.environ.get("ALLOW_TEST") == "1" or (
            purpose in TEST_GENDER_ONLY_PURPOSES and not with_target
        )
        if not allowed:
            raise PermissionError("Test split is locked: set ALLOW_TEST=1 (stage 8 only)")
        if processed is None:  # smoke runs use a separate dir and must not pollute the real log
            _log_test_access(cfg, purpose or "ALLOW_TEST", with_target)
    proc = processed or resolve(cfg["paths"]["processed"])
    path = proc / f"{split}.parquet"
    cols = None
    if not with_target:  # read only feature columns so TARGET is never loaded
        cols = [c for c in pq.read_schema(path).names if c != cfg["target_column"]]
    x = pd.read_parquet(path, columns=cols)
    a = pd.read_parquet(proc / f"{split}_gender.parquet")["gender"]
    return x, a
