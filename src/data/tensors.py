"""Model-ready arrays from the processed parquet (train/val only; test is locked until Phase 6)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.io import load_split
from src.data.prep import data_dirs


@dataclass
class Arrays:
    cont: np.ndarray  # (N, F) float32: scaled numerics + binary flags (+ gender for GI only)
    cat: np.ndarray  # (N, C) int64 label-encoded categoricals, 0 = Missing/unseen
    y: np.ndarray  # (N,) int
    a: np.ndarray  # (N,) int gender array, used for fairness metrics / G1 loss, never in cont
    cont_names: list[str]
    cat_names: list[str]
    binary_names: list[str]

    def subset(self, ix: np.ndarray) -> "Arrays":
        return Arrays(
            self.cont[ix],
            self.cat[ix],
            self.y[ix],
            self.a[ix],
            self.cont_names,
            self.cat_names,
            self.binary_names,
        )


def load_arrays(
    cfg: dict[str, Any], split: str, smoke: bool = False, include_gender: bool = False
) -> tuple[Arrays, list[int]]:
    """Load a split and the categorical cardinalities. `include_gender` is for arm GI only."""
    if split == "test":
        raise PermissionError("Phase 2 never reads the test split")
    proc, prep = data_dirs(cfg, smoke)
    order = json.loads((prep / "feature_order.json").read_text())
    card = json.loads((prep / "cardinalities.json").read_text())
    x, a = load_split(cfg, split, with_target=True, processed=proc)
    cont_names = order["numeric"] + order["binary"]
    cat_names = order["categorical"]
    cont = x[cont_names].to_numpy(dtype="float32")
    if include_gender:  # reference arm only; clearly labelled non-deployable
        cont = np.concatenate([cont, a.to_numpy(dtype="float32")[:, None]], axis=1)
        cont_names = cont_names + ["GENDER_REFERENCE_ONLY"]
    arr = Arrays(
        cont,
        x[cat_names].to_numpy(dtype="int64"),
        x[cfg["target_column"]].to_numpy(),
        a.to_numpy(),
        cont_names,
        cat_names,
        list(order["binary"]),
    )
    return arr, [card[c] for c in cat_names]


def stratified_subsample(arr: Arrays, fraction: float, seed: int) -> Arrays:
    """Subsample stratified on TARGET x gender (used for Optuna trials)."""
    strata = arr.y * 2 + arr.a
    ix, _ = train_test_split(
        np.arange(len(arr.y)), train_size=fraction, stratify=strata, random_state=seed
    )
    return arr.subset(np.sort(ix))


def smote_nc(arr: Arrays, ratio: float, k: int, seed: int) -> Arrays:
    """SMOTE-NC on TRAIN only (CRITICAL_RULES 12). Binary flags are re-rounded to {0,1}."""
    from imblearn.over_sampling import SMOTENC

    n_cont = arr.cont.shape[1]
    X = np.concatenate([arr.cont, arr.cat.astype("float32")], axis=1)
    cat_idx = list(range(n_cont, X.shape[1]))
    Xr, yr = SMOTENC(
        categorical_features=cat_idx, sampling_strategy=ratio, k_neighbors=k, random_state=seed
    ).fit_resample(X, arr.y)
    cont, cat = Xr[:, :n_cont].astype("float32"), Xr[:, n_cont:].astype("int64")
    for i, name in enumerate(arr.cont_names):
        if name in arr.binary_names:
            cont[:, i] = np.round(cont[:, i])
    # synthetic rows have no real gender (-1); only G1 uses gender in the loss and it has no SMOTE
    a = np.concatenate([arr.a, np.full(len(yr) - len(arr.a), -1)])
    return Arrays(cont, cat, yr, a, arr.cont_names, arr.cat_names, arr.binary_names)
