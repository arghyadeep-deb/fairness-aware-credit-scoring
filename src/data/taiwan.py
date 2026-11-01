"""UCI Default of Credit Card Clients (Taiwan, 30,000 rows, CC BY 4.0): second dataset (Phase 6).

Follows specs/08: `header=1`; SEX is the gender array (a = 1 for male (SEX 1), 0 for female (SEX 2),
like CODE_GENDER on Home Credit) and is NEVER in X; ID is never a feature. Stratified 70/15/15 split on
default x SEX (seed 42). Scaler fitted on TRAIN only. Numeric block: LIMIT_BAL, AGE, PAY_0..PAY_6
(ordinal, kept numeric), BILL_AMT1..6, PAY_AMT1..6 (amounts: signed log1p, then standardised).
Categorical: EDUCATION (valid 1-4; 0/5/6 -> index 0), MARRIAGE (valid 1-3; 0 -> index 0); index 0 is
the Missing/other/unseen slot like on Home Credit. The test split is only reachable through
`access_test`, which writes a guard-log line per call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.prep import class_weights
from src.data.tensors import Arrays

RAW = "data/raw/uci_taiwan/default of credit card clients.xls"
TARGET = "default payment next month"
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
AMT = ["LIMIT_BAL"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
NUM = AMT + ["AGE"] + PAY
CAT = ["EDUCATION", "MARRIAGE"]
VALID = {"EDUCATION": [1, 2, 3, 4], "MARRIAGE": [1, 2, 3]}
SPLIT = (0.70, 0.15, 0.15)


def load_raw(root: Path) -> pd.DataFrame:
    df = pd.read_excel(root / RAW, header=1)
    assert len(df) == 30000 and TARGET in df.columns and "SEX" in df.columns
    return df


def _cats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out, counts = pd.DataFrame(index=df.index), {}
    for c in CAT:
        odd = ~df[c].isin(VALID[c])
        counts[c] = {str(k): int(v) for k, v in df.loc[odd, c].value_counts().items()}
        out[c] = df[c].where(~odd, 0).astype(int)
    return out, counts


def _build(root: Path, seed: int = 42):
    df = load_raw(root)
    a = (df["SEX"] == 1).astype(int).to_numpy()
    y = df[TARGET].to_numpy()
    strata = y * 2 + a
    ix = np.arange(len(df))
    tr, rest = train_test_split(ix, train_size=SPLIT[0], stratify=strata, random_state=seed)
    va, te = train_test_split(
        rest, train_size=SPLIT[1] / (SPLIT[1] + SPLIT[2]), stratify=strata[rest], random_state=seed
    )
    x = df[NUM].astype("float64").copy()
    for c in AMT:
        x[c] = np.sign(x[c]) * np.log1p(np.abs(x[c]))
    sc = StandardScaler().fit(x.iloc[tr])  # train only
    xs = sc.transform(x).astype("float32")
    cat_df, odd_counts = _cats(df)
    cat = cat_df[CAT].to_numpy().astype("int64")
    cards = [int(cat[tr, j].max()) + 2 for j in range(cat.shape[1])]  # + unseen/other slot 0 + room
    arrays = {
        name: Arrays(xs[i], cat[i], y[i], a[i], list(NUM), list(CAT), [])
        for name, i in (("train", tr), ("val", va), ("test", te))
    }
    return arrays, cards, class_weights(y[tr]), odd_counts


def prepare(root: Path, seed: int = 42):
    """({'train','val': Arrays}, cardinalities, class_weights). The test split is NOT returned."""
    arrays, cards, weights, _ = _build(root, seed)
    arrays.pop("test")
    return arrays, cards, weights


def odd_code_counts(root: Path) -> dict:
    return _build(root)[3]


def access_test(root: Path, purpose: str, log_path: Path, git: str = "") -> Arrays:
    """THE guarded access to the Taiwan test split: one guard-log line per call."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"utc": datetime.now(timezone.utc).isoformat(), "purpose": purpose, "git": git}
            )
            + "\n"
        )
    return _build(root)[0]["test"]
