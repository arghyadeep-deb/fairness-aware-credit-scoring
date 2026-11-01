"""UCI Default of Credit Card Clients (Taiwan, 30,000 rows, CC BY 4.0): second dataset (Phase 6).

SEX is the gender array (a = 1 for male, 0 for female, like CODE_GENDER on Home Credit) and is NEVER
in X. Stratified 60/20/20 split on default x SEX (seed 42). Scaler fitted on TRAIN only.
Numeric block: LIMIT_BAL, AGE, PAY_0..PAY_6 (ordinal, kept numeric), BILL_AMT1..6, PAY_AMT1..6
(amounts: signed log1p, then standardised). Categorical: EDUCATION (0/5/6 -> 'other'), MARRIAGE
(0 -> 'other'); index 0 is reserved for Missing/unseen like on Home Credit. No binary block.
"""

from __future__ import annotations

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


def load_raw(root: Path) -> pd.DataFrame:
    df = pd.read_excel(root / RAW, header=1)
    assert len(df) == 30000 and TARGET in df.columns
    return df


def _cats(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3, 4]), 5).astype(int)
    out["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2, 3]), 4).astype(int)
    return out


def prepare(root: Path, seed: int = 42):
    """Returns ({'train','val','test': Arrays}, cardinalities, class_weights)."""
    df = load_raw(root)
    a = (df["SEX"] == 1).astype(int).to_numpy()
    y = df[TARGET].to_numpy()
    strata = y * 2 + a
    ix = np.arange(len(df))
    tr, rest = train_test_split(ix, train_size=0.6, stratify=strata, random_state=seed)
    va, te = train_test_split(rest, train_size=0.5, stratify=strata[rest], random_state=seed)
    x = df[NUM].astype("float64").copy()
    for c in AMT:
        x[c] = np.sign(x[c]) * np.log1p(np.abs(x[c]))
    sc = StandardScaler().fit(x.iloc[tr])
    xs = sc.transform(x).astype("float32")
    cat = _cats(df)[CAT].to_numpy().astype("int64")
    cards = [int(cat[tr, j].max()) + 2 for j in range(cat.shape[1])]  # + unseen 0 + room
    arrays = {
        name: Arrays(xs[i], cat[i], y[i], a[i], list(NUM), list(CAT), [])
        for name, i in (("train", tr), ("val", va), ("test", te))
    }
    return arrays, cards, class_weights(y[tr])
