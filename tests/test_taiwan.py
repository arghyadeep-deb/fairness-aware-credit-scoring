"""Taiwan prep follows specs/08; hyper-parameters and lambdas are exactly the Home Credit ones."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import taiwan
from src.run_joint import tag

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    return taiwan._build(ROOT)


def test_header_row_and_columns():
    raw = pd.read_excel(ROOT / taiwan.RAW, header=1)
    assert taiwan.TARGET in raw.columns and "ID" in raw.columns and len(raw) == 30000
    assert set(raw["SEX"].unique()) == {1, 2}
    assert "ID" not in taiwan.NUM + taiwan.CAT and "SEX" not in taiwan.NUM + taiwan.CAT


def test_split_70_15_15_stratified_seed42(built):
    arr = built[0]
    n = {k: len(v.y) for k, v in arr.items()}
    assert n == {"train": 21000, "val": 4500, "test": 4500}
    strata = {k: (v.y * 2 + v.a) for k, v in arr.items()}
    for c in range(4):
        shares = [float(np.mean(strata[k] == c)) for k in arr]
        assert max(shares) - min(shares) < 0.005
    again = taiwan._build(ROOT)[0]
    assert np.array_equal(arr["test"].y, again["test"].y)  # deterministic seed


def test_gender_not_in_x_and_encoding(built):
    arr = built[0]
    raw = pd.read_excel(ROOT / taiwan.RAW, header=1)
    male_share = float((raw["SEX"] == 1).mean())
    tr = arr["train"]
    assert abs(tr.a.mean() - male_share) < 0.01  # a=1 male, 0 female
    assert not any("SEX" in n for n in tr.cont_names + tr.cat_names)
    assert tr.cont.shape[1] == len(taiwan.NUM)


def test_odd_codes_to_index_zero_and_counts_logged(built):
    arr, _, _, odd = built
    raw = pd.read_excel(ROOT / taiwan.RAW, header=1)
    n_odd_edu = int((~raw["EDUCATION"].isin([1, 2, 3, 4])).sum())
    n_odd_mar = int((~raw["MARRIAGE"].isin([1, 2, 3])).sum())
    assert (
        sum(odd["EDUCATION"].values()) == n_odd_edu and sum(odd["MARRIAGE"].values()) == n_odd_mar
    )
    total0 = sum(int((v.cat[:, 0] == 0).sum()) for v in arr.values())
    assert total0 == n_odd_edu
    assert all(int(v.cat.max()) < 5 + 1 for v in arr.values())


def test_scaler_fit_on_train_only(built):
    arr = built[0]
    tr, va = arr["train"].cont, arr["val"].cont
    assert np.abs(tr.mean(0)).max() < 1e-4 and np.abs(tr.std(0) - 1).max() < 1e-3
    assert np.abs(va.mean(0)).max() > 1e-4  # val is transformed with train statistics, not its own


def test_test_split_only_through_guarded_access(tmp_path):
    arr, _, _ = taiwan.prepare(ROOT)
    assert set(arr) == {"train", "val"}
    log = tmp_path / "g.jsonl"
    te = taiwan.access_test(ROOT, "unit", log, "x")
    assert len(te.y) == 4500 and len(log.read_text().strip().splitlines()) == 1


def test_same_hyperparameters_and_lambdas_as_home_credit():
    spec = importlib.util.spec_from_file_location("p6tw", ROOT / "scripts" / "phase6_taiwan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.JT_LAMBDAS == (0.05, 2.0, 0.0)
    # Home Credit JT chosen lives in the f0p05_c2_a0 folder and uses the A2 hyper-parameters
    assert tag(mod.JT_LAMBDAS) == "f0p05_c2_a0"
    assert mod.HP_SOURCES == {
        "A0": "a0/tuning.json",
        "A2": "a2/tuning.json",
        "JT chosen": "a2/tuning.json",
    }
    p6test = (ROOT / "scripts" / "phase6_test.py").read_text()
    assert '"hp": "a2"' in p6test and "f0p05_c2_a0" in p6test
    src = (ROOT / "scripts" / "phase6_taiwan.py").read_text()
    assert not re.search(r"optuna|tune_|suggest_|study\.optimize", src)  # no re-tuning code path
    assert mod.EG_EPS == 0.01
