"""Repeated-splits replication: split isolation and guard-log discipline."""

import json
from pathlib import Path

import numpy as np

from src.data.prep import split_indices

ROOT = Path(__file__).resolve().parents[1]


def test_new_seed_changes_rows_keeps_fractions_and_partitions():
    y = np.random.default_rng(0).integers(0, 2, 4000)
    a = np.random.default_rng(1).integers(0, 2, 4000)
    fr = {"train": 0.70, "val": 0.15, "test": 0.15}
    s42, s43 = split_indices(y, a, fr, 42), split_indices(y, a, fr, 43)
    assert set(s42["test"]) != set(s43["test"])
    assert abs(len(s43["test"]) - 600) <= 2
    assert sorted(np.concatenate(list(s43.values())).tolist()) == list(range(4000))


def test_repeated_guard_log_has_one_entry_per_arm_per_split():
    log = ROOT / "artifacts/guard/repeated_splits.jsonl"
    if not log.exists():
        return
    purposes = [json.loads(x)["purpose"] for x in log.read_text().splitlines()]
    assert all(p.startswith("repeated:split") for p in purposes)
    assert len(purposes) == len(set(purposes))


def test_original_guard_log_has_no_repeated_entries():
    text = (ROOT / "artifacts/guard/test_access.jsonl").read_text()
    assert "repeated:" not in text
