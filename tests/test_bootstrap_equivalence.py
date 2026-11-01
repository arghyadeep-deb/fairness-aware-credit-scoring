"""The chunked/threaded bootstrap must equal a plain single-process reference exactly,
including the paired case (replicate b uses rng([seed, b]) for every arm)."""

import numpy as np
import pytest

from src.eval.bootstrap import bootstrap_metrics, cell_index, resample
from src.eval.metrics import METRIC_NAMES, compute_metrics


def _data(n=1500, seed=3):
    r = np.random.default_rng(seed)
    a = r.integers(0, 2, n)
    y = (r.random(n) < 0.15 + 0.05 * a).astype(int)
    p = np.clip(0.1 + 0.5 * y * r.random(n) + 0.2 * r.random(n), 1e-4, 1 - 1e-4)
    flags = p >= np.quantile(p, 0.9)
    return y, p, a, flags


def _reference(y, p, a, flags, n_boot, seed, bins=15):
    cells = cell_index(y, a)
    rows = []
    for b in range(n_boot):
        ix = resample(cells, seed, b)
        m = compute_metrics(y[ix], p[ix], a[ix], flags[ix], bins)
        rows.append([m[k] for k in METRIC_NAMES])
    return np.array(rows)


@pytest.mark.parametrize("chunk_size,n_jobs", [(100, 4), (37, 4), (250, 1), (1, 2)])
def test_chunked_threaded_matches_reference(chunk_size, n_jobs):
    y, p, a, flags = _data()
    ref = _reference(y, p, a, flags, 250, 0)
    got = bootstrap_metrics(y, p, a, flags, 250, 0, 15, n_jobs=n_jobs, chunk_size=chunk_size)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref, equal_nan=True)


def test_paired_differences_identical():
    y, p, a, flags = _data()
    p2 = np.clip(p + 0.02 * np.random.default_rng(9).normal(size=len(p)), 1e-4, 1 - 1e-4)
    f2 = p2 >= np.quantile(p2, 0.9)
    ref = _reference(y, p, a, flags, 120, 0) - _reference(y, p2, a, f2, 120, 0)
    got = bootstrap_metrics(y, p, a, flags, 120, 0, 15, n_jobs=4, chunk_size=50) - (
        bootstrap_metrics(y, p2, a, f2, 120, 0, 15, n_jobs=3, chunk_size=33)
    )
    assert np.array_equal(got, ref, equal_nan=True)


def test_replicate_indices_independent_of_chunking():
    y, _, a, _ = _data()
    cells = cell_index(y, a)
    assert np.array_equal(resample(cells, 0, 7), resample(cells, 0, 7))
    assert not np.array_equal(resample(cells, 0, 7), resample(cells, 0, 8))
