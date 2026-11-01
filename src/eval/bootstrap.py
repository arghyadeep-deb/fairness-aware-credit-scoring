"""Stratified bootstrap (TARGET x gender cells) with indices shared across arms and seeds.

Sharing replicate indices makes arm comparisons paired and the work parallel-safe: replicate b
is drawn from default_rng([seed, b]) regardless of how the replicates are chunked.
"""

from __future__ import annotations

import gc

import numpy as np
from joblib import Parallel, delayed

from src.eval.metrics import METRIC_NAMES, compute_metrics


def cell_index(y: np.ndarray, a: np.ndarray) -> list[np.ndarray]:
    """Row indices of the four TARGET x gender cells."""
    return [np.flatnonzero((y == t) & (a == g)) for t in (0, 1) for g in (0, 1)]


def resample(cells: list[np.ndarray], seed: int, b: int) -> np.ndarray:
    """One stratified resample: each cell is redrawn with replacement at its own size."""
    rng = np.random.default_rng([seed, b])
    return np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])


def _chunk(y, p, a, flags, cells, seed, bs, ece_bins):
    rows = []
    for b in bs:
        ix = resample(cells, seed, b)
        m = compute_metrics(y[ix], p[ix], a[ix], flags[ix], ece_bins)
        rows.append([m[k] for k in METRIC_NAMES])
    return np.array(rows)


def bootstrap_metrics(
    y: np.ndarray,
    p: np.ndarray,
    a: np.ndarray,
    flags: np.ndarray,
    n_boot: int,
    seed: int,
    ece_bins: int = 15,
    n_jobs: int = -1,
    chunk_size: int = 100,
) -> np.ndarray:
    """Array (n_boot, len(METRIC_NAMES)) of metrics on stratified resamples."""
    cells = cell_index(y, a)
    if chunk_size < 1 or n_boot < 1:
        raise ValueError("Bootstrap count and chunk_size must be positive")
    jobs = min(4, max(1, n_jobs if n_jobs > 0 else 4))
    result = np.empty((n_boot, len(METRIC_NAMES)), dtype=np.float64)
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        chunks = np.array_split(np.arange(start, stop), min(jobs, stop - start))
        # Threads share the input arrays instead of copying them into four processes.
        parts = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(_chunk)(y, p, a, flags, cells, seed, bs, ece_bins) for bs in chunks
        )
        result[start:stop] = np.vstack(parts)
        del parts, chunks
        gc.collect()
    return result


def pooled_ci(per_seed: list[np.ndarray], level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """CI of the across-seed mean: average the paired replicates over seeds, then percentiles."""
    mean_per_b = np.nanmean(np.stack(per_seed), axis=0)  # (B, n_metrics)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return np.nanpercentile(mean_per_b, lo, axis=0), np.nanpercentile(mean_per_b, hi, axis=0)
