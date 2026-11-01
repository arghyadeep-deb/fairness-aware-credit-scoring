"""Synthetic checks for bounded analysis; never load a real held-out split."""

import numpy as np
import pytest
import torch

from src.data.tensors import Arrays
from src.eval.bootstrap import _chunk, bootstrap_metrics, cell_index
from src.eval.final import logits_and_z, shap_vectors
from src.models.nets import TabTransformer


def test_bootstrap_chunks_preserve_shared_replicates():
    rng = np.random.default_rng(42)
    y, a = np.tile([0, 0, 1, 1], 100), np.tile([0, 1, 0, 1], 100)
    p = rng.uniform(0.01, 0.99, len(y))
    flags = p >= 0.9
    expected = _chunk(y, p, a, flags, cell_index(y, a), 13, range(105), 15)
    actual = bootstrap_metrics(y, p, a, flags, 105, 13, n_jobs=4, chunk_size=100)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_bootstrap_rejects_invalid_chunk():
    a = np.tile([0, 1], 4)
    with pytest.raises(ValueError, match="positive"):
        bootstrap_metrics(a, a / 2, a, a > 0, 10, 0, chunk_size=0)


def test_shap_and_logits_smoke_float32_in_small_batches():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rows = Arrays(
        rng.normal(size=(12, 3)).astype(np.float32),
        rng.integers(0, 3, size=(12, 2)),
        np.tile([0, 1], 6),
        np.tile([0, 1], 6),
        ["a", "b", "c"],
        ["d", "e"],
        [],
    )
    model = TabTransformer(3, [3, 3], emb_dim=4, heads=2, layers=1, mlp=[4], dropout=0).eval()
    device = torch.device("cpu")
    logits, z = logits_and_z(model, rows, device, batch=3)
    assert logits.shape == (12,) and z.shape == (12, 4)
    values, method = shap_vectors(
        model, rows.subset(np.arange(4)), rows, device, nsamples=3, batch_size=5
    )
    assert values.shape == (12, 5) and values.dtype == np.float32
    assert np.isfinite(values).all() and method
