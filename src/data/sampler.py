"""Stratified batch sampler: every batch has >= `min_per_cell` rows of each TARGET x gender cell."""

from __future__ import annotations

import torch


def stratified_epoch(
    y: torch.Tensor, a: torch.Tensor, batch_size: int, min_per_cell: int, gen: torch.Generator
) -> torch.Tensor:
    """Index matrix (steps, batch_size) for one epoch (steps = n // batch_size).

    Per-cell quota = max(min_per_cell, natural share of the batch); the largest cell is trimmed so
    the batch has exactly `batch_size` rows. Rows of a cell are drawn without replacement and the
    cell is reshuffled when exhausted, so the smallest cell can be seen slightly more than once
    per epoch only when its floor exceeds its natural share (it barely does on Home Credit).
    """
    dev, n = y.device, len(y)
    cell = (y.long() * 2 + a.long()).to(dev)
    steps = n // batch_size
    ids = [torch.nonzero(cell == c).squeeze(1) for c in range(4)]
    if any(len(i) == 0 for i in ids):
        raise ValueError("a TARGET x gender cell is empty; cannot stratify")
    share = torch.tensor([len(i) / n for i in ids])
    quota = torch.clamp((share * batch_size).round().long(), min=min_per_cell)
    quota[int(torch.argmax(quota))] -= int(quota.sum()) - batch_size  # keep the total exact
    cols = []
    for c in range(4):
        need = steps * int(quota[c])
        reps = -(-need // len(ids[c]))
        stream = torch.cat(
            [
                ids[c][torch.randperm(len(ids[c]), generator=gen, device="cpu").to(dev)]
                for _ in range(reps)
            ]
        )[:need]
        cols.append(stream.view(steps, int(quota[c])))
    batches = torch.cat(cols, dim=1)
    return batches
