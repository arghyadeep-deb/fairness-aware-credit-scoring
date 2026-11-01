"""TabTransformer, 1D CNN and MLP backbones (specs/09). Every model returns (logit, z).

z is the last hidden layer (the representation probed for leakage in Phase 6). Models never
load data or move themselves to a device; the trainer does that.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _mlp(in_dim: int, hidden: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    return nn.Sequential(*layers)


class _Embeddings(nn.Module):
    """One embedding table per categorical column; index 0 is Missing/unseen."""

    def __init__(self, cardinalities: list[int], emb_dim: int) -> None:
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(c, emb_dim) for c in cardinalities)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:  # (B, C) -> (B, C, d)
        return torch.stack([t(x_cat[:, i]) for i, t in enumerate(self.tables)], dim=1)


class TabTransformer(nn.Module):
    def __init__(
        self,
        n_cont: int,
        cardinalities: list[int],
        emb_dim: int = 32,
        heads: int = 4,
        layers: int = 3,
        attn_dropout: float = 0.1,
        ffn_mult: int = 2,
        mlp: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mlp = mlp or [128, 64]
        n_cat = len(cardinalities)
        self.emb = _Embeddings(cardinalities, emb_dim)
        self.col = nn.Parameter(torch.randn(1, n_cat, emb_dim) * 0.02)  # column identity
        layer = nn.TransformerEncoderLayer(
            emb_dim,
            heads,
            emb_dim * ffn_mult,
            dropout=attn_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.cont_norm = nn.LayerNorm(n_cont)
        self.mlp = _mlp(n_cont + n_cat * emb_dim, mlp, dropout)
        self.z_dim = mlp[-1]
        self.head = nn.Linear(self.z_dim, 1)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
        ctx = self.encoder(self.emb(x_cat) + self.col)  # (B, C, d) contextual embeddings
        h = torch.cat([self.cont_norm(x_cont), ctx.flatten(1)], dim=1)  # (B, F + C*d)
        z = self.mlp(h)  # (B, z_dim)
        return self.head(z).squeeze(-1), z  # (B,), (B, z_dim)


class CNN1D(nn.Module):
    def __init__(
        self,
        n_cont: int,
        cardinalities: list[int],
        emb_dim: int = 8,
        channels: list[int] | None = None,
        kernel: int = 3,
        mlp: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels, mlp = channels or [32, 64], mlp or [64]
        self.emb = _Embeddings(cardinalities, emb_dim)
        convs: list[nn.Module] = []
        c_in = 1
        for c in channels:
            convs += [nn.Conv1d(c_in, c, kernel, padding=kernel // 2), nn.ReLU()]
            c_in = c
        self.convs = nn.Sequential(*convs)
        self.mlp = _mlp(channels[-1], mlp, dropout)
        self.z_dim = mlp[-1]
        self.head = nn.Linear(self.z_dim, 1)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
        seq = torch.cat([x_cont, self.emb(x_cat).flatten(1)], dim=1).unsqueeze(1)  # (B, 1, L)
        z = self.mlp(self.convs(seq).amax(dim=2))  # global max pool -> (B, z_dim)
        return self.head(z).squeeze(-1), z


class MLP(nn.Module):
    def __init__(
        self,
        n_cont: int,
        cardinalities: list[int],
        emb_dim: int = 8,
        hidden: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = hidden or [256, 128, 64]
        self.emb = _Embeddings(cardinalities, emb_dim)
        self.mlp = _mlp(n_cont + len(cardinalities) * emb_dim, hidden, dropout)
        self.z_dim = hidden[-1]
        self.head = nn.Linear(self.z_dim, 1)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
        z = self.mlp(torch.cat([x_cont, self.emb(x_cat).flatten(1)], dim=1))
        return self.head(z).squeeze(-1), z


def build_model(
    name: str, n_cont: int, cardinalities: list[int], defaults: dict[str, Any], hp: dict[str, Any]
) -> nn.Module:
    """Build a backbone from config defaults overridden by tuned hyperparameters."""
    kw = dict(defaults[name])
    if "emb_dim" in hp and name == "tabtransformer":
        kw["emb_dim"] = int(hp["emb_dim"])
    if "dropout" in hp:
        kw["dropout"] = float(hp["dropout"])
    cls = {"tabtransformer": TabTransformer, "cnn1d": CNN1D, "mlp": MLP}[name]
    return cls(n_cont, cardinalities, **kw)
