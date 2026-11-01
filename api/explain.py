"""SHAP (GradientExplainer) contributions for one applicant, 100-row bundle background."""

from __future__ import annotations

import math

import numpy as np
import torch

from api.bundle import Bundle
from api.schemas import ApplicantRecord
from api.service import record_to_raw, to_arrays
from src.data.prep import derive


class _FromEmb(torch.nn.Module):
    """TabTransformer with the categorical embeddings as an explicit input (as in src.eval.final)."""

    def __init__(self, bb) -> None:
        super().__init__()
        self.bb = bb

    def forward(self, cont, emb_flat):
        bb = self.bb
        emb = emb_flat.view(emb_flat.shape[0], len(bb.emb.tables), -1)
        ctx = bb.encoder(emb + bb.col)
        z = bb.mlp(torch.cat([bb.cont_norm(cont), ctx.flatten(1)], dim=1))
        return bb.head(z)


def _display(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def explain(b: Bundle, rec: ApplicantRecord, top_k: int = 10, nsamples: int = 100) -> dict:
    import shap

    bb = b.model
    f = _FromEmb(bb).eval()
    n_cat = len(bb.emb.tables)

    def emb_of(xk):
        with torch.no_grad():
            return bb.emb(xk).flatten(1).float()

    bc = torch.from_numpy(b.bg_cont.astype("float32"))
    be = emb_of(torch.from_numpy(b.bg_cat.astype("int64")))
    d = be.shape[1] // n_cat
    cont, cat = to_arrays(b, rec)
    c, e = torch.from_numpy(cont), emb_of(torch.from_numpy(cat))
    torch.manual_seed(0)  # deterministic: same input -> same output
    np.random.seed(0)
    ex = shap.GradientExplainer(f, [bc, be])
    sv = ex.shap_values([c, e], nsamples=nsamples)
    sv = [np.asarray(s, dtype="float64").reshape(1, -1) for s in sv]
    vec = np.concatenate([sv[0], sv[1].reshape(1, n_cat, d).sum(2)], axis=1)[0]
    names = b.spec["cont_names"] + b.spec["cat_names"]
    derived = derive(record_to_raw(rec), b.pre.cfg).iloc[0]
    items = [
        {"feature": n, "value": _display(derived[n]), "shap": float(vec[i])}
        for i, n in enumerate(names)
    ]
    items.sort(key=lambda t: -abs(t["shap"]))
    with torch.no_grad():
        base = float(f(bc, be).mean())
    return {
        "base_value": base,
        "contributions": items[:top_k],
        "top_k": top_k,
        "model_version": b.version,
    }
