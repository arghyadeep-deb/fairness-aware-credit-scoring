"""Phase 6 helpers: model loading for test-time scoring, z-probe on test, SHAP vectors,
matched-pair check. Nothing here reads the test split itself; callers pass arrays in.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data.tensors import Arrays
from src.eval.leakage import fit_lgbm_probe
from src.models.heads import JointModel
from src.models.nets import build_model


def load_backbone(name, arr_like: Arrays, cards, cfg, hp, state, device):
    m = build_model(name, arr_like.cont.shape[1], cards, cfg["model_defaults"], hp)
    m.load_state_dict(state)
    return m.to(device).eval()


def load_joint(arr_like: Arrays, cards, cfg, hp, state, device):
    bb = build_model("tabtransformer", arr_like.cont.shape[1], cards, cfg["model_defaults"], hp)
    m = JointModel(bb, cfg["joint"]["adversary_hidden"]).to(device)
    m.load_state_dict(state)
    return m.eval()


@torch.no_grad()
def logits_and_z(model, arr: Arrays, device, batch: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """(logit, z) for a backbone or a JointModel (Head B ignored)."""
    xc, xk = torch.from_numpy(arr.cont), torch.from_numpy(arr.cat)
    lg, zs = [], []
    for i in range(0, len(xc), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(xc[i : i + batch].to(device), xk[i : i + batch].to(device))
        lg.append(out[0].float().cpu())
        zs.append(out[1].float().cpu())
    return torch.cat(lg).numpy(), torch.cat(zs).numpy()


def z_probe_test(cfg, ztr, atr, zva, ava, zte, ate, seed: int) -> dict[str, float]:
    """Phase 1 'before' protocol on z: LightGBM trained on train, early-stopped on val AUC,
    scored on TEST (plus val); logistic reference. Gender labels only, TARGET never used."""
    cols = [f"z{i}" for i in range(ztr.shape[1])]
    df = lambda z: pd.DataFrame(np.asarray(z, dtype=np.float32), columns=cols)  # noqa: E731
    gbm = fit_lgbm_probe(df(ztr), atr, df(zva), ava, [], cfg["leakage"]["lightgbm"], seed)
    ztr = np.asarray(ztr, dtype=np.float32)
    sc = StandardScaler().fit(ztr)
    lr = LogisticRegression(max_iter=300, n_jobs=4).fit(sc.transform(ztr), atr)
    return {
        "probe_lgbm_test_auc": float(roc_auc_score(ate, gbm.predict_proba(df(zte))[:, 1])),
        "probe_lgbm_val_auc": float(roc_auc_score(ava, gbm.predict_proba(df(zva))[:, 1])),
        "probe_logistic_test_auc": float(roc_auc_score(ate, lr.predict_proba(sc.transform(zte))[:, 1])),
        "lgbm_best_iteration": int(gbm.best_iteration_ or 0),
    }  # fmt: skip


class _FromEmb(torch.nn.Module):
    """TabTransformer with the categorical embeddings as an explicit input (for attributions)."""

    def __init__(self, bb) -> None:
        super().__init__()
        self.bb = bb

    def forward(self, cont, emb_flat):
        bb = self.bb
        emb = emb_flat.view(emb_flat.shape[0], len(bb.emb.tables), -1)
        ctx = bb.encoder(emb + bb.col)
        z = bb.mlp(torch.cat([bb.cont_norm(cont), ctx.flatten(1)], dim=1))
        return bb.head(z)


def shap_vectors(
    model, bg: Arrays, rows: Arrays, device, nsamples: int = 100, batch_size: int = 500
) -> tuple[np.ndarray, str]:
    """Per-row attribution vectors (n_cont + n_cat) of the default logit.

    shap.GradientExplainer on the model with the categorical embedding vectors as inputs; the
    attribution of a categorical column is the sum over its embedding dimensions. Falls back to
    gradient x input on the same inputs if GradientExplainer fails (the method used is returned).
    """
    bb = model.backbone if hasattr(model, "backbone") else model
    f = _FromEmb(bb).to(device).eval()

    def emb_of(arr):
        xk = torch.from_numpy(arr.cat).to(device)
        with torch.no_grad():
            return bb.emb(xk).flatten(1).float()

    bc, be = torch.from_numpy(np.asarray(bg.cont, dtype=np.float32)).to(device), emb_of(bg)
    n_cat = len(bb.emb.tables)
    d = be.shape[1] // n_cat
    out, method = [], "shap.GradientExplainer"
    try:
        import shap

        ex = shap.GradientExplainer(f, [bc, be])
        for i in range(0, len(rows.cont), batch_size):
            c = torch.from_numpy(np.asarray(rows.cont[i : i + batch_size], dtype=np.float32)).to(
                device
            )
            e = emb_of(_slice(rows, i, i + batch_size))
            sv = ex.shap_values([c, e], nsamples=nsamples)
            sv = [np.asarray(s, dtype=np.float32).reshape(len(c), -1) for s in sv]
            out.append(np.concatenate([sv[0], sv[1].reshape(len(c), n_cat, d).sum(2)], axis=1))
            del c, e, sv
            gc.collect()
            torch.cuda.empty_cache()
    except (MemoryError, torch.cuda.OutOfMemoryError):
        raise  # The sequential runner records the failure and retries the declared smaller sample.
    except Exception:  # pragma: no cover - fallback path
        method = "gradient x input (GradientExplainer unavailable)"
        out = []
        for i in range(0, len(rows.cont), batch_size):
            c = torch.from_numpy(rows.cont[i : i + batch_size]).to(device).requires_grad_(True)
            e = emb_of(_slice(rows, i, i + batch_size)).requires_grad_(True)
            dc, de = torch.autograd.grad(f(c, e).sum(), (c, e))
            gc_, ge = dc * (c - bc.mean(0)), de * (e - be.mean(0))
            out.append(
                torch.cat([gc_, ge.view(len(c), n_cat, d).sum(2)], dim=1).detach().cpu().numpy()
            )
            del c, e, dc, de, gc_, ge
            gc.collect()
            torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32, copy=False), method


def _slice(arr: Arrays, i: int, j: int) -> Arrays:
    return arr.subset(np.arange(i, min(j, len(arr.y))))


def shap_probe(cfg, sv_tr, a_tr, sv_te, a_te, seed: int) -> float:
    """LightGBM gender probe on SHAP vectors: train on train rows, early-stop on 20% of them,
    AUC on the test rows."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sv_tr))
    k = int(0.8 * len(idx))
    cols = [f"s{i}" for i in range(sv_tr.shape[1])]
    df = lambda z: pd.DataFrame(np.asarray(z, dtype=np.float32), columns=cols)  # noqa: E731
    g = fit_lgbm_probe(
        df(sv_tr[idx[:k]]), a_tr[idx[:k]], df(sv_tr[idx[k:]]), a_tr[idx[k:]], [],
        cfg["leakage"]["lightgbm"], seed,
    )  # fmt: skip
    return float(roc_auc_score(a_te, g.predict_proba(df(sv_te))[:, 1]))


def matched_pairs(test: Arrays, n_query: int = 5000, k_cand: int = 50, seed: int = 0):
    """Nearest applicant of the OTHER gender in the same split (k=1).

    distance = Euclidean on the scaled numeric+binary block + 0.5 per mismatching categorical.
    50 numeric nearest candidates are re-ranked with the categorical term. Returns
    (query index, matched index).
    """
    rng = np.random.default_rng(seed)
    q = np.sort(rng.choice(len(test.y), n_query, replace=False))
    match = np.zeros(len(q), dtype=int)
    for g in (0, 1):
        qi = np.flatnonzero(test.a[q] == g)
        cand = np.flatnonzero(test.a != g)
        nn = NearestNeighbors(n_neighbors=k_cand).fit(test.cont[cand])
        dist, nbr = nn.kneighbors(test.cont[q[qi]])
        for r, row in enumerate(qi):
            c = cand[nbr[r]]
            mism = (test.cat[c] != test.cat[q[row]]).sum(1)
            match[row] = c[np.argmin(dist[r] + 0.5 * mism)]
    return q, match


def delta_stats(p: np.ndarray, q: np.ndarray, m: np.ndarray) -> dict[str, float]:
    d = np.abs(p[q] - p[m])
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
    }


def new_proxy_check(
    sv_a: np.ndarray, sv_b: np.ndarray, names: list[str], gender_auc: dict[str, float],
    top: int = 15, proxy_auc: float = 0.60,
) -> dict[str, Any]:  # fmt: skip
    """Features in the top-`top` by mean |SHAP| for arm B (e.g. JT) but not for arm A (A2),
    with their single-feature gender AUC (>= proxy_auc counts as a proxy)."""
    ra = np.argsort(-np.abs(sv_a).mean(0))[:top]
    rb = np.argsort(-np.abs(sv_b).mean(0))[:top]
    new = [names[i] for i in rb if i not in set(ra)]
    return {
        "top_A": [names[i] for i in ra],
        "top_B": [names[i] for i in rb],
        "new_in_B": new,
        "new_proxy_flags": {n: gender_auc.get(n, float("nan")) >= proxy_auc for n in new},
        "new_gender_auc": {n: gender_auc.get(n, float("nan")) for n in new},
    }
