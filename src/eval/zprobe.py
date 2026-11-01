"""Fresh gender probe on the frozen representation z (same protocol as the Phase 1 "before" probe).

LightGBM with the Phase 1 settings (early stopping on validation AUC) plus the logistic
reference, trained on TRAIN z and scored on VALIDATION z. The adversary head's own AUC is NOT
used: a head that is being fooled can score below 0.5 while gender is still recoverable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.tensors import Arrays
from src.eval.leakage import fit_lgbm_probe
from src.models.heads import JointModel
from src.models.nets import build_model
from src.train_joint import predict_joint


def probe_z(
    cfg: dict[str, Any],
    hp: dict[str, Any],
    state: dict,
    train: Arrays,
    val: Arrays,
    cards: list[int],
    device,
    seed: int = 0,
) -> dict[str, float]:
    backbone = build_model("tabtransformer", train.cont.shape[1], cards, cfg["model_defaults"], hp)
    model = JointModel(backbone, cfg["joint"]["adversary_hidden"]).to(device)
    model.load_state_dict({k: v for k, v in state.items()})
    ztr, zva = predict_joint(model, train, device)[2], predict_joint(model, val, device)[2]
    cols = [f"z{i}" for i in range(ztr.shape[1])]
    gbm = fit_lgbm_probe(
        pd.DataFrame(ztr, columns=cols),
        train.a,
        pd.DataFrame(zva, columns=cols),
        val.a,
        [],
        cfg["leakage"]["lightgbm"],
        seed,
    )
    p_gbm = gbm.predict_proba(pd.DataFrame(zva, columns=cols))[:, 1]
    sc = StandardScaler().fit(ztr)
    lr = LogisticRegression(max_iter=300).fit(sc.transform(ztr), train.a)
    p_lr = lr.predict_proba(sc.transform(zva))[:, 1]
    return {
        "z_probe_lgbm_auc": float(roc_auc_score(val.a, p_gbm)),
        "z_probe_logistic_auc": float(roc_auc_score(val.a, p_lr)),
        "lgbm_best_iteration": int(gbm.best_iteration_ or 0),
    }
