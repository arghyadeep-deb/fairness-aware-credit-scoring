"""Matched-pair lookup: nearest applicant of the OTHER gender in the saved reference pool.

Gender only selects the pool; it is never an input to the model. Distance = Euclidean on the scaled
numeric+binary block + 0.5 per mismatching categorical; the 50 numeric-nearest candidates are re-ranked.
"""

from __future__ import annotations

import numpy as np

from api.bundle import Bundle
from api.schemas import ApplicantRecord
from api.service import predict_arrays, to_arrays

K_CAND = 50
# A record this close to a pool row with the same categoricals IS that row (sample applicants are rounded copies of
# pool rows). Measured: samples sit 0.002 to 0.0075 from their own row; among 1,791 sampled pool rows the nearest
# other row with the same categoricals is at least 0.26 away.
SELF_TOL = 0.05


def nearest_other_gender(
    cont_pool, cat_pool, a_pool, cont, cat, own_gender_code: int
) -> tuple[int, bool]:
    """(pool index of the match, whether the applicant's own row was left out of the search).

    The own-row check runs over the WHOLE pool before the gender filter, so the returned flag is the same
    whichever gender is submitted and cannot reveal a pool row's recorded gender.
    """
    d_all = np.sqrt(((cont_pool - cont[None, :]) ** 2).sum(1))
    is_self = (d_all < SELF_TOL) & (cat_pool == cat[None, :]).all(1)
    cand = np.flatnonzero((a_pool != own_gender_code) & ~is_self)
    d = d_all[cand]
    top = np.argpartition(d, K_CAND)[:K_CAND]
    mism = (cat_pool[cand[top]] != cat[None, :]).sum(1)
    return int(cand[top[np.argmin(d[top] + 0.5 * mism)]]), bool(is_self.any())


def matched_pair(b: Bundle, rec: ApplicantRecord, gender: str) -> dict:
    cont, cat = to_arrays(b, rec)
    own = float(predict_arrays(b, cont, cat)[0])
    j, self_excluded = nearest_other_gender(
        b.pool_cont, b.pool_cat, b.pool_a, cont[0], cat[0], 1 if gender == "M" else 0
    )
    pm = float(predict_arrays(b, b.pool_cont[j : j + 1], b.pool_cat[j : j + 1])[0])
    cn, kn = b.spec["cont_names"], b.spec["cat_names"]
    feats = {n: float(v) for n, v in zip(cn, b.pool_cont[j])}
    feats.update({n: int(v) for n, v in zip(kn, b.pool_cat[j])})
    return {
        "own_probability": own,
        "matched_probability": pm,
        "delta": pm - own,
        "self_excluded": self_excluded,
        "matched_features": feats,
        "population_mean_abs_delta": float(b.docs["matched_stats"]["mean_abs_delta"]),
        "model_version": b.version,
    }
