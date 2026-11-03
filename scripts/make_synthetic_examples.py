"""Make 8 synthetic demo applicants (api/data/synthetic_examples.json); 2026-11-03.

The demo's "Load example" button used 8 real held-out rows (bundle examples.json). These synthetic
records replace them in /examples; the frozen bundle is not changed.

How a record is built: categorical fields and flags come from one TRAINING row; each numeric group
(age and employment, income, loan amounts, region, external scores) comes from a different training
row with the same income type; amounts are rescaled by a random factor and rounded. Training rows
only are drawn on (no held-out row is used as a source).

Check against reproducing a real applicant: in the model's scaled space (Euclidean on the numeric
block + 0.5 per mismatching categorical, as in the matched-pair search) a record is kept only if its
nearest real applicant (training rows AND the held-out reference pool) is at least as far away as
the median distance from a real held-out applicant to its nearest training applicant.

python scripts/make_synthetic_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from api.bundle import load_bundle  # noqa: E402
from api.schemas import ApplicantRecord  # noqa: E402
from api.service import predict, to_arrays  # noqa: E402
from scripts.freeze_bundle import inverse_example  # noqa: E402
from src.data.tensors import load_arrays  # noqa: E402
from src.utils import load_config  # noqa: E402

OUT = ROOT / "api" / "data" / "synthetic_examples.json"
WANT = {"Low": 3, "Medium": 3, "High": 2}
SOURCE = "synthetic: assembled from training-row fields, not real applicants"
RULE = "nearest real applicant (train + held-out pool) >= median real-to-real nearest distance"


def nearest(cont_ref, cat_ref, cont, cat):
    d = np.sqrt(((cont_ref - cont[None, :]) ** 2).sum(1)) + 0.5 * (cat_ref != cat[None, :]).sum(1)
    return float(d.min())


def main() -> None:
    torch.set_num_threads(4)
    rng = np.random.default_rng(20261103)
    b = load_bundle()
    train, _ = load_arrays(load_config("configs/train.yaml"), "train")
    ref_cont = np.concatenate([train.cont, b.pool_cont]).astype("float32")
    ref_cat = np.concatenate([train.cat, b.pool_cat])

    # typical real-to-real distance: held-out row -> nearest training row
    probe = rng.choice(len(b.pool_cont), 400, replace=False)
    real_nn = [nearest(train.cont, train.cat, b.pool_cont[i], b.pool_cat[i]) for i in probe]
    floor = float(np.median(real_nn))

    donors = []
    for i in rng.choice(len(train.cont), 6000, replace=False):
        r = inverse_example(b.pre, train.cont[i], train.cat[i], b.spec)
        if r is not None:
            donors.append(r)
    by_type: dict[str, list[dict]] = {}
    for r in donors:
        by_type.setdefault(r["name_income_type"], []).append(r)
    pick = lambda t: by_type[t][rng.integers(len(by_type[t]))]  # noqa: E731

    out, got, tries, nn = [], {k: 0 for k in WANT}, 0, []
    while sum(got.values()) < sum(WANT.values()) and tries < 20000:
        tries += 1
        base = donors[rng.integers(len(donors))]
        t = base["name_income_type"]
        if len(by_type[t]) < 50:
            continue
        rec = {
            k: base[k]
            for k in (
                "name_family_status",
                "name_education_type",
                "name_income_type",
                "name_housing_type",
                "occupation_type",
                "organization_type",
                "flag_own_car",
                "flag_own_realty",
                "cnt_children",
                "cnt_fam_members",
            )
        }
        ae = pick(t)
        rec["age_years"] = float(round(ae["age_years"]))
        emp = ae["years_employed"]
        rec["years_employed"] = None if emp is None else round(min(emp, rec["age_years"] - 18), 1)
        rec["amt_income_total"] = float(
            round(pick(t)["amt_income_total"] * rng.uniform(0.85, 1.15), -3)
        )
        ln = pick(t)
        f = rng.uniform(0.85, 1.15)
        rec["amt_goods_price"] = float(round(ln["amt_goods_price"] * f, -3))
        rec["amt_credit"] = float(round(max(ln["amt_credit"] * f, rec["amt_goods_price"]), -2))
        rec["amt_annuity"] = float(round(ln["amt_annuity"] * f, -1))
        rec["region_population_relative"] = pick(t)["region_population_relative"]
        es = pick(t)
        for k in (1, 2, 3):
            v = es[f"ext_source_{k}"]
            rec[f"ext_source_{k}"] = None if v is None else round(v, 2)
        if min(rec["amt_income_total"], rec["amt_goods_price"], rec["amt_annuity"]) <= 0:
            continue
        try:
            ar = ApplicantRecord(**rec)
        except Exception:
            continue
        res = predict(b, ar)
        if res["warnings"] or got[res["risk_band"]] >= WANT[res["risk_band"]]:
            continue
        c, k = to_arrays(b, rec)
        d = nearest(ref_cont, ref_cat, c[0], k[0])
        if d < floor:
            continue
        got[res["risk_band"]] += 1
        nn.append(d)
        out.append(rec)
    assert len(out) == sum(WANT.values()), (got, tries)
    order = np.argsort([predict(b, ApplicantRecord(**r))["probability"] for r in out])
    out = [out[i] for i in order]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": SOURCE,
                "check": {
                    "rule": RULE,
                    "median_real_nn_distance": floor,
                    "min_synthetic_nn_distance": min(nn),
                    "synthetic_nn_distances": sorted(nn),
                },
                "examples": out,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"floor {floor:.3f}  nearest-real distances {np.round(sorted(nn), 3)}  tries {tries}")
    for r in out:
        p = predict(b, ApplicantRecord(**r))
        print(
            f"{p['probability']:.3f} {p['risk_band']:6s} age {r['age_years']:.0f} "
            f"income {r['amt_income_total']:.0f} credit {r['amt_credit']:.0f}"
        )


if __name__ == "__main__":
    main()
