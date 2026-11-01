"""Phase 4b-2: extended GRL rescue (spec GRL, lambda_f = lambda_c = 0, validation only), as pre-declared
in artifacts/RUN_LOG.md (Phase 4b PLAN). Judge: fresh LightGBM probe on frozen z (probe_z).
Success: probe drops >= 0.03 AND AUC-ROC loss <= 0.01 vs control (3-seed means for 3-seed variants).

python scripts/phase4b_grl_rescue.py <out_dir>        (resumable; time box 1 h of compute)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.eval.zprobe import probe_z  # noqa: E402
from src.run_arms import save_json, sigmoid  # noqa: E402
from src.run_joint import load_setup  # noqa: E402
from src.train_joint import fit_joint  # noqa: E402
from src.utils import load_config  # noqa: E402

LAMS = [0.05, 0.1, 0.2, 0.5]
PROBE_DROP, AUC_LOSS = 0.03, 0.01
PROM_DROP, PROM_LOSS = 0.02, 0.015
BOX_H = 1.0


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/train.yaml")
    device, train, val, cards, weights, hp = load_setup(cfg)
    f = out / "attempts.json"
    rows: list[dict] = json.loads(f.read_text()) if f.exists() else []
    t_start = time.time() - sum(r["seconds"] for r in rows)

    def key(stage, lam, extra, seed):
        return json.dumps([stage, lam, extra, seed], sort_keys=True)

    done = {key(r["stage"], r["lambda_a"], r["extra"], r["seed"]): r for r in rows}

    def attempt(stage: str, lam_a: float, extra: dict, seed: int = 0) -> dict:
        k = key(stage, lam_a, extra, seed)
        if k in done:
            return done[k]
        if (time.time() - t_start) / 3600 > BOX_H:
            raise TimeoutError("time box (1 h compute) reached")
        t0 = time.time()
        r = fit_joint(
            cfg, hp, train, val, cards, weights, seed, device, (0.0, 0.0, lam_a), schedule=extra
        )
        p = sigmoid(r["val_logits"])
        pr = probe_z(cfg, hp, r["state"], train, val, cards, device, seed=seed)
        h = r["history"]
        norms = [x["adv_gz_norm_med"] for x in h if x.get("adv_gz_norm_med") and x["ramp"] > 0]
        row = {
            "stage": stage, "lambda_a": lam_a, "extra": extra, "seed": seed,
            "auc_roc": float(roc_auc_score(val.y, p)),
            "auc_pr": float(average_precision_score(val.y, p)),
            "best_epoch": r["best_epoch"], "epochs": len(h),
            "adv_head_auc_best": h[r["best_epoch"]]["adv_head_auc"],
            "max_train_adv_loss": max((x.get("train_adv", 0.0) for x in h), default=0.0),
            "gz_norms": norms, **pr, "seconds": time.time() - t0,
        }  # fmt: skip
        rows.append(row)
        done[k] = row
        save_json(f, rows)
        print(f"{stage:8s} la={lam_a:<5} {extra} s{seed} AUC-ROC {row['auc_roc']:.4f} "
              f"probe {row['z_probe_lgbm_auc']:.4f} headAUC {row['adv_head_auc_best']:.3f} "
              f"maxAdvLoss {row['max_train_adv_loss']:.1f} ({row['seconds']:.0f}s)", flush=True)  # fmt: skip
        return row

    def ctl(seed):
        return attempt("control", 0.0, {}, seed)

    def judge(r, seed=0):
        c = ctl(seed)
        return c["z_probe_lgbm_auc"] - r["z_probe_lgbm_auc"], c["auc_roc"] - r["auc_roc"]

    try:
        ctl(0)
        fam: dict[str, list[dict]] = {
            "a": [{"adv_steps": 3}, {"adv_steps": 5}],
            "c": [{"adv_ramp": "ganin"}],
            "d": [{"adv_smooth": 0.1}, {"adv_smooth": 0.3}],
        }
        s0: list[tuple[str, dict, float, float, float]] = []  # fam, setting, lam, drop, loss
        for st in fam["a"]:
            for la in LAMS:
                r = attempt("a_steps", la, st)
                s0.append(("a", st, la, *judge(r)))
        norms = [n for r in rows if r["stage"] == "a_steps" for n in r["gz_norms"]]
        m = float(np.median(norms))
        print(f"== median per-row reversed-gradient norm logged in (a): {m:.4g}", flush=True)
        fam["b"] = [{"adv_clip": round(0.5 * m, 6)}, {"adv_clip": round(0.2 * m, 6)}]
        for fm, stage in (("b", "b_clip"), ("c", "c_ganin"), ("d", "d_smooth")):
            for st in fam[fm]:
                for la in LAMS:
                    r = attempt(stage, la, st)
                    s0.append((fm, st, la, *judge(r)))
        prom = [x for x in s0 if x[3] >= PROM_DROP and x[4] <= PROM_LOSS]
        print(f"== promising seed-0 entries: {len(prom)}", flush=True)
        fams = sorted({x[0] for x in prom})
        best_by_fam = {}
        for fm in "abcd":
            c = [x for x in (prom if fm in fams else s0) if x[0] == fm]
            if c:
                best_by_fam[fm] = max(c, key=lambda x: x[3])[1]
        chosen_f = (
            fams
            if len(fams) >= 2
            else sorted(best_by_fam, key=lambda fm: -max(x[3] for x in s0 if x[0] == fm))[:2]
        )
        combo: dict = {}
        for fm in chosen_f:
            combo.update(best_by_fam[fm])
        print(f"== (e) combination of families {chosen_f}: {combo}", flush=True)
        for la in LAMS:
            r = attempt("e_combo", la, combo)
            s0.append(("e", combo, la, *judge(r)))
        prom = [x for x in s0 if x[3] >= PROM_DROP and x[4] <= PROM_LOSS]
        print(f"== promising incl. (e): {len(prom)} -> seeds 1, 2", flush=True)
        stage_of = {"a": "a_steps", "b": "b_clip", "c": "c_ganin", "d": "d_smooth", "e": "e_combo"}
        for s in (1, 2):
            ctl(s)
        for fm, st, la, _, _ in prom:
            for s in (1, 2):
                attempt(stage_of[fm], la, st, s)
    except TimeoutError as e:
        print("==", e, flush=True)

    # summary: per variant (stage, lambda_a, extra) over the seeds available
    summ = []
    ctrls = {r["seed"]: r for r in rows if r["stage"] == "control"}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["stage"] != "control":
            groups.setdefault(
                json.dumps([r["stage"], r["lambda_a"], r["extra"]], sort_keys=True), []
            ).append(r)
    for k, rs in groups.items():
        sd = sorted(r["seed"] for r in rs)
        drop = float(
            np.mean([ctrls[r["seed"]]["z_probe_lgbm_auc"] - r["z_probe_lgbm_auc"] for r in rs])
        )
        loss = float(np.mean([ctrls[r["seed"]]["auc_roc"] - r["auc_roc"] for r in rs]))
        stage, la, extra = json.loads(k)
        summ.append({"stage": stage, "lambda_a": la, "extra": extra, "seeds": sd,
                     "probe_mean": float(np.mean([r["z_probe_lgbm_auc"] for r in rs])),
                     "auc_roc_mean": float(np.mean([r["auc_roc"] for r in rs])),
                     "probe_drop": drop, "auc_loss": loss,
                     "works": bool(len(sd) == 3 and drop >= PROBE_DROP and loss <= AUC_LOSS),
                     "works_seed0_only": bool(sd == [0] and drop >= PROBE_DROP and loss <= AUC_LOSS)})  # fmt: skip
    save_json(out / "summary.json", {"criterion": {"probe_drop": PROBE_DROP, "auc_loss": AUC_LOSS},
              "promising": {"probe_drop": PROM_DROP, "auc_loss": PROM_LOSS},
              "hours": (time.time() - t_start) / 3600, "variants": summ})  # fmt: skip
    print("works (3 seeds):", [s for s in summ if s["works"]])


if __name__ == "__main__":
    main()
