"""Three-way lambda exploration (VALIDATION ONLY; declared in RUN_LOG commit 58453d5; post-test; never served).

python scripts/phaseE_three_way.py <out_dir>     (resumable)

Grid W: class-weighted CE + A2 hp (as JT): lf {0.05,0.1} x lc {0.5,1,2} x la {0.1,0.25}.
Grid U: unweighted CE (weights 1/1) + hp from Phase 2 a0 tuning: lf {0.05,0.1} x lc {0,0.5} x la {0,0.25}.
3 seeds, adv_steps 5. Does not apply or edit the selection rule.
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.eval.bootstrap import bootstrap_metrics  # noqa: E402
from src.eval.metrics import METRIC_NAMES  # noqa: E402
from src.eval.zprobe import probe_z  # noqa: E402
from src.run_arms import flags_from_shared_rule, save_json  # noqa: E402
from src.run_joint import load_setup, run_point, tag  # noqa: E402
from src.utils import load_config  # noqa: E402

SEEDS = [0, 1, 2]
N_BOOT = 500
SCHED = {"adv_steps": 5}
PHASE2 = ROOT / "artifacts" / "runs" / "20261101T055729Z_phase2"
GRIDS = {
    "grid_w": list(itertools.product([0.05, 0.1], [0.5, 1.0, 2.0], [0.1, 0.25])),
    "grid_u": list(itertools.product([0.05, 0.1], [0.0, 0.5], [0.0, 0.25])),
}
COLS = ["auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]
DIFF_COLS = ["auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]
A2_AUC, S1_ECE = 0.7460, 0.0123


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    p6 = ROOT / Path((ROOT / "artifacts" / "phase6_rundir.txt").read_text().strip())
    cfg = load_config("configs/train.yaml")
    fr, eb = cfg["evaluation"]["flag_rate"], cfg["evaluation"]["ece_bins"]
    tun = PHASE2 / "a0" / "tuning.json"
    assert tun.exists(), tun
    setup_w = load_setup(cfg)
    device, train, val, cards, weights, hp = setup_w
    setup_u = (device, train, val, cards, {"0": 1.0, "1": 1.0}, json.loads(tun.read_text())["best_params"])
    setups = {"grid_w": setup_w, "grid_u": setup_u}

    rows = []
    for gname, pts in GRIDS.items():
        setup = setups[gname]
        for lam in pts:
            for s in SEEDS:
                info = run_point(cfg, out, lam, s, setup, name=gname, schedule=SCHED)
                d = out / gname / tag(lam) / f"seed_{s}"
                if not (d / "probe.json").exists():
                    state = torch.load(d / "best.pt")
                    pr = probe_z(cfg, setup[5], state, train, val, cards, device, seed=s)
                    save_json(d / "probe.json", pr)
                    logging.info("%s %s s%d probe %.4f", gname, tag(lam), s, pr["z_probe_lgbm_auc"])
                pr = json.loads((d / "probe.json").read_text())
                rows.append({"grid": gname, "lambda_f": lam[0], "lambda_c": lam[1], "lambda_a": lam[2], "seed": s,
                             **{c: info[f"val_{c}"] for c in COLS}, "z_probe_lgbm_auc": pr["z_probe_lgbm_auc"]})
            torch.cuda.empty_cache()
    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(out / "three_way_seed_rows.csv", index=False)

    metrics = COLS + ["z_probe_lgbm_auc"]
    key = ["grid", "lambda_f", "lambda_c", "lambda_a"]
    g = seed_df.groupby(key, sort=False)
    tab = g[metrics].mean().add_suffix("_mean").join(g[metrics].std(ddof=1).add_suffix("_sd")).join(g.size().rename("n_seeds")).reset_index()
    tab["passes_eod"] = tab["eod_mean"] <= 0.04
    tab["passes_dir"] = tab["dir_mean"] >= 0.80
    tab["passes_auc"] = tab["auc_roc_mean"] >= A2_AUC - 0.02
    tab["passes_ece"] = tab["worst_group_ece_mean"] <= max(1.5 * S1_ECE, 0.02)
    tab["passes_all"] = tab[["passes_eod", "passes_dir", "passes_auc", "passes_ece"]].all(axis=1)
    pr_, eo, we = tab["auc_pr_mean"].to_numpy(), tab["eod_mean"].to_numpy(), tab["worst_group_ece_mean"].to_numpy()
    front = []
    for i in range(len(tab)):
        dom = (pr_ >= pr_[i]) & (eo <= eo[i]) & (we <= we[i]) & ((pr_ > pr_[i]) | (eo < eo[i]) | (we < we[i]))
        front.append(not dom.any())
    tab["on_frontier"] = front
    tab.to_csv(out / "three_way_table.csv", index=False)

    # paired bootstrap vs references
    def boot(prob_by_seed, flags_by_seed):
        return np.nanmean(np.stack([bootstrap_metrics(val.y, prob_by_seed[s], val.a, flags_by_seed[s], N_BOOT, s, eb,
                                                      n_jobs=4, chunk_size=100) for s in SEEDS]), axis=0)

    pj = {s: np.load(p6 / "jta_ctl" / "f0p05_c2_a0" / f"seed_{s}" / "val_prob.npy") for s in SEEDS}
    refs = {
        "JT chosen": boot(pj, {s: flags_from_shared_rule(pj[s], fr)[0] for s in SEEDS}),
        "S2-EG": boot({s: np.load(p6 / "s2_eg_v2" / f"seed_{s}" / "val_prob.npy") for s in SEEDS},
                      {s: np.load(p6 / "s2_eg_v2" / f"seed_{s}" / "val_flags.npy").astype(bool) for s in SEEDS}),
    }
    diffs = []
    for _, r in tab[tab["passes_all"] | tab["on_frontier"]].iterrows():
        lam = (r["lambda_f"], r["lambda_c"], r["lambda_a"])
        ds = [out / r["grid"] / tag(lam) / f"seed_{s}" for s in SEEDS]
        pp = {s: np.load(ds[s] / "val_prob.npy") for s in SEEDS}
        ff = {s: np.load(ds[s] / "val_flags.npy").astype(bool) for s in SEEDS}
        b = boot(pp, ff)
        for rn, rb in refs.items():
            dd = b - rb
            for mn in DIFF_COLS:
                j = METRIC_NAMES.index(mn)
                lo, hi = np.nanpercentile(dd[:, j], [2.5, 97.5])
                diffs.append({"grid": r["grid"], "lambda_f": lam[0], "lambda_c": lam[1], "lambda_a": lam[2], "minus": rn,
                              "metric": mn, "diff": float(np.nanmean(dd[:, j])), "lo": float(lo), "hi": float(hi),
                              "sig": bool(lo > 0 or hi < 0)})
        logging.info("bootstrap done %s %s", r["grid"], tag(lam))
    pd.DataFrame(diffs).to_csv(out / "three_way_diffs.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    print(tab.round(4).to_string(index=False))
    print(pd.DataFrame(diffs).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
