"""Build notebooks/02_baselines.ipynb (thin report notebook; reads results written by src/)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
WHAT_IT_SHOWS = (ROOT / "scripts" / "nb02_what_this_shows.md").read_text(encoding="utf-8")

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "name": "fairness-credit",
    "display_name": "fairness-credit",
    "language": "python",
}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md(
        "# 02 Baselines: imbalance fixes vs calibration and fairness\n\n"
        "**Goal:** compare plain CE, SMOTE-NC, weighted CE, focal loss, other backbones and the "
        "post-hoc fixes (S1 prior correction, S2 group thresholds) under one threshold rule "
        "(flag the top 10% by risk).\n"
        "All numbers are on the **validation** split (test is locked until Phase 6): mean over "
        "3 seeds, 95% CI from 1,000 paired stratified bootstrap resamples."
    ),
    code(
        "import json, sys\nfrom pathlib import Path\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n\n"
        "import numpy as np, pandas as pd\nimport matplotlib.pyplot as plt\n"
        "from src.utils import load_config, resolve\nfrom src.data.tensors import load_arrays\n"
        "from src.eval.metrics import ece_equal_freq\n\n"
        "cfg = load_config('configs/train.yaml')\n"
        "runs = sorted(p for p in resolve(cfg['runs_dir']).glob('*_phase2') if (p / 'results.csv').exists())\n"
        "RUN = runs[-1]\nprint('run folder:', RUN.name)\n"
        "res = pd.read_csv(RUN / 'results.csv').set_index('key')\n"
        "meta = json.loads((RUN / 'run_meta.json').read_text())\n"
        "print('git', meta['git_hash'][:10], '| seeds', meta['seeds'], '|', meta['split'])\n"
        "val, _ = load_arrays(cfg, 'val')"
    ),
    md("## Main table (shared rule: top 10% flagged)"),
    code(
        "def fmt(k, m, d=3):\n"
        "    return res.apply(lambda r: f\"{r[m]:.{d}f} [{r[m+'_lo']:.{d}f}, {r[m+'_hi']:.{d}f}]\", axis=1)\n"
        "cols = {'auc_roc': 'AUC-ROC', 'auc_pr': 'AUC-PR', 'ks': 'KS', 'recall': 'Recall@10%',\n"
        "        'dir': 'DIR', 'eod': 'EOD', 'ece': 'ECE', 'worst_group_ece': 'Worst-group ECE',\n"
        "        'brier': 'Brier'}\n"
        "main = pd.DataFrame({v: fmt(None, k) for k, v in cols.items()})\n"
        "main.insert(0, 'arm', res['arm'])\nmain"
    ),
    md("## REF (LightGBM): what changed since the first Phase 2 run"),
    code(
        "old = sorted(p for p in resolve(cfg['runs_dir']).glob('*_phase2') if (p / 'results.csv').exists())[0]\n"
        "o = pd.read_csv(old / 'results.csv').set_index('key').loc['ref']\n"
        "it_old = json.loads((old / 'ref' / 'seed_0' / 'pred_meta.json').read_text())['best_iteration']\n"
        "it_new = json.loads((RUN / 'ref' / 'seed_0' / 'pred_meta.json').read_text())['best_iteration']\n"
        "n = res.loc['ref']\n"
        "pd.DataFrame({'first run': [o['auc_roc'], o['auc_pr'], it_old], 'this run': [n['auc_roc'], n['auc_pr'], it_new]},\n"
        "             index=['AUC-ROC', 'AUC-PR', 'boosting rounds kept']).round(4)"
    ),
    md("## Imbalance damages calibration; prior correction repairs it"),
    code(
        "keys = ['a0', 'a1', 'a1b', 'a2', 's1']\n"
        "cal = res.loc[keys, ['ece', 'worst_group_ece', 'cal_slope', 'cal_intercept', 'brier',\n"
        "                      'recall_at_0.5', 'flag_rate_at_0.5']].round(4)\n"
        "cal.columns = ['ECE', 'worst-group ECE', 'cal slope', 'cal intercept', 'Brier',\n"
        "               'recall at p>0.5 (diagnostic)', 'flag rate at p>0.5 (diagnostic)']\ncal"
    ),
    code(
        "order = [k for k in ['a0','a1','a1b','a2','a3','a4','a5','s1','g1','ref'] if k in res.index]\n"
        "fig, ax = plt.subplots(figsize=(8, 3.5))\n"
        "y = res.loc[order, 'ece']\n"
        "err = [(y - res.loc[order, 'ece_lo']).clip(lower=0), (res.loc[order, 'ece_hi'] - y).clip(lower=0)]\n"
        "ax.bar(order, y, yerr=err, color='#3b6ea5', capsize=3)\n"
        "ax.set_ylabel('ECE (15 equal-frequency bins)'); ax.set_title('Calibration error by arm (val, 95% CI)')\n"
        "plt.tight_layout(); plt.show()"
    ),
    code(
        "fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)\n"
        "for a_, key in zip(ax, ['a0', 'a2', 's1']):\n"
        "    p = np.load(RUN / key / 'seed_0' / 'val_prob.npy')\n"
        "    order_ = np.argsort(p); idx = np.array_split(order_, 15)\n"
        "    mp = [p[i].mean() for i in idx]; my = [val.y[i].mean() for i in idx]\n"
        "    a_.plot([0, 1], [0, 1], 'k--', lw=1); a_.plot(mp, my, 'o-', color='#c0504d')\n"
        "    a_.set_title(f\"{key.upper()} reliability (seed 0)\"); a_.set_xlabel('mean predicted')\n"
        "ax[0].set_ylabel('observed default rate')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## Does the imbalance fix change group-wise calibration? (A0 per gender)"),
    code(
        "g = res.loc[['a0', 'a1', 'a1b', 'a2', 's1', 'ref'], ['ece_f', 'ece_m', 'tpr_f', 'tpr_m', 'fpr_f', 'fpr_m']].round(4)\n"
        "g.columns = ['ECE female', 'ECE male', 'TPR female', 'TPR male', 'FPR female', 'FPR male']\ng"
    ),
    md("## Fairness at the shared threshold"),
    code(
        "fo = [k for k in ['a0','a1','a1b','a2','a3','a4','a5','s1','s2_gt','s2_eg','g1','gi'] if k in res.index]\n"
        "fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))\n"
        "ax[0].bar(fo, res.loc[fo, 'dir'], color='#3b6ea5'); ax[0].axhline(0.8, color='r', ls='--', lw=1)\n"
        "ax[0].set_title('Disparate impact ratio (>= 0.8 target)'); ax[0].set_ylim(0.6, 1.02)\n"
        "ax[1].bar(fo, res.loc[fo, 'eod'], color='#8a6fb0'); ax[1].set_title('Equalized-odds difference')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## Discrimination vs calibration trade-off per arm"),
    code(
        "d = res.loc[[k for k in res.index if k not in ('s2_fairlearn',)]]\n"
        "fig, ax = plt.subplots(figsize=(6.5, 4.5))\n"
        "ax.scatter(d['worst_group_ece'], d['auc_pr'], color='#3b6ea5')\n"
        "for k, r in d.iterrows(): ax.annotate(k, (r['worst_group_ece'], r['auc_pr']), xytext=(4, 3), textcoords='offset points')\n"
        "ax.set_xlabel('worst-group ECE (lower is better)'); ax.set_ylabel('AUC-PR (higher is better)')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## Tuning and training cost"),
    code(
        "rows = []\n"
        "for k in ['a0','a1','a1b','a2','a3','a4','a5','g1']:\n"
        "    t = json.loads((RUN / k / 'tuning.json').read_text())\n"
        "    s = json.loads((RUN / k / 'summary.json').read_text())['seeds']\n"
        "    rows.append({'arm': k, 'best tuning AUC-PR': round(t['best_value'], 4),\n"
        "                 'tuning s': round(t['seconds']), 'mean epochs': np.mean([v['epochs_run'] for v in s.values()]),\n"
        "                 'mean train s': round(np.mean([v['seconds'] for v in s.values()]), 1),\n"
        "                 **{p: round(v, 5) for p, v in t['best_params'].items()}})\n"
        "pd.DataFrame(rows).set_index('arm')"
    ),
    md(WHAT_IT_SHOWS),
]
out = ROOT / "notebooks" / "02_baselines.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
