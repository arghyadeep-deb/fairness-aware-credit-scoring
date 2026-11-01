"""Build notebooks/04_lambda_sweep_and_selection.ipynb (thin report; reads results written by src/)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
WHAT = (ROOT / "scripts" / "nb04_what_this_shows.md").read_text(encoding="utf-8")
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "name": "fairness-credit",
    "display_name": "fairness-credit",
    "language": "python",
}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md(
        "# 04 Lambda sweep, adversary rescue and operating-point selection\n\n"
        "**Goal:** show the fairness x calibration trade-off surface of the joint model (JT), the "
        "outcome of the time-boxed adversary rescue, the committed selection rule applied once, "
        "and JT against A2, S1, S2-EG and S2-GT at the shared 10% flag rate.\n"
        "Validation only (test untouched). Mean over 3 seeds."
    ),
    code(
        "import json, sys, subprocess\nfrom pathlib import Path\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import numpy as np, pandas as pd\nimport matplotlib.pyplot as plt\n"
        "pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)\n"
        "RUN = ROOT / Path((ROOT / 'artifacts' / 'phase4_rundir.txt').read_text().strip())\n"
        "P2 = ROOT / 'artifacts' / 'runs' / '20261101T055729Z_phase2'\n"
        "FIG = RUN / 'figures'; FIG.mkdir(exist_ok=True)\n"
        "pts = pd.read_csv(RUN / 'sweep_points.csv'); sel = json.loads((RUN / 'selection.json').read_text())\n"
        "print('run folder:', RUN.name, '| sweep points:', len(pts), '| rule step used:', sel['step_used'])\n"
        "print('rule file:', subprocess.run(['git','log','--oneline','-1','--','configs/selection_rule.yaml'],\n"
        "      cwd=ROOT, capture_output=True, text=True).stdout.strip())\n"
        "print('lambda-grid commit:', subprocess.run(['git','log','--oneline','-1','--','specs/09_resolved_decisions.md'],\n"
        "      cwd=ROOT, capture_output=True, text=True).stdout.strip())"
    ),
    md(
        "## 1. Main sweep: fairness x calibration, adversary off (20 points x 3 seeds)\n"
        "Grid: lambda_f in {0, 0.05, 0.1, 0.25, 0.5}, lambda_c in {0, 0.5, 1, 2}. Terms are normalised by "
        "their initial magnitude."
    ),
    code(
        "main = pts[pts.name == 'jt'].copy()\n"
        "cols = ['tag','auc_roc','auc_roc_sd','auc_pr','eod','eod_sd','worst_group_ece','worst_group_ece_sd','dir','pareto']\n"
        "main[cols].round(4).reset_index(drop=True)"
    ),
    md("## 2. Trade-off surface: default AUC-ROC vs EOD vs worst-group ECE"),
    code(
        "fs, cs = sorted(main.lambda_f.unique()), sorted(main.lambda_c.unique())\n"
        "fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))\n"
        "for a, (m, t, cm) in zip(ax, [('auc_roc','AUC-ROC (higher better)','viridis'),\n"
        "                              ('eod','EOD (lower better)','viridis_r'),\n"
        "                              ('worst_group_ece','worst-group ECE (lower better)','viridis_r')]):\n"
        "    g = main.pivot(index='lambda_f', columns='lambda_c', values=m).loc[fs, cs]\n"
        "    im = a.imshow(g.values, origin='lower', cmap=cm, aspect='auto')\n"
        "    a.set_xticks(range(len(cs)), cs); a.set_yticks(range(len(fs)), fs)\n"
        "    a.set_xlabel('lambda_c'); a.set_ylabel('lambda_f'); a.set_title(t, fontsize=9)\n"
        "    for i in range(len(fs)):\n"
        "        for j in range(len(cs)): a.text(j, i, f'{g.values[i, j]:.3f}', ha='center', va='center', color='w', fontsize=7)\n"
        "    plt.colorbar(im, ax=a, fraction=0.046)\n"
        "plt.tight_layout(); plt.savefig(FIG / 'surface_heatmaps.png', dpi=150); plt.show()"
    ),
    code(
        "allp = pts.copy()\n"
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))\n"
        "for a, (x, y, c) in zip(ax, [('eod','auc_roc','worst_group_ece'), ('worst_group_ece','eod','auc_roc')]):\n"
        "    sc = a.scatter(allp[x], allp[y], c=allp[c], cmap='viridis_r' if c != 'auc_roc' else 'viridis', s=45,\n"
        "                   marker='o', edgecolor='k', linewidth=0.3)\n"
        "    f = allp[allp.pareto]\n"
        "    a.scatter(f[x], f[y], facecolor='none', edgecolor='red', s=110, linewidth=1.2, label='Pareto frontier')\n"
        "    ch = allp[(allp.tag == sel['chosen']['tag']) & (allp.name == sel['chosen']['name'])]\n"
        "    a.scatter(ch[x], ch[y], marker='*', s=260, color='gold', edgecolor='k', label='chosen', zorder=5)\n"
        "    a.set_xlabel(x); a.set_ylabel(y); plt.colorbar(sc, ax=a, label=c); a.legend(fontsize=8)\n"
        "ax[1].set_xscale('log'); ax[0].axvline(0.04, color='grey', ls='--', lw=0.8)\n"
        "plt.tight_layout(); plt.savefig(FIG / 'tradeoff_scatter.png', dpi=150); plt.show()\n"
        "rng = main[['auc_roc','eod','worst_group_ece']].agg(['min','max']).T\n"
        "rng['range'] = rng['max'] - rng['min']\n"
        "rng['typical seed sd'] = [main.auc_roc_sd.mean(), main.eod_sd.mean(), main.worst_group_ece_sd.mean()]\n"
        "rng['range / seed sd'] = rng['range'] / rng['typical seed sd']\n"
        "rng.round(4)"
    ),
    md(
        "## 3. Adversary rescue (seed 0, fairness and calibration terms off)\n"
        "Success criterion fixed before the first attempt: fresh LightGBM probe on frozen z drops by at least "
        "0.03 AUC vs the control AND default AUC-ROC loses at most 0.01. Judged by the probe, not by the "
        "adversary head's own AUC."
    ),
    code(
        "att = pd.DataFrame(json.loads((RUN / 'adversary_rescue' / 'attempts.json').read_text()))\n"
        "ctrl = att[att.stage == 'control'].iloc[0]\n"
        "att['probe_drop'] = ctrl.z_probe_lgbm_auc - att.z_probe_lgbm_auc\n"
        "att['auc_loss'] = ctrl.auc_roc - att.auc_roc\n"
        "att['works'] = (att.probe_drop >= 0.03) & (att.auc_loss <= 0.01) & (att.stage != 'control')\n"
        "att['adv_mode'] = att.get('adv_mode', pd.Series(['grl'] * len(att))).fillna('grl')\n"
        "att['adv_lr_mult'] = att.get('adv_lr_mult', pd.Series([1.0] * len(att))).fillna(1.0)\n"
        "att[['stage','lambda_a','adv_mode','adv_lr_mult','auc_roc','z_probe_lgbm_auc','z_probe_logistic_auc',\n"
        "     'probe_drop','auc_loss','adv_head_auc_best','max_train_adv_loss','works']].round(4)"
    ),
    code(
        "# the earlier GRL failure at the original strength (artifacts/smoke/joint_stability/real_schedule.json)\n"
        "old = json.loads((ROOT / 'artifacts/smoke/joint_stability/real_schedule.json').read_text())\n"
        "pd.DataFrame([{'setting': k, 'min AUC-ROC after ramp': min(h['val_auc_roc'] for h in v['history'] if h['ramp'] >= 1),\n"
        "               'max adversary loss': max(h.get('train_adv', 0) for h in v['history'])}\n"
        "              for k, v in old.items() if k in ['(0.0, 0.0, 0.5)', '(0.0, 0.0, 1.0)']]).round(3)"
    ),
    md("## 4. Committed selection rule, applied once to the combined set"),
    code(
        "print(json.dumps({k: v for k, v in sel.items() if k not in ('chosen', 'git_hash')}, indent=1))\n"
        "print('bound for worst-group ECE:', sel['ece_bound'], '| AUC-ROC floor:', round(sel['auc_floor'], 4))\n"
        "pts[pts.all_ok][['name','tag','auc_roc','auc_pr','eod','worst_group_ece','dir']].round(4)"
    ),
    code(
        "near = pts.assign(ece_gap=pts.worst_group_ece - sel['ece_bound']).sort_values('worst_group_ece')\n"
        "near[(near.eod <= 0.04) & (near.auc_roc >= sel['auc_floor'])][\n"
        "    ['name','tag','auc_roc','auc_pr','eod','worst_group_ece','ece_gap','all_ok']].round(4).head(10)"
    ),
    md("## 5. Chosen JT point vs A2, S1, S2-EG (deployable), S2-GT (analysis only)"),
    code(
        "cmp = pd.read_csv(RUN / 'compare.csv')\n"
        "def fmt(r, k): return f\"{r[k]:.4f} [{r[k+'_lo']:.4f}, {r[k+'_hi']:.4f}]\"\n"
        "tbl = pd.DataFrame({'arm': cmp.arm, **{k: [fmt(r, k) for _, r in cmp.iterrows()]\n"
        "                    for k in ['auc_roc','auc_pr','worst_group_ece','eod','dir','recall']}})\n"
        "tbl"
    ),
    code(
        "fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))\n"
        "for a, k in zip(ax, ['auc_pr','eod','worst_group_ece','recall']):\n"
        "    y = cmp[k].values; e = np.clip([y - cmp[k+'_lo'], cmp[k+'_hi'] - y], 0, None)\n"
        "    a.barh(range(len(cmp)), y, xerr=e, color=['#999','#999','#3b6ea5','#c98f2e','#b03a2e'])\n"
        "    a.set_yticks(range(len(cmp)), [s.split(' (')[0].split(' chosen')[0] for s in cmp.arm] if k == 'auc_pr' else [])\n"
        "    a.set_title(k, fontsize=9)\n"
        "plt.tight_layout(); plt.savefig(FIG / 'jt_vs_baselines.png', dpi=150); plt.show()"
    ),
    code((ROOT / "scripts" / "nb04_cell_paired.py").read_text(encoding="utf-8")),
    md("## 6. Ablations at the chosen settings (3 seeds) with a fresh probe on frozen z"),
    code(
        "ab = pd.read_csv(RUN / 'ablation.csv')\n"
        "ab['AUC-ROC'] = [fmt(r, 'auc_roc') for _, r in ab.iterrows()]\n"
        "ab['EOD'] = [fmt(r, 'eod') for _, r in ab.iterrows()]\n"
        "ab[['arm','AUC-ROC','auc_pr','EOD','worst_group_ece','z_probe_lgbm_auc','z_probe_lgbm_sd','z_probe_logistic_auc']].round(4)"
    ),
    md("## 7. Test set untouched"),
    code(
        "log = (ROOT / 'artifacts/guard/test_access.jsonl').read_text().strip().splitlines()\n"
        "print(len(log), 'guard-log entry:', json.loads(log[0]).get('purpose'))\n"
        "assert len(log) == 1 and json.loads(log[0]).get('purpose') == 'leakage_probe_gender_only'"
    ),
    md(WHAT),
]
out = ROOT / "notebooks" / "04_lambda_sweep_and_selection.ipynb"
nbf.write(nb, out)
print("wrote", out)
