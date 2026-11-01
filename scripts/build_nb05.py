"""Build notebooks/05_posthoc_comparison.ipynb (thin report; reads results written by src/ and scripts/)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "name": "fairness-credit",
    "display_name": "fairness-credit",
    "language": "python",
}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md(
        "# 05 Phase 4b and Phase 5: fine lambda stage, tuned JT, GRL rescue, H arms, post-hoc comparison\n\n"
        "**Goal:** (4b-0) second application of the committed rule on 34 JT points; (4b-1) JT with the same "
        "Optuna budget as the other arms; (4b-2) extended GRL rescue; H1 to H3 (training for fairness, formula "
        "for calibration); Phase 5 comparison at the shared 10% flag rate and at matched fairness, with every "
        "arm labelled deployable (gender-blind at inference) or analysis only (uses gender).\n"
        "Validation only, test set untouched. Mean over 3 seeds."
    ),
    code(
        "import json, sys, subprocess\nfrom pathlib import Path\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import numpy as np, pandas as pd\nimport matplotlib.pyplot as plt\n"
        "pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)\n"
        "P4 = ROOT / Path((ROOT / 'artifacts' / 'phase4_rundir.txt').read_text().strip())\n"
        "P4B = ROOT / Path((ROOT / 'artifacts' / 'phase4b_rundir.txt').read_text().strip())\n"
        "P5 = P4B / 'phase5'; FIG = P5 / 'figures'; FIG.mkdir(exist_ok=True)\n"
        "git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()\n"
        "print('phase4b run folder:', P4B.name)\n"
        "print('rule file:', git('log', '--oneline', '-1', '--', 'configs/selection_rule.yaml'))\n"
        "print('plan commits:', git('log', '--oneline', '--grep=phase4b', '-5'))"
    ),
    md("## 1. 4b-0: fine stage, rule applied a second time to the union of 34 JT points"),
    code(
        "sel = json.loads((P4 / 'fine_union' / 'selection_union.json').read_text())\n"
        "u = pd.read_csv(P4 / 'fine_union' / 'sweep_points_union.csv')\n"
        "print({k: sel[k] for k in ['application', 'n_points', 'step_used', 'n_feasible', 'ece_bound', 'soft_flag']})\n"
        "print('winner:', sel['chosen']['name'], sel['chosen']['tag'])\n"
        "fine = u[u.name == 'jt_fine'][['tag','auc_roc','auc_pr','eod','worst_group_ece','worst_group_ece_sd','dir','all_ok']]\n"
        "pd.concat([u[(u.name == 'jt') & (u.tag == 'f0p05_c2_a0')][fine.columns], fine]).round(4).reset_index(drop=True)"
    ),
    code(
        "ok = u[u.all_ok][['name','tag','auc_roc','auc_pr','eod','worst_group_ece','dir']]\n"
        "print('points passing the BASE rule (union of 34):'); ok.round(4)"
    ),
    md(
        "## 2. 4b-1: JT tuned with the same Optuna budget as the other arms (10 trials, 30% subsample, 8 epochs)"
    ),
    code(
        "tu = json.loads((P4B / 'jt_tuning.json').read_text())\n"
        "print('lambdas', tu['lambdas'], '| tuning schedule', tu['tune_schedule'], '| seconds', round(tu['seconds']))\n"
        "pd.DataFrame({'untuned (A2 hp)': tu['untuned_hp (A2)'], 'tuned JT': tu['best_params']})"
    ),
    code(
        "t10 = pd.read_csv(P5 / 'table_10pct.csv')\n"
        "def fmt(r, k): return f\"{r[k]:.4f} [{r[k+'_lo']:.4f}, {r[k+'_hi']:.4f}]\"\n"
        "pick = t10[t10.arm.str.startswith('JT') | t10.arm.isin(['S1', 'S2-EG'])]\n"
        "pd.DataFrame({'arm': pick.arm, 'deployable': pick.deployable,\n"
        "              **{k: [fmt(r, k) for _, r in pick.iterrows()] for k in ['auc_roc','auc_pr','worst_group_ece','eod','dir','recall']}})"
    ),
    code(
        "pdiff = pd.read_csv(P5 / 'paired_diffs.csv')\n"
        "J, T = 'JT chosen = JT-v1 (0.05,2,0)', 'JT tuned (0.05,2,0)'\n"
        "print('Paired bootstrap, tuned JT minus S2-EG, and tuned minus untuned JT (95% CI):')\n"
        "pd.concat([pdiff[(pdiff.A == T) & (pdiff.B == 'S2-EG')], pdiff[(pdiff.A == T) & (pdiff.B == J)]])[\n"
        "    ['A','B','metric','diff_A_minus_B','ci_lo','ci_hi','significant']].round(4).reset_index(drop=True)"
    ),
    md("## 3. 4b-2: extended GRL rescue (spec GRL, fairness and calibration terms off)"),
    code(
        "rs = json.loads((P4B / 'grl_rescue' / 'summary.json').read_text())\n"
        "v = pd.DataFrame(rs['variants']); v['extra'] = v.extra.astype(str); v['seeds'] = v.seeds.astype(str)\n"
        "print('criterion', rs['criterion'], '| compute hours', round(rs['hours'], 2))\n"
        "v[['stage','lambda_a','extra','seeds','probe_mean','auc_roc_mean','probe_drop','auc_loss','works']].round(4)"
    ),
    code(
        "att = pd.DataFrame(json.loads((P4B / 'grl_rescue' / 'attempts.json').read_text()))\n"
        "print('control probe per seed:', att[att.stage == 'control'].set_index('seed').z_probe_lgbm_auc.round(4).to_dict())\n"
        "att.groupby('stage').agg(attempts=('seed', 'size'), max_adv_loss=('max_train_adv_loss', 'max'),\n"
        "                         min_auc_roc=('auc_roc', 'min')).round(3)"
    ),
    md("## 4. Phase 5 at the shared 10% flag rate (deployable = gender-blind at inference)"),
    code(
        "tbl = pd.DataFrame({'arm': t10.arm, 'deployable': t10.deployable,\n"
        "                    **{k: [fmt(r, k) for _, r in t10.iterrows()] for k in ['auc_roc','auc_pr','worst_group_ece','eod','dir','recall']}})\n"
        "tbl"
    ),
    code(
        "d = pdiff[pdiff.A == J][['B','metric','diff_A_minus_B','ci_lo','ci_hi','significant']]\n"
        "print('Paired bootstrap difference, JT chosen minus each comparator (10% flag rate):')\n"
        "d.pivot(index='B', columns='metric', values='diff_A_minus_B').round(4)"
    ),
    code(
        "print('Significant (95% CI excludes 0) differences, JT chosen minus comparator:')\n"
        "d[d.significant].round(4).reset_index(drop=True)"
    ),
    code(
        "h = pdiff[pdiff.A.str.startswith('H') & (pdiff.B == 'S2-EG')]\n"
        "print('Each H arm minus S2-EG (deployable vs deployable):')\n"
        "h.pivot(index='A', columns='metric', values='diff_A_minus_B').round(4)"
    ),
    md("## 5. Matched fairness (target = EOD of JT chosen)"),
    code(
        "mt = pd.read_csv(P5 / 'table_matched.csv')\n"
        "print('target EOD:', round(json.loads((P5 / 's1_note.json').read_text())['targets']['eod_target'], 4))\n"
        "mt[['arm','deployable','knob','flag_rate','eod','eod_gap_to_target','matched(|gap|<=0.005)','recall','precision','auc_pr','worst_group_ece','dir']].round(4)"
    ),
    code(
        "eg = pd.read_csv(P5 / 'eg_eps_curve.csv'); fr = pd.read_csv(P5 / 'flagrate_curve.csv')\n"
        "print('S2-EG: EOD barely moves with eps (cannot reach the target):'); print(eg[['eps','eod','recall','flag_rate']].round(4).to_string(index=False))\n"
        "print('S1 vs flag rate:'); fr[['rate','eod','recall','precision']].round(4).iloc[::4]"
    ),
    code(
        "pm = pd.read_csv(P5 / 'paired_diffs_matched.csv')\n"
        "print('Paired differences at the matched settings (JT chosen minus comparator):')\n"
        "pm[pm.A == 'JT chosen'].pivot_table(index='B', columns='metric', values='diff_A_minus_B').round(4)"
    ),
    code(
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))\n"
        "col = {True: '#3b6ea5', False: '#b03a2e'}\n"
        "for _, r in t10.iterrows():\n"
        "    ax[0].scatter(r.eod, r.recall, color=col[r.deployable], s=45)\n"
        "    ax[0].annotate(r.arm.split(' (')[0].split(' =')[0], (r.eod, r.recall), fontsize=6, xytext=(3, 3), textcoords='offset points')\n"
        "    ax[1].scatter(r.eod, r.worst_group_ece, color=col[r.deployable], s=45)\n"
        "ax[0].set_xlabel('EOD (lower better)'); ax[0].set_ylabel('recall at 10% flag rate')\n"
        "ax[1].set_xlabel('EOD'); ax[1].set_ylabel('worst-group ECE'); ax[1].set_yscale('log')\n"
        "ax[0].set_title('blue = deployable, red = analysis only (uses gender)', fontsize=9)\n"
        "plt.tight_layout(); plt.savefig(FIG / 'posthoc_scatter.png', dpi=150); plt.show()"
    ),
    md(
        "## 6. Why S1 repairs calibration exactly while the learned calibration term is only approximate"
    ),
    code(
        "n = json.loads((P5 / 's1_note.json').read_text())\n"
        "print('class weights', n['class_weights'], '-> logit offset log(w1/w0) =', round(n['prior_shift_logit'], 4))\n"
        "print({k: round(v, 4) for k, v in n.items() if k.endswith('worst_group_ece') and isinstance(v, float)})\n"
        "print('H arms (training for fairness, then the S1 formula):', {k: round(v, 4) for k, v in n['H_worst_group_ece'].items()})\n"
        "pd.DataFrame(n['lambda_c_curve_at_lambda_f_0']).rename(columns={'lambda_c': 'lambda_c (lambda_f = 0)'})"
    ),
    md("## 7. 4b-3: configurations tried per arm (for the paper's disclosure)"),
    code(
        "P2 = ROOT / 'artifacts' / 'runs' / '20261101T055729Z_phase2'\n"
        "rows = []\n"
        "for a in sorted(p.name for p in P2.iterdir() if (p / 'tuning.json').exists()):\n"
        "    rows.append({'arm': a, 'optuna trials': len(json.loads((P2 / a / 'tuning.json').read_text())['trials'])})\n"
        "rows.append({'arm': 'JT tuned (4b-1)', 'optuna trials': len(tu['trials'])})\n"
        "pd.DataFrame(rows)"
    ),
    code(
        "cnt = {name: len(list((P4 / name).glob('*/'))) for name in ['jt', 'jt_conf', 'abl_conf', 'jt_fine']}\n"
        "att4 = json.loads((P4 / 'adversary_rescue' / 'attempts.json').read_text())\n"
        "att4b = json.loads((P4B / 'grl_rescue' / 'attempts.json').read_text())\n"
        "var4b = {(a['stage'], a['lambda_a'], json.dumps(a['extra'], sort_keys=True)) for a in att4b if a['stage'] != 'control'}\n"
        "pd.Series({\n"
        "    'JT lambda points, main sweep (5 x 4)': cnt['jt'],\n"
        "    'JT lambda points, confusion adversary extra sweep': cnt['jt_conf'],\n"
        "    'JT lambda points, fine stage 4b-0': cnt['jt_fine'],\n"
        "    'JT lambda points, ablation-only': cnt['abl_conf'],\n"
        "    'JT lambda points compared by the rule (union)': int(len(u)),\n"
        "    'H arms (lambda_f values)': 3,\n"
        "    'smoke/diagnostic configurations (4 + 3 + 2)': 9,\n"
        "    'Phase 4 adversary rescue variants (a, b, c)': len(att4) - 1,\n"
        "    '4b-2 GRL rescue variants (distinct, seed 0 first)': len({v for v in var4b}),\n"
        "    '4b-2 GRL rescue runs including extra seeds': len(att4b),\n"
        "    'S2-EG eps values refit (Phase 5)': 6,\n"
        "    'S2-EG implementations tried (Phase 2: v1 unmatched, final)': 2,\n"
        "}).to_frame('count')"
    ),
    md("## 8. Test set untouched"),
    code(
        "log = (ROOT / 'artifacts/guard/test_access.jsonl').read_text().strip().splitlines()\n"
        "print(len(log), 'guard-log entry:', json.loads(log[0]).get('purpose'))\n"
        "assert len(log) == 1 and json.loads(log[0]).get('purpose') == 'leakage_probe_gender_only'\n"
        "print('selection rule last changed in:', git('log', '--oneline', '-1', '--', 'configs/selection_rule.yaml'))"
    ),
    md((ROOT / "scripts" / "nb05_what_this_shows.md").read_text(encoding="utf-8")),
]
out = ROOT / "notebooks" / "05_posthoc_comparison.ipynb"
nbf.write(nb, out)
print("wrote", out)
