"""Build notebooks/06_final_verification_and_taiwan.ipynb (thin report: reads saved tables/figures only;
no test-set access)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "fairness-credit", "display_name": "fairness-credit", "language": "python"}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md(
        "# 06 Phase 6: final verification (Home Credit test set, once) and UCI Taiwan retrain\n\n"
        "**What this shows.** Every arm was scored on the held-out test set exactly once (one guard-log entry per "
        "arm; predictions were written to disk at that access and every number below is computed from those saved "
        "files). Operating points were chosen on validation before the test run. Every table marks each arm as "
        "*deployable* (gender-blind at inference) or *analysis only* (uses gender). Mean over seeds with 95% paired "
        "bootstrap CIs (1,000 resamples). The leakage 'after' probe, the SHAP-vector probe, the new-proxy check and "
        "the matched-pair check follow, then the Taiwan retrain with identical hyper-parameters and lambdas."
    ),
    code(
        "import json, sys, subprocess\nfrom pathlib import Path\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import numpy as np, pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image, display\n"
        "pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 200)\n"
        "P6 = ROOT / Path((ROOT / 'artifacts' / 'phase6_rundir.txt').read_text().strip())\n"
        "R = P6 / 'report'\n"
        "git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()\n"
        "print('phase6 run folder:', P6.name); print('HEAD:', git('log', '--oneline', '-1'))"
    ),
    md("## 1. Test-set discipline: guard-log audit"),
    code(
        "audit = json.loads((R / 'guard_audit.json').read_text()); print(json.dumps(audit, indent=1))\n"
        "assert audit['one_entry_per_arm'], 'guard log does not have exactly one entry per arm'"
    ),
    md("## 2. Test results, all arms (deployable vs analysis only; 95% paired-bootstrap CI)"),
    code(
        "t = pd.read_csv(R / 'test_table.csv')\n"
        "t['use'] = np.where(t.deployable, 'deployable', 'analysis only')\n"
        "cols = ['arm','use','n_seeds','auc_roc','auc_pr','worst_group_ece','eod','dir','recall','test_flag_rate']\n"
        "t[cols].round(4)"
    ),
    code(
        "def ci(r, k): return f\"{r[k]:.4f} [{r[k+'_lo']:.4f}, {r[k+'_hi']:.4f}]\"\n"
        "pd.DataFrame({'arm': t.arm, 'use': t.use, 'AUC-PR': t.apply(lambda r: ci(r,'auc_pr'), axis=1),\n"
        "              'EOD': t.apply(lambda r: ci(r,'eod'), axis=1), 'worst-group ECE': t.apply(lambda r: ci(r,'worst_group_ece'), axis=1)})"
    ),
    md("## 3. Pareto view of the deployable arms (test)"),
    code("display(Image(filename=str(R / 'pareto_deployable.png')))"),
    md("## 4. Paired differences on test (JT chosen minus each arm; H1 to H3 and tuned JT versus S2-EG; JT-A versus JT chosen)"),
    code(
        "d = pd.read_csv(R / 'paired_diffs_test.csv')\n"
        "d[d.A == 'JT chosen'].pivot_table(index='B', columns='metric', values='diff_A_minus_B').round(4)"
    ),
    code("d[d.significant & (d.metric.isin(['eod','auc_pr','worst_group_ece']))].round(4).head(60)"),
    md("## 5. Validation vs test: where the selected point and JT were optimistic"),
    code("v = pd.read_csv(R / 'val_vs_test.csv'); v[['arm','deployable','val_auc_pr','test_auc_pr','val_eod','test_eod','val_worst_group_ece','test_worst_group_ece']].round(4)"),
    md("## 6. Leakage 'after': fresh gender probe on frozen z (same protocol as the Phase 1 'before' probe)"),
    code(
        "la = pd.read_csv(R / 'leakage_after.csv'); before = json.loads((ROOT / 'artifacts/leakage/before.json').read_text())\n"
        "print('Phase 1 before: gender from features, test AUC =', round(before['gender_from_features_auc'], 4))\n"
        "la.round(4)"
    ),
    md("## 7. SHAP-vector probe and new-proxy check"),
    code("sp = json.loads((R / 'shap_proxy.json').read_text()); print(json.dumps({k: v for k, v in sp.items() if k != 'before_top15_proxy_features'}, indent=1)[:4000])"),
    md("## 8. Matched-pair check (nearest applicant of the other gender; model never sees gender)"),
    code("pd.read_csv(R / 'matched_pair.csv').round(5)"),
    md("## 9. Retrain checks (REF and GI reproduce their validation predictions)"),
    code("pd.read_csv(R / 'retrain_check.csv')"),
    md("## 10. UCI Taiwan: identical hyper-parameters and lambdas, SEX as gender, no re-tuning"),
    code(
        "TW = P6 / 'taiwan'\n"
        "print(json.dumps(json.loads((TW / 'prep.json').read_text()), indent=1)[:1800])\n"
        "print('gender from features (Taiwan):', json.dumps(json.loads((TW / 'taiwan_leak_before.json').read_text())))"
    ),
    code(
        "tw = pd.read_csv(TW / 'taiwan_results.csv'); tw['use'] = np.where(tw.deployable, 'deployable', 'analysis only')\n"
        "tw[tw.split == 'test'][['arm','use','rate','auc_roc','auc_pr','worst_group_ece','eod','dir','recall']].round(4)"
    ),
    code("pd.read_csv(TW / 'taiwan_zprobe_after.csv').groupby('arm').mean(numeric_only=True).round(4)"),
    code("display(Image(filename=str(ROOT / 'artifacts/figures/taiwan_pareto.png')))"),
    md("## 11. Soft flags and conclusions\n\nSee `artifacts/RUN_LOG.md` (Phase 6 entry) for the full list and the plain-language conclusions."),
    code("print('\\n'.join(l for l in (ROOT / 'artifacts/RUN_LOG.md').read_text(encoding='utf-8').splitlines() if 'SOFT FLAG' in l)[-6000:])"),
]
out = ROOT / "notebooks" / "06_final_verification_and_taiwan.ipynb"
nbf.write(nb, out)
print("wrote", out)
