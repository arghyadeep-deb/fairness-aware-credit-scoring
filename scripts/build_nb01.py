"""Build notebooks/01_data_and_leakage_before.ipynb (thin report notebook; logic lives in src/)."""

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
        '# 01 Data preparation and proxy leakage ("before")\n\n'
        "**Goal:** show the fixed 70/15/15 split (stratified on TARGET x gender) and measure how well "
        "gender can be predicted from the features alone, even though gender is not a feature.\n"
        "Results are loaded from `artifacts/`; nothing is trained here."
    ),
    code(
        "import json, sys\nfrom pathlib import Path\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n\n"
        "import pandas as pd\nimport matplotlib.pyplot as plt\n"
        "from src.utils import load_config, resolve\n\n"
        "cfg = load_config('configs/data.yaml')\n"
        "prep = json.loads((resolve(cfg['paths']['prep_artifacts']) / 'prep_report.json').read_text())\n"
        "before = json.loads((resolve(cfg['paths']['leakage_artifacts']) / 'before.json').read_text())\n"
        "print('XNA gender rows dropped:', prep['xna_dropped'], '| rows kept:', prep['n_rows'])"
    ),
    md("## Split table"),
    code(
        "split = pd.DataFrame(prep['splits']).T\n"
        "split['default_rate_%'] = (split['default_rate'] * 100).round(2)\n"
        "split['female_%'] = (split['pct_female'] * 100).round(2)\n"
        "split[['rows', 'default_rate_%', 'female_%']].astype({'rows': int})"
    ),
    code(
        "dg = prep['default_rate_by_gender']\n"
        "pd.Series({'F': f\"{dg['F']*100:.2f}%\", 'M': f\"{dg['M']*100:.2f}%\"}, name='default rate by gender')"
    ),
    code(
        "fig, ax = plt.subplots(1, 2, figsize=(9, 3))\n"
        "split['default_rate_%'].plot.bar(ax=ax[0], color='#3b6ea5', title='Default rate % per split')\n"
        "split['female_%'].plot.bar(ax=ax[1], color='#8a6fb0', title='Female share % per split')\n"
        "for a in ax: a.set_ylim(0, a.get_ylim()[1] * 1.2); a.tick_params(axis='x', rotation=0)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## Gender predicted from features only (TARGET never used)"),
    code(
        "aucs = pd.DataFrame({\n"
        "    'val AUC': [before['lightgbm']['val_auc'], before['logistic']['val_auc'], before['shuffled_control']['val_auc']],\n"
        "    'test AUC': [before['lightgbm']['test_auc'], before['logistic']['test_auc'], before['shuffled_control']['test_auc']],\n"
        "}, index=['LightGBM probe', 'Logistic probe', 'Shuffled-gender control']).round(4)\n"
        "print('Soft flag (premise weak, AUC < 0.60):', before['soft_flag_premise_weak'])\n"
        "print('Shuffled control within 0.5 +/- 0.02:', before['shuffled_control']['within_tolerance'])\n"
        "aucs"
    ),
    md("## Top-15 proxy features (LightGBM gain)"),
    code(
        "from IPython.display import Image\n"
        "Image(str(resolve(cfg['paths']['leakage_artifacts']) / 'top15_proxy_features_before.png'))"
    ),
    code(
        "pd.DataFrame(before['top_features']).assign(gain_share=lambda d: d.gain_share.round(4)).set_index('feature')"
    ),
    md(
        "## What this shows\n\n"
        "- The split is stable: default rate and female share match across train/val/test to well under 0.5 percentage points, and every TARGET x gender cell is large.\n"
        "- Gender is excluded from X, yet a gradient-boosted probe recovers it from the remaining features with a high AUC. Features such as occupation, organisation and car ownership act as proxies, so dropping the column does not remove gender information.\n"
        "- The shuffled-label control sits at 0.5, so the high AUC is real signal and not a pipeline artefact.\n"
        '- This is the "before" number. Phase 6 probes the learned representation z to see whether the adversarial term reduces it.'
    ),
]
out = ROOT / "notebooks" / "01_data_and_leakage_before.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
