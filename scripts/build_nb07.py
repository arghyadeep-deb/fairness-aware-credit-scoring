"""Build notebooks/07_serving_demo_check.ipynb (thin report: loads the frozen bundle, measures latency live,
shows the manifest, UI check and screenshots). No test-set access."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "fairness-credit", "display_name": "fairness-credit", "language": "python"}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md(
        "# 07 Phase 7: freeze, serving API and demo check\n\n"
        "**What this shows.** The frozen bundle (`artifacts/bundle/v1`, sha256 manifest verified at load), the stateless "
        "API scoring the saved example applicants, measured latency, the end-to-end browser check (headless Edge, "
        "screenshots) and the clean-environment load. The served model is the rule-selected JT (0.05, 2, 0), seed 0, "
        "scored in fp32 on CPU with no post-hoc correction. The dashboard states plainly that on this dataset the "
        "sequential baseline matched or beat the joint model."
    ),
    code(
        "import json, sys, time\nfrom pathlib import Path\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import numpy as np, pandas as pd\nfrom IPython.display import Image, display\n"
        "from fastapi.testclient import TestClient\nfrom api.main import create_app\n"
        "B = ROOT / 'artifacts' / 'bundle' / 'v1'\n"
        "man = json.loads((B / 'manifest.json').read_text())\n"
        "print({k: man[k] for k in man if k not in ('files',)})\nprint(len(man.get('files', {})), 'files hashed')"
    ),
    md("## 1. Load the bundle and score the example applicants"),
    code(
        "c = TestClient(create_app(B)).__enter__()\n"
        "print(c.get('/health').json())\n"
        "ex = c.get('/examples').json(); rows = ex if isinstance(ex, list) else ex.get('examples', ex)\n"
        "res = [c.post('/predict', json=(r.get('record', r))).json() for r in rows]\n"
        "pd.DataFrame([{'probability': round(r['probability'], 4), 'band': r.get('risk_band')} for r in res])"
    ),
    md("## 2. Latency (measured here, 3 warm-up calls discarded)"),
    code(
        "rec = rows[0].get('record', rows[0])\n"
        "for _ in range(3): c.post('/predict', json=rec)\n"
        "t = []\n"
        "for _ in range(100):\n    s = time.perf_counter(); c.post('/predict', json=rec); t.append((time.perf_counter() - s) * 1000)\n"
        "t = np.array(t)\n"
        "s = time.perf_counter(); r = c.post('/explain', json=rec); ex_ms = (time.perf_counter() - s) * 1000\n"
        "print(f'/predict ms: p50 {np.percentile(t, 50):.1f}, p95 {np.percentile(t, 95):.1f} (limit 300)')\n"
        "print(f'/explain ms: {ex_ms:.0f} (predict + explain limit 3000), status {r.status_code}')"
    ),
    md("## 3. Fail-closed check: the verifier rejects a tampered bundle (see tests/test_api.py, 503 MODEL_NOT_READY)"),
    code(
        "import subprocess\n"
        "out = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_api.py', '-q', '-x', '-p', 'no:cacheprovider'], cwd=ROOT, capture_output=True, text=True)\n"
        "print(out.stdout.strip().splitlines()[-1])"
    ),
    md("## 4. End-to-end browser check (headless Edge, `scripts/ui_screenshots.py`)"),
    code(
        "ui = json.loads((ROOT / 'artifacts' / 'ui' / 'ui_check.json').read_text())\n"
        "print(json.dumps(ui, indent=1))\n"
        "for n in ('01_form.png', '02_result.png', '03_dashboard.png'):\n"
        "    print(n); display(Image(filename=str(ROOT / 'artifacts' / 'ui' / n), width=700))"
    ),
    md(
        "## 5. Clean-environment load\n\n"
        "A fresh venv with serving dependencies only (fastapi, uvicorn, pydantic, numpy, pandas, scikit-learn, scipy, "
        "joblib, PyYAML, shap, torch; no lightgbm, fairlearn, mlflow, matplotlib or jupyter) loaded the same bundle and "
        "returned identical probabilities for the 8 example applicants (checked to 6 decimals against the main venv)."
    ),
]
nbf.write(nb, ROOT / "notebooks" / "07_serving_demo_check.ipynb")
print("written")
