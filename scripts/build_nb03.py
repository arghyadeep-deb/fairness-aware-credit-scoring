"""Build notebooks/03_joint_model.ipynb (thin report notebook; logic lives in src/)."""

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
        "# 03 Joint objective: building blocks and unit-test evidence\n\n"
        "**Goal:** show that each piece of `L = L_wCE + lambda_f L_fair + lambda_c L_cal + lambda_a L_adv` "
        "behaves as specified (GRL, soft fairness, soft calibration, warm-up/ramp, lambda = 0 equals A2).\n"
        "No model is trained here beyond tiny demos; joint training runs start in Phase 4."
    ),
    code(
        "import json, subprocess, sys\nfrom pathlib import Path\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n\n"
        "import numpy as np, pandas as pd, torch\nimport matplotlib.pyplot as plt\n"
        "from src.utils import load_config, resolve, seed_everything\n"
        "from src.losses.grl import grad_reverse\n"
        "from src.losses.fairness import fairness_loss\n"
        "from src.losses.calibration import soft_group_ece\n"
        "from src.losses.joint import JointConfig, JointLoss\n"
        "from src.losses.weighted_ce import weighted_bce\n"
        "from src.models.heads import JointModel\nfrom src.models.nets import build_model\n"
        "from src.eval.metrics import ece_equal_freq, compute_metrics\n"
        "from src.data.tensors import load_arrays\n\n"
        "cfg = load_config('configs/train.yaml'); seed_everything(0)\n"
        "W = json.loads((resolve(cfg['paths']['prep_artifacts']) / 'class_weights.json').read_text())\n"
        "print('class weights', {k: round(v, 3) for k, v in W.items()})"
    ),
    md("## 1. Gradient reversal: identity forward, gradient x (-lambda) backward"),
    code(
        "rows = []\n"
        "for lam in [0.0, 0.5, 1.0, 2.0]:\n"
        "    x = torch.randn(64, 5, requires_grad=True); w = torch.randn(64, 5)\n"
        "    out = grad_reverse(x, lam); (out * w).sum().backward()\n"
        "    rows.append({'lambda': lam, 'forward identical': bool(torch.equal(out, x)),\n"
        "                 'max |grad + lambda*w|': float((x.grad + lam * w).abs().max())})\n"
        "pd.DataFrame(rows).set_index('lambda')"
    ),
    md("## 2. Warm-up and lambda ramp (3 warm-up epochs, then linear over 5)"),
    code(
        "jc = JointConfig.from_config(cfg, lambda_f=2.0, lambda_c=1.0, lambda_a=0.5)\n"
        "loss = JointLoss(jc, W['0'], W['1']); eps = np.linspace(0, 12, 121); lam = []\n"
        "for e in eps: loss.set_epoch(e); lam.append(loss.lambdas())\n"
        "lam = pd.DataFrame(lam, index=eps)\n"
        "ax = lam.plot(figsize=(6.5, 3.3), drawstyle='steps-post')\n"
        "ax.axvspan(0, 3, color='grey', alpha=0.15, label='warm-up'); ax.set_xlabel('epoch'); ax.set_ylabel('effective lambda')\n"
        "ax.legend(['lambda_f', 'lambda_c', 'lambda_a', 'warm-up']); plt.tight_layout(); plt.show()"
    ),
    md("## 3. Soft fairness loss tracks the hard flag-rate / equalized-odds gaps"),
    code(
        "g = torch.Generator().manual_seed(1); n = 40000\n"
        "y = (torch.rand(n, generator=g) < 0.2).float(); a = (torch.rand(n, generator=g) < 0.5).long()\n"
        "base = 2.0 * y + torch.randn(n, generator=g); rows = []\n"
        "for shift in [0, 0.25, 0.5, 1.0, 1.5, 2.0]:\n"
        "    s = base + shift * a; thr = torch.quantile(s, 0.90); fl = (s > thr).numpy()\n"
        "    m = compute_metrics(y.numpy().astype(int), torch.sigmoid(s).numpy(), a.numpy(), fl)\n"
        "    rows.append({'group score shift': shift, 'hard DPD': m['dpd'], 'soft dp': fairness_loss(s, y, a, 'dp').item(),\n"
        "                 'hard EOD': m['eod'], 'hard |dTPR|+|dFPR|': abs(m['tpr_m'] - m['tpr_f']) + abs(m['fpr_m'] - m['fpr_f']),\n"
        "                 'soft eo': fairness_loss(s, y, a, 'eo').item()})\n"
        "fair_tbl = pd.DataFrame(rows).set_index('group score shift').round(4); fair_tbl"
    ),
    code(
        "ax = fair_tbl[['hard DPD', 'soft dp', 'hard |dTPR|+|dFPR|', 'soft eo']].plot(marker='o', figsize=(6.5, 3.4))\n"
        "ax.set_ylabel('gap'); plt.tight_layout(); plt.show()"
    ),
    md("## 4. Soft group ECE tracks hard ECE and is zero for calibrated scores"),
    code(
        "n = 200000; p = torch.rand(n, generator=g) * 0.9 + 0.05; yy = (torch.rand(n, generator=g) < p).float()\n"
        "aa = (torch.rand(n, generator=g) < 0.5).long(); lg0 = torch.log(p / (1 - p)); rows = []\n"
        "for shift in [0, 0.25, 0.5, 1.0, 2.0]:\n"
        "    lg = lg0 + shift\n"
        "    rows.append({'logit shift': shift, 'hard ECE': ece_equal_freq(yy.numpy(), torch.sigmoid(lg).numpy()),\n"
        "                 'soft group ECE': soft_group_ece(lg, yy, aa).item()})\n"
        "pd.DataFrame(rows).set_index('logit shift').round(4)"
    ),
    md(
        "## 5. All lambdas = 0: joint loss equals the A2 loss (real validation batch, real class weights)"
    ),
    code(
        "val, cards = load_arrays(cfg, 'val')\n"
        "ix = np.random.default_rng(0).choice(len(val.y), 2048, replace=False)\n"
        "xc, xk = torch.from_numpy(val.cont[ix]), torch.from_numpy(val.cat[ix])\n"
        "yb, ab = torch.from_numpy(val.y[ix]).float(), torch.from_numpy(val.a[ix]).long()\n"
        "model = JointModel(build_model('tabtransformer', xc.shape[1], cards, cfg['model_defaults'], {})).eval()\n"
        "logit, _, adv = model(xc, xk); a2 = weighted_bce(logit, yb, W['0'], W['1']); rows = []\n"
        "for epoch in [0, 4, 20]:\n"
        "    jt = JointLoss(JointConfig.from_config(cfg), W['0'], W['1']); jt.set_epoch(epoch)\n"
        "    tot, parts = jt(logit, yb, ab, adv)\n"
        "    rows.append({'epoch': epoch, 'A2 loss': a2.item(), 'joint loss (lambdas=0)': tot.item(), 'difference': (tot - a2).item(), 'terms computed': list(parts)})\n"
        "pd.DataFrame(rows).set_index('epoch')"
    ),
    md("## 6. Adversary is inactive during warm-up, active afterwards"),
    code(
        "rows = []\n"
        "for epoch in [1, 6, 20]:\n"
        "    jt = JointLoss(JointConfig.from_config(cfg, lambda_f=1.0, lambda_c=1.0, lambda_a=1.0), W['0'], W['1'])\n"
        "    jt.set_epoch(epoch); model.zero_grad(); logit, _, adv = model(xc, xk)\n"
        "    tot, parts = jt(logit, yb, ab, adv); tot.backward()\n"
        "    head = sum(float(p.grad.abs().sum()) for p in model.adversary.parameters() if p.grad is not None)\n"
        "    rows.append({'epoch': epoch, 'effective lambda_a': jt.lambdas()['a'], 'terms computed': list(parts), 'adversary-head |grad| sum': head})\n"
        "pd.DataFrame(rows).set_index('epoch')"
    ),
    md("## 7. A 200-row batch is memorised (wCE + fairness + calibration terms on)"),
    code(
        "torch.manual_seed(0); r = np.random.default_rng(0); sub = r.choice(len(val.y), 200, replace=False)\n"
        "xc2, xk2 = torch.from_numpy(val.cont[sub]), torch.from_numpy(val.cat[sub])\n"
        "y2, a2_ = torch.from_numpy(val.y[sub]).float(), torch.from_numpy(val.a[sub]).long()\n"
        "m2 = JointModel(build_model('mlp', xc2.shape[1], cards, cfg['model_defaults'], {'dropout': 0.0}))\n"
        "jc7 = JointConfig.from_config(cfg, lambda_f=1.0, lambda_c=1.0, lambda_a=0.0); jc7.warmup_epochs = 0; jc7.ramp_epochs = 0\n"
        "jt = JointLoss(jc7, W['0'], W['1'])\n"
        "jt.set_epoch(0); opt = torch.optim.Adam(m2.parameters(), lr=5e-3); curve = []\n"
        "for _ in range(600):\n"
        "    opt.zero_grad(); lg, _, ad = m2(xc2, xk2); tot, parts = jt(lg, y2, a2_, ad); tot.backward(); opt.step(); curve.append(tot.item())\n"
        "plt.figure(figsize=(6, 3)); plt.semilogy(curve); plt.xlabel('step'); plt.ylabel('joint loss'); plt.tight_layout(); plt.show()\n"
        "print('start %.4f -> end %.5f' % (curve[0], curve[-1]))"
    ),
    md("## 8. Stage 4 unit tests (live run)"),
    code(
        "r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_joint.py', '-q', '-p', 'no:warnings'],\n"
        "                   cwd=ROOT, capture_output=True, text=True)\n"
        "print(r.stdout.strip().splitlines()[-1]); assert r.returncode == 0"
    ),
    md((ROOT / "scripts" / "nb03_what_this_shows.md").read_text(encoding="utf-8")),
]
out = ROOT / "notebooks" / "03_joint_model.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
