"""Aggregate JT-A results (validation). python scripts/phase6b_jta_report.py <p6> <p2>"""

import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

p6 = Path(sys.argv[1])
p2 = Path(sys.argv[2])
a2 = (
    json.loads((p2 / "a2" / "summary.json").read_text())
    if (p2 / "a2" / "summary.json").exists()
    else None
)
rows, probes = [], {}
for d in sorted(p6.glob("jta_*/*")):
    name = d.parent.name + " " + d.name
    ss = sorted(d.glob("seed_*"))
    ms = [json.loads((s / "summary.json").read_text()) for s in ss]
    pr = np.array([json.loads((s / "probe.json").read_text())["z_probe_lgbm_auc"] for s in ss])
    probes[name] = pr
    m = pd.DataFrame(ms).rename(columns=lambda c: c[4:] if c.startswith("val_") else c)
    r = {"variant": name, "n": len(ss), "probe": pr.mean(), "probe_sd": pr.std(ddof=1)}
    for k in ("auc_roc", "auc_pr", "eod", "worst_group_ece", "dir"):
        r[k] = m[k].mean()
    rows.append(r)
df = pd.DataFrame(rows)
ref = probes["jta_ctl f0_c0_a0"]
jt = probes["jta_ctl f0p05_c2_a0"]


def t(a, b):
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return d.mean(), h


for i, r in df.iterrows():
    pr = probes[r.variant]
    dj, hj = t(pr, jt)
    dr, hr = t(pr, ref)
    df.loc[i, "drop_vs_JT"], df.loc[i, "drop_vs_JT_ci"] = dj, hj
    df.loc[i, "drop_vs_ctl"], df.loc[i, "drop_vs_ctl_ci"] = dr, hr
a2roc = df.set_index("variant").loc["jta_ctl f0_c0_a0"].auc_roc
df["within_rule"] = (
    (df.eod <= 0.04) & (df.auc_roc >= a2roc - 0.02) & (df.worst_group_ece <= 0.02) & (df.dir >= 0.8)
)
df["lowers_probe_>=0.03_beyond_noise"] = (df.drop_vs_JT - df.drop_vs_JT_ci) >= 0.03
df.to_csv(p6 / "jta_summary.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(4).to_string())
