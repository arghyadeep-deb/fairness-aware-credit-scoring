from src.data.tensors import load_arrays
from src.utils import load_config
from src.eval.bootstrap import bootstrap_metrics
from src.eval.metrics import METRIC_NAMES

cfg = load_config("configs/train.yaml")
val, _ = load_arrays(cfg, "val")


def boots(base):
    out = []
    for s in [0, 1, 2]:
        d = base / f"seed_{s}"
        out.append(
            bootstrap_metrics(
                val.y, np.load(d / "val_prob.npy"), val.a, np.load(d / "val_flags.npy"), 1000, 0, 15
            )
        )
    return np.nanmean(np.stack(out), axis=0)


jt_b = boots(RUN / sel["chosen"]["name"] / sel["chosen"]["tag"])
eg_b = boots(P2 / "s2_eg")
rows = []
for k in ["auc_roc", "auc_pr", "worst_group_ece", "eod", "dir", "recall"]:
    i = METRIC_NAMES.index(k)
    d = jt_b[:, i] - eg_b[:, i]
    rows.append(
        {
            "metric": k,
            "JT - S2-EG (mean)": d.mean(),
            "95% CI lo": np.percentile(d, 2.5),
            "95% CI hi": np.percentile(d, 97.5),
        }
    )
print("Paired bootstrap difference, JT chosen minus S2-EG (same resamples):")
pd.DataFrame(rows).set_index("metric").round(4)
