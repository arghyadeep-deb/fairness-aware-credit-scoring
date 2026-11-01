"""Standalone figures from saved aggregate metrics; no model or held-out loader."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_deployable(table: pd.DataFrame, out: Path) -> None:
    """Plot means and actual bootstrap interval endpoints, including every deployable JT-A arm."""
    data = table[table.deployable].copy()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, metric, label in zip(
        axes,
        ("auc_pr", "worst_group_ece"),
        ("AUC-PR (higher better)", "Worst-group ECE (lower better)"),
    ):
        for _, row in data.iterrows():
            color = (
                "#b03a2e"
                if row.arm.startswith("JT")
                else ("#2373a5" if row.arm.startswith("S") else "#596666")
            )
            ax.plot(row.eod, row[metric], "o", color=color, markersize=5)
            # Endpoints remain truthful even if a point estimate lies outside its bootstrap CI.
            ax.hlines(row[metric], row.eod_lo, row.eod_hi, color=color, alpha=0.45, linewidth=0.7)
            ax.vlines(
                row.eod,
                row[metric + "_lo"],
                row[metric + "_hi"],
                color=color,
                alpha=0.45,
                linewidth=0.7,
            )
            ax.annotate(
                row.arm,
                (row.eod, row[metric]),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
        values = data[["eod", metric]].to_numpy()
        costs = values.copy()
        if metric == "auc_pr":
            costs[:, 1] *= -1
        frontier = np.array(
            [
                not np.any(np.all(costs <= value, axis=1) & np.any(costs < value, axis=1))
                for value in costs
            ]
        )
        ordered = data.iloc[np.flatnonzero(frontier)].sort_values("eod")
        ax.plot(ordered.eod, ordered[metric], "--", color="#333", alpha=0.6, linewidth=0.8)
        ax.set_xlabel("Equalized-odds difference (lower better)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.15)
        if metric == "worst_group_ece":
            ax.set_yscale("log")
    fig.suptitle(
        "Deployable arms on cached Home Credit test predictions; 95% bootstrap CIs\n"
        "Dashed lines: empirical pairwise Pareto frontiers, not a selected operating point"
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"pareto_deployable.{suffix}", dpi=170)
    plt.close(fig)
