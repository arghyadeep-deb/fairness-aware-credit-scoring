"""Regenerate Phase 6 Pareto figures after the Taiwan stage, from saved reports only."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from src.eval.phase6_figures import plot_deployable  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1]) / "report"
    plot_deployable(pd.read_csv(out / "test_table.csv"), out)
