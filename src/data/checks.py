"""Stage 1 data checks (pandera schema + asserts) and the printed GATE table (specs/06, 08)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

SPLITS = ("train", "val", "test")


def load_prepared(proc: Path, prep: Path) -> dict[str, Any]:
    """Read processed parquet + prep metadata (descriptive checks only, no model evaluation)."""
    order = json.loads((prep / "feature_order.json").read_text())
    data = {
        s: {
            "X": pd.read_parquet(proc / f"{s}.parquet"),
            "G": pd.read_parquet(proc / f"{s}_gender.parquet"),
        }
        for s in SPLITS
    }
    return {
        "data": data,
        "order": order,
        "card": json.loads((prep / "cardinalities.json").read_text()),
    }


def build_schema(order: dict[str, Any], card: dict[str, int], target: str) -> pa.DataFrameSchema:
    """No NaN, right dtypes, categorical index within [0, cardinality), binary in {0,1}."""
    cols: dict[str, pa.Column] = {}
    for c in order["numeric"]:
        cols[c] = pa.Column("float32", nullable=False, checks=pa.Check(np.isfinite))
    for c in order["binary"]:
        cols[c] = pa.Column("float32", nullable=False, checks=pa.Check.isin([0.0, 1.0]))
    for c in order["categorical"]:
        cols[c] = pa.Column("int32", nullable=False, checks=pa.Check.in_range(0, card[c] - 1))
    cols[target] = pa.Column("int8", nullable=False, checks=pa.Check.isin([0, 1]))
    return pa.DataFrameSchema(cols, strict=True, ordered=True)


def run_checks(proc: Path, prep: Path, cfg: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Return (name, passed, detail) for every Stage 1 criterion."""
    p = load_prepared(proc, prep)
    data, order, card = p["data"], p["order"], p["card"]
    ck = cfg["checks"]
    target, idcol = cfg["target_column"], cfg["id_column"]
    res: list[tuple[str, bool, str]] = []

    feats = [c for s in SPLITS for c in data[s]["X"].columns]
    leaked = [c for c in feats if c.upper().startswith(("CODE_GENDER", "GENDER", "SEX"))]
    leaked += [c for c in order["order"] if c.upper().startswith(("CODE_GENDER", "GENDER", "SEX"))]
    res.append(("gender absent from X and feature_order", not leaked, str(leaked or "ok")))
    id_in_x = [c for c in feats if c in (idcol, "SK_ID_CURR")]
    res.append(("no ID column in X", not id_in_x, str(id_in_x or "ok")))
    res.append(
        (
            "aligned gender row counts",
            all(len(data[s]["X"]) == len(data[s]["G"]) for s in SPLITS),
            "per split",
        )
    )

    schema = build_schema(order, card, target)
    try:
        for s in SPLITS:
            schema.validate(data[s]["X"])
        res.append(("pandera schema (no NaN, dtypes, cat < cardinality)", True, "3 splits"))
    except pa.errors.SchemaError as exc:
        res.append(("pandera schema (no NaN, dtypes, cat < cardinality)", False, str(exc)[:200]))

    ids = [set(data[s]["G"][idcol]) for s in SPLITS]
    overlap = len(ids[0] & ids[1]) + len(ids[0] & ids[2]) + len(ids[1] & ids[2])
    res.append(("no ID overlap between splits", overlap == 0, f"overlap={overlap}"))

    rates = [data[s]["X"][target].mean() * 100 for s in SPLITS]
    fem = [(data[s]["G"]["gender"] == 0).mean() * 100 for s in SPLITS]
    gap_r, gap_f = max(rates) - min(rates), max(fem) - min(fem)
    res.append(
        ("default-rate gap across splits (pp)", gap_r <= ck["max_split_gap_pp"], f"{gap_r:.3f}")
    )
    res.append(
        ("female-share gap across splits (pp)", gap_f <= ck["max_split_gap_pp"], f"{gap_f:.3f}")
    )

    min_cell = min(
        int(((data[s]["X"][target] == t) & (data[s]["G"]["gender"] == g)).sum())
        for s in SPLITS
        for t in (0, 1)
        for g in (0, 1)
    )
    res.append(("min TARGET x GENDER cell rows", min_cell >= ck["min_cell_rows"], str(min_cell)))

    num = order["numeric"]
    tr = data["train"]["X"][num].astype("float64")
    res.append(
        (
            "train numeric mean ~ 0",
            float(tr.mean().abs().max()) < ck["mean_abs_tol"],
            f"max|mean|={tr.mean().abs().max():.4f}",
        )
    )
    res.append(
        (
            "train numeric std ~ 1",
            float((tr.std(ddof=0) - 1).abs().max()) < ck["std_tol"],
            f"max|std-1|={(tr.std(ddof=0) - 1).abs().max():.4f}",
        )
    )
    const = [c for c in order["order"] if data["train"]["X"][c].nunique() <= 1]
    res.append(("no constant columns (warn only)", True, str(const or "none")))
    return res


def gate_report(
    proc: Path, prep: Path, cfg: dict[str, Any], results: list[tuple[str, bool, str]]
) -> str:
    """Fixed-format Stage 1 GATE text (the only place besides env_check that prints)."""
    p = load_prepared(proc, prep)
    target = cfg["target_column"]
    lines = ["STAGE 1 GATE: DATA PREP", f"{'split':8}{'rows':>9}{'default%':>10}{'female%':>9}"]
    allx, allg = [], []
    for s in SPLITS:
        x, g = p["data"][s]["X"], p["data"][s]["G"]
        lines.append(
            f"{s:8}{len(x):>9}{x[target].mean() * 100:>10.2f}"
            f"{(g['gender'] == 0).mean() * 100:>9.2f}"
        )
        allx.append(x[target])
        allg.append(g["gender"])
    y, a = pd.concat(allx), pd.concat(allg)
    lines.append(f"default rate  F={y[a == 0].mean() * 100:.2f}%  M={y[a == 1].mean() * 100:.2f}%")
    for name, ok, detail in results:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    lines.append("STAGE 1 RESULT: " + ("PASS" if all(ok for _, ok, _ in results) else "FAIL"))
    return "\n".join(lines)
