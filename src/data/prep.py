"""Stage 1 data preparation for Home Credit (specs/08).

The `Preprocessor` is the single source of truth for feature construction so training and
serving share one code path (CRITICAL_RULES 26). Everything it learns comes from TRAIN only.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import load_config, resolve, seed_everything

log = logging.getLogger(__name__)
MISSING_LABEL = "Missing"


def feature_columns(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Ordered feature groups (numeric to scale, binary, categorical) derived from the config."""
    numeric = list(cfg["numeric_raw"]) + ["AGE_YEARS", "YEARS_EMPLOYED"]
    binary = list(cfg["binary_yn"]) + ["EMPLOYED_MISSING"]
    if cfg["include_ext_source"]:
        numeric += list(cfg["ext_source_columns"])
        binary += [f"{c}_MISSING" for c in cfg["ext_source_columns"]]
    if cfg["add_ratios"]:
        numeric += ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM"]
    return {"numeric": numeric, "binary": binary, "categorical": list(cfg["categorical"])}


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Ratio with zero/NaN denominators -> NaN (imputed later), never inf."""
    den = den.where(den != 0)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def _yn_to_int(s: pd.Series, name: str) -> pd.Series:
    mapped = s.map({"Y": 1, "N": 0, 1: 1, 0: 0})
    if mapped.isna().any():
        raise ValueError(f"Column {name} has values outside Y/N/0/1")
    return mapped.astype("float32")


def derive(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Row-wise feature derivation (no fitting): ages, placeholder fix, flags, ratios."""
    out = pd.DataFrame(index=df.index)
    for c in cfg["numeric_raw"]:
        out[c] = df[c].astype("float64")
    d = float(cfg["days_per_year"])
    out["AGE_YEARS"] = -df[cfg["days_birth"]].astype("float64") / d
    emp = df[cfg["days_employed"]].astype("float64")
    placeholder = emp == cfg["days_employed_placeholder"]
    out["EMPLOYED_MISSING"] = placeholder.astype("float32")
    out["YEARS_EMPLOYED"] = (-emp.mask(placeholder)) / d
    for c in cfg["binary_yn"]:
        out[c] = _yn_to_int(df[c], c)
    if cfg["include_ext_source"]:
        for c in cfg["ext_source_columns"]:
            out[c] = df[c].astype("float64")
            out[f"{c}_MISSING"] = df[c].isna().astype("float32")
    if cfg["add_ratios"]:
        inc = df["AMT_INCOME_TOTAL"].astype("float64")
        out["CREDIT_INCOME_RATIO"] = _safe_ratio(df["AMT_CREDIT"].astype("float64"), inc)
        out["ANNUITY_INCOME_RATIO"] = _safe_ratio(df["AMT_ANNUITY"].astype("float64"), inc)
        out["CREDIT_TERM"] = _safe_ratio(
            df["AMT_CREDIT"].astype("float64"), df["AMT_ANNUITY"].astype("float64")
        )
    for c in cfg["categorical"]:
        out[c] = df[c].astype("object").where(df[c].notna(), MISSING_LABEL)
    return out


@dataclass
class Preprocessor:
    """Fit on train, then transform any split or a single serving record."""

    cfg: dict[str, Any]
    medians: dict[str, float] = field(default_factory=dict)
    clip_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    scaler: StandardScaler | None = None
    encoders: dict[str, dict[str, int]] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, raw_train: pd.DataFrame) -> "Preprocessor":
        self.groups = feature_columns(self.cfg)
        d = derive(raw_train, self.cfg)
        num = self.groups["numeric"]
        clip = self.cfg.get("clip_quantiles")
        for c in num:
            if clip:
                lo, hi = d[c].quantile(clip[0]), d[c].quantile(clip[1])
                self.clip_bounds[c] = (float(lo), float(hi))
                d[c] = d[c].clip(lo, hi)
            self.medians[c] = float(d[c].median())
        filled = d[num].fillna(self.medians)
        self.scaler = StandardScaler().fit(filled.to_numpy(dtype="float64"))
        for c in self.groups["categorical"]:
            cats = sorted(v for v in d[c].unique() if v != MISSING_LABEL)
            self.encoders[c] = {v: i + 1 for i, v in enumerate(cats)}  # 0 = Missing/unseen
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return the model-ready feature frame (numeric+binary float32, categorical int32)."""
        if self.scaler is None:
            raise RuntimeError("Preprocessor is not fitted")
        d = derive(raw, self.cfg)
        num = self.groups["numeric"]
        for c, (lo, hi) in self.clip_bounds.items():
            d[c] = d[c].clip(lo, hi)
        scaled = self.scaler.transform(d[num].fillna(self.medians).to_numpy(dtype="float64"))
        out = pd.DataFrame(scaled.astype("float32"), columns=num, index=raw.index)
        for c in self.groups["binary"]:
            out[c] = d[c].astype("float32")
        for c in self.groups["categorical"]:
            out[c] = d[c].map(self.encoders[c]).fillna(0).astype("int32")
        return out[self.feature_order]

    @property
    def feature_order(self) -> list[str]:
        g = self.groups
        return g["numeric"] + g["binary"] + g["categorical"]

    @property
    def cardinalities(self) -> dict[str, int]:
        return {c: len(m) + 1 for c, m in self.encoders.items()}

    def save(self, out_dir: Path, class_weights: dict[str, float]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, out_dir / "scaler.joblib")

        def dump(name: str, obj: Any) -> None:
            (out_dir / name).write_text(json.dumps(obj, indent=2))

        dump("encoders.json", self.encoders)
        dump("cardinalities.json", self.cardinalities)
        dump("feature_order.json", {**self.groups, "order": self.feature_order})
        dump("class_weights.json", class_weights)
        dump("medians.json", {"medians": self.medians, "clip_bounds": self.clip_bounds})


def load_raw(cfg: dict[str, Any], fraction: float | None = None) -> tuple[pd.DataFrame, int]:
    """Read the CSV, drop XNA gender rows (count returned), optionally subsample for smoke."""
    df = pd.read_csv(resolve(cfg["paths"]["raw_csv"]))
    if df[cfg["id_column"]].duplicated().any():
        raise ValueError("Duplicate SK_ID_CURR in raw data")
    n0 = len(df)
    df = df[~df[cfg["gender_column"]].isin(cfg["gender_drop_values"])]
    dropped = n0 - len(df)
    unknown = set(df[cfg["gender_column"]].unique()) - set(cfg["gender_encoding"])
    if unknown:
        raise ValueError(f"Unexpected gender values: {unknown}")
    if fraction:
        df = df.sample(frac=fraction, random_state=cfg["split"]["seed"])
    return df.reset_index(drop=True), dropped


def split_indices(
    target: np.ndarray, gender: np.ndarray, fractions: dict[str, float], seed: int
) -> dict[str, np.ndarray]:
    """70/15/15 split stratified jointly on TARGET x gender (four cells)."""
    strata = target.astype(int) * 2 + gender.astype(int)
    idx = np.arange(len(target))
    val_test = fractions["val"] + fractions["test"]
    tr, rest = train_test_split(idx, test_size=val_test, stratify=strata, random_state=seed)
    val, te = train_test_split(
        rest, test_size=fractions["test"] / val_test, stratify=strata[rest], random_state=seed
    )
    return {"train": np.sort(tr), "val": np.sort(val), "test": np.sort(te)}


def class_weights(y_train: np.ndarray) -> dict[str, float]:
    """n_total / (2 * n_class), from train only."""
    n = len(y_train)
    return {str(c): float(n / (2 * (y_train == c).sum())) for c in (0, 1)}


def data_dirs(cfg: dict[str, Any], smoke: bool = False) -> tuple[Path, Path]:
    """(processed dir, prep-artifact dir); smoke runs never touch the real ones."""
    if smoke:
        root = resolve(cfg["paths"]["smoke_root"])
        return root / "processed", root / "prep"
    return resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["prep_artifacts"])


def run_prep(cfg: dict[str, Any], smoke: bool = False, force: bool = False) -> dict[str, Any]:
    """Full stage 1: clean, split, fit on train, write parquet + artifacts, return a report."""
    seed_everything(cfg["split"]["seed"])
    proc, prep_dir = data_dirs(cfg, smoke)
    if (prep_dir / "scaler.joblib").exists() and not force and not smoke:
        raise FileExistsError(
            f"{prep_dir} exists. The split is fixed; --force only to rebuild identically."
        )
    raw, dropped = load_raw(cfg, cfg["smoke"]["fraction"] if smoke else None)
    y = raw[cfg["target_column"]].to_numpy()
    a = raw[cfg["gender_column"]].map(cfg["gender_encoding"]).to_numpy()
    idx = split_indices(y, a, cfg["split"]["fractions"], cfg["split"]["seed"])
    pre = Preprocessor(cfg).fit(raw.iloc[idx["train"]])
    proc.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"xna_dropped": dropped, "n_rows": len(raw), "splits": {}}
    for name, ix in idx.items():
        sub = raw.iloc[ix]
        x = pre.transform(sub)
        x[cfg["target_column"]] = y[ix].astype("int8")
        x.reset_index(drop=True).to_parquet(proc / f"{name}.parquet", index=False)
        g = pd.DataFrame(
            {cfg["id_column"]: sub[cfg["id_column"]].to_numpy(), "gender": a[ix].astype("int8")}
        )
        g.to_parquet(proc / f"{name}_gender.parquet", index=False)
        report["splits"][name] = {
            "rows": len(ix),
            "default_rate": float(y[ix].mean()),
            "pct_female": float((a[ix] == 0).mean()),
        }
    report["default_rate_by_gender"] = {"F": float(y[a == 0].mean()), "M": float(y[a == 1].mean())}
    pre.save(prep_dir, class_weights(y[idx["train"]]))
    (prep_dir / "prep_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--smoke", action="store_true", help="5%% of rows, outputs to artifacts/smoke/")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from src.data.checks import gate_report, run_checks

    cfg = load_config(args.config)
    run_prep(cfg, smoke=args.smoke, force=args.force)
    proc, prep = data_dirs(cfg, args.smoke)
    print(gate_report(proc, prep, cfg, run_checks(proc, prep, cfg)))


if __name__ == "__main__":
    main()
