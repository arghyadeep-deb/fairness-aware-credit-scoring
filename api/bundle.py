"""Load and verify the frozen bundle (artifacts/bundle/<version>/). Fails closed on any mismatch."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml

from src.data.prep import Preprocessor, feature_columns
from src.models.nets import build_model

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "bundle" / "v1"


class BundleError(RuntimeError):
    """Bundle missing, incomplete or tampered: the API answers 503 MODEL_NOT_READY."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_sha(files: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{k}:{files[k]}\n" for k in sorted(files)).encode()).hexdigest()


def verify(root: Path) -> dict[str, Any]:
    """Check manifest.json and the sha256 of EVERY listed file; unlisted extra files are an error."""
    mf = root / "manifest.json"
    if not mf.is_file():
        raise BundleError(f"manifest.json not found in {root}")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    listed = manifest["files"]
    for rel, digest in listed.items():
        p = root / rel
        if not p.is_file():
            raise BundleError(f"missing bundle file: {rel}")
        if sha256_file(p) != digest:
            raise BundleError(f"hash mismatch: {rel}")
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {
        "manifest.json"
    }
    if on_disk - set(listed):
        raise BundleError(f"unlisted files in bundle: {sorted(on_disk - set(listed))}")
    if bundle_sha(listed) != manifest["bundle_sha"]:
        raise BundleError("bundle_sha mismatch")
    return manifest


@dataclass
class Bundle:
    root: Path
    manifest: dict[str, Any]
    pre: Preprocessor
    model: torch.nn.Module
    spec: dict[str, Any]
    bg_cont: np.ndarray
    bg_cat: np.ndarray
    pool_cont: np.ndarray
    pool_cat: np.ndarray
    pool_a: np.ndarray
    docs: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def version(self) -> str:
        return self.manifest["model_version"]

    @property
    def sha(self) -> str:
        return self.manifest["bundle_sha"]


def load_bundle(root: Path | str | None = None) -> Bundle:
    root = Path(root or os.environ.get("CREDIT_BUNDLE_DIR") or DEFAULT_DIR)
    manifest = verify(root)
    spec = json.loads((root / "model/model_spec.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((root / "prep/data_cfg.yaml").read_text(encoding="utf-8"))
    pre = Preprocessor(cfg)
    order = json.loads((root / "prep/feature_order.json").read_text(encoding="utf-8"))
    pre.groups = {k: order[k] for k in ("numeric", "binary", "categorical")}
    if pre.groups != feature_columns(cfg):
        raise BundleError("feature groups differ from config")
    meds = json.loads((root / "prep/medians.json").read_text(encoding="utf-8"))
    pre.medians = meds["medians"]
    pre.clip_bounds = {k: (float(v[0]), float(v[1])) for k, v in meds["clip_bounds"].items()}
    pre.scaler = joblib.load(root / "prep/scaler.joblib")
    pre.encoders = json.loads((root / "prep/encoders.json").read_text(encoding="utf-8"))
    model = build_model(
        "tabtransformer", spec["n_cont"], spec["cards"], spec["model_defaults"], spec["hp"]
    )
    model.load_state_dict(
        torch.load(root / "model/backbone.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    bg = np.load(root / "shap_background.npz")
    pool = np.load(root / "matched_pool.npz")
    docs = {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (root / "json").glob("*.json")
    }
    examples = json.loads((root / "examples.json").read_text(encoding="utf-8"))["examples"]
    return Bundle(root, manifest, pre, model, spec, bg["cont"], bg["cat"], pool["cont"], pool["cat"],
                  pool["a"], docs, examples)  # fmt: skip
