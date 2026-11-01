"""Run unfinished cached-test analyses strictly sequentially, with RSS evidence per attempt."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config  # noqa: E402


def attempt_failed(native_exit: int, done_exists: bool) -> bool:
    """ANY non-zero exit (negative signals, positive Windows codes such as 3221225477) is a failure."""
    return native_exit != 0 or not done_exists


def may_retry(attempt: int) -> bool:
    """One retry, with the smaller SHAP workload; the second failure is a hard stop."""
    return attempt == 0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_log(text: str) -> None:
    with (ROOT / "artifacts/RUN_LOG.md").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def monitor(command: list[str], destination: Path, env: dict[str, str], poll: float) -> dict:
    """Wait for this process to EXIT; retain its sampled RSS high-water mark and native status."""
    started = time.monotonic()
    peak = 0
    peak_tree = 0
    with destination.with_suffix(".log").open("w", encoding="utf-8") as output:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss = process.memory_info().rss
                peak = max(peak, rss)
                tree = rss
                for descendant in process.children(recursive=True):
                    try:
                        tree += descendant.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_tree = max(peak_tree, tree)
            except psutil.NoSuchProcess:
                pass
            destination.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "pid": child.pid,
                        "peak_rss_bytes": peak,
                        "peak_tree_rss_bytes": peak_tree,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            time.sleep(poll)
        native_exit = child.wait()
    result = {
        "command": command,
        "pid": child.pid,
        "native_exit": native_exit,
        "peak_rss_bytes": peak,
        "peak_tree_rss_bytes": peak_tree,
        "elapsed_seconds": time.monotonic() - started,
    }
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    cfg = load_config("configs/phase6_memory.yaml")
    p2 = Path(load_config("configs/train.yaml")["joint"]["hp_source"]).parents[1]
    p4 = Path((ROOT / "artifacts/phase4_rundir.txt").read_text().strip())
    p4b = Path((ROOT / "artifacts/phase4b_rundir.txt").read_text().strip())
    p6 = ROOT / (ROOT / "artifacts/phase6_rundir.txt").read_text().strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = p6 / ("memory_resume_" + stamp)
    run.mkdir(exist_ok=False)
    guard = ROOT / "artifacts/guard/test_access.jsonl"
    before = digest(guard)
    selection_before = digest(ROOT / "configs/selection_rule.yaml")
    env = dict(os.environ)
    env.pop("ALLOW_TEST", None)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        env[key] = str(cfg["cpu_threads"])
    # Explicit predeclared names; no invented optional JT-W or JT-A+PH arms.
    arms = ["JT chosen", "JT tuned", "H1", "H2", "H3"] + [
        f"JT-A k{k} la{la}" for k in (3, 5) for la in (0.1, 0.25, 0.5)
    ]
    records = []
    append_log(
        f"\n## Phase 6 memory-safe resume {stamp}\n"
        "Owner requested sequential one-arm processes; no new Home Credit test access. "
        "All done.json arms preserved. Threads/jobs4, float32, SHAP100background/4000train/"
        "5000cachedtest/batch500 (seed0 only), bootstrap100-resample blocks. "
        "Earlier completed arms retain their original4000test-row SHAP protocol; "
        "mixed sample counts are a documented deviation. JT-W/JT-A+PH undeclared, not run. "
        "Two earlier analyze kills were memory stops; incomplete attempt outputs are archived.\n"
    )
    try:
        for arm in arms:
            slug = arm.replace(" ", "_")
            output = p6 / "test" / slug
            if (output / "done.json").exists():
                append_log(f"- Skip `{arm}`: existing done.json, unchanged.")
                records.append({"arm": arm, "status": "skipped_done"})
                continue
            if not (ROOT / "artifacts/test_preds" / slug / "access_done.json").exists():
                raise RuntimeError(f"Missing cached predictions for {arm}; never re-access test")
            for attempt, rows in enumerate((cfg["shap_test_rows"], cfg["shap_fallback_test_rows"])):
                if output.exists():
                    archive = run / (slug + f"_incomplete_before_attempt{attempt}")
                    assert output.resolve().is_relative_to(p6.resolve())
                    assert archive.resolve().is_relative_to(run.resolve())
                    output.rename(archive)  # Preserve every partial result instead of overwriting.
                command = [
                    sys.executable,
                    "scripts/phase6_test.py",
                    "analyze",
                    str(p2),
                    str(p4),
                    str(p4b),
                    str(p6.relative_to(ROOT)),
                    arm,
                    "--shap-test-rows",
                    str(rows),
                ]
                destination = run / (slug + f"_attempt{attempt}")
                append_log(f"- START `{arm}` attempt{attempt}: SHAP test rows={rows}.")
                result = monitor(command, destination, env, cfg["memory_poll_seconds"])
                result.update(arm=arm, shap_test_rows=rows, attempt=attempt)
                records.append(result)
                append_log(
                    f"- EXIT `{arm}` attempt{attempt}: native={result['native_exit']}; "
                    f"psutil peak RSS={result['peak_rss_bytes']}bytes; "
                    f"peak process-tree RSS={result['peak_tree_rss_bytes']}bytes; "
                    f"wall={result['elapsed_seconds']}s; SHAP test rows={rows}."
                )
                (run / "arms.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                assert digest(guard) == before, "Home Credit test access log changed: hard stop"
                assert digest(ROOT / "configs/selection_rule.yaml") == selection_before
                gc.collect()
                if not attempt_failed(result["native_exit"], (output / "done.json").exists()):
                    break
                if not may_retry(attempt):
                    raise RuntimeError(
                        f"Arm {arm} failed, native={result['native_exit']}; see {destination}"
                    )
                append_log(
                    f"- DEVIATION `{arm}`: non-zero exit (assumed memory); retry in a fresh process with 2000 cached SHAP test rows."
                )
            else:
                raise RuntimeError(f"No successful analysis for {arm}")
        (run / "complete.json").write_text(
            json.dumps(
                {
                    "guard_sha256_unchanged": before,
                    "selection_rule_sha256_unchanged": selection_before,
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        append_log(
            "- Remaining arm analysis complete; original Home Credit guard and selection rule hashes unchanged."
        )
    except BaseException as exc:
        append_log(f"- HARD STOP in memory-safe resume: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
