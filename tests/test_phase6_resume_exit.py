"""phase6_resume treats ANY non-zero exit as failure and retries exactly once."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("phase6_resume", Path("scripts/phase6_resume.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.mark.parametrize("code", [1, 2, -9, -1073741819, 3221225477, 137])
def test_nonzero_exit_is_failure(code):
    assert mod.attempt_failed(code, True)


def test_zero_exit_without_done_is_failure_and_success_otherwise():
    assert mod.attempt_failed(0, False)
    assert not mod.attempt_failed(0, True)


def test_retry_once_only():
    assert mod.may_retry(0) and not mod.may_retry(1)
