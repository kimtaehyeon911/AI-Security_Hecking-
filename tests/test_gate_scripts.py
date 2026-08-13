"""The go-live runbook's pre-flight window check (scripts/gate/check_window.py).

Exit-code contract (the scripts branch on these):
    0 clean/exempt · 3 contaminated · 4 unknown · 2 bad args.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gate" / "check_window.py"
_spec = importlib.util.spec_from_file_location("gate_check_window", _PATH)
check_window = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_window)  # type: ignore[union-attr]


def run(*argv: str) -> int:
    return check_window.main(list(argv))


def test_clean_v3_window_exits_zero():
    # Entirely after deepseek-v3's effective cutoff (2024-09-29).
    assert run("deepseek-v3", "2024-10-01", "2025-06-30") == 0


def test_contaminated_v3_window_exits_three():
    # Spans the cutoff — one pre-cutoff day taints the whole window.
    assert run("deepseek-v3", "2024-08-01", "2024-12-31") == 3


def test_contaminated_v4_era_window_exits_three():
    assert run("deepseek-v4-pro", "2026-05-01", "2026-06-01") == 3


def test_exempt_momentum_is_clean_any_window():
    assert run("fake-momentum", "2020-01-01", "2099-01-01") == 0


def test_unknown_model_exits_four():
    assert run("gpt-5.5", "2024-01-01") == 4


def test_single_date_defaults_end_to_start():
    assert run("deepseek-v3", "2025-01-01") == 0        # clean
    assert run("deepseek-v3", "2024-09-29") == 3        # boundary is contaminated


@pytest.mark.parametrize("argv", [
    (),                                   # too few
    ("m", "2024-01-01", "x", "y"),        # too many
    ("deepseek-v3", "not-a-date"),        # unparseable
    ("deepseek-v3", "2025-06-30", "2025-01-01"),  # end before start
])
def test_bad_args_exit_two(argv):
    assert run(*argv) == 2
