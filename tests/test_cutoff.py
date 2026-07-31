"""Knowledge-cutoff gate: contamination classification."""

from __future__ import annotations

from datetime import date

from vts.backtest.cutoff import Contamination, CutoffRegistry


def _reg(cutoff="2024-06-01", verified=True):
    return CutoffRegistry({"m": {"cutoff": cutoff, "verified": verified}})


def test_date_after_cutoff_is_clean():
    assert _reg().classify("m", date(2024, 7, 1)) == Contamination.CLEAN


def test_date_on_or_before_cutoff_is_contaminated():
    reg = _reg()
    assert reg.classify("m", date(2024, 6, 1)) == Contamination.CONTAMINATED
    assert reg.classify("m", date(2024, 5, 1)) == Contamination.CONTAMINATED


def test_unverified_or_missing_cutoff_is_unknown():
    assert _reg(verified=False).classify("m", date(2024, 7, 1)) == Contamination.UNKNOWN
    assert _reg(cutoff=None).classify("m", date(2024, 7, 1)) == Contamination.UNKNOWN
    assert CutoffRegistry({}).classify("absent", date(2024, 7, 1)) == Contamination.UNKNOWN


def test_segment_status_contaminated_if_any_date_before():
    reg = _reg()
    dates = [date(2024, 5, 20), date(2024, 7, 1)]
    assert reg.segment_status("m", dates) == Contamination.CONTAMINATED


def test_segment_status_clean_only_if_all_after():
    reg = _reg()
    dates = [date(2024, 6, 2), date(2024, 7, 1)]
    assert reg.segment_status("m", dates) == Contamination.CLEAN


def test_shipped_registry_loads_and_is_unknown_by_default():
    """The bundled registry ships with null/unverified cutoffs -> UNKNOWN (never a guess)."""
    reg = CutoffRegistry.load()
    assert reg.classify("deepseek-v4-pro", date(2024, 1, 1)) == Contamination.UNKNOWN
