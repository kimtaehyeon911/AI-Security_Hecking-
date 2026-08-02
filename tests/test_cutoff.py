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


def test_buffer_days_pushes_cutoff_later_never_earlier():
    """Conservatism direction: buffer SHRINKS the clean window (contamination is
    BEFORE the cutoff, so uncertainty must push the boundary later)."""
    reg = CutoffRegistry({"m": {"cutoff": "2026-04-30", "buffer_days": 60, "verified": True}})
    assert reg.cutoff_for("m") == date(2026, 6, 29)                     # +60d
    # A date after the raw cutoff but inside the buffer is still CONTAMINATED.
    assert reg.classify("m", date(2026, 5, 15)) == Contamination.CONTAMINATED
    assert reg.classify("m", date(2026, 6, 29)) == Contamination.CONTAMINATED  # boundary
    assert reg.classify("m", date(2026, 6, 30)) == Contamination.CLEAN


def test_negative_buffer_refused_as_unknown():
    """A negative buffer would WIDEN the clean window — refuse, never permit."""
    reg = CutoffRegistry({"m": {"cutoff": "2026-04-30", "buffer_days": -30, "verified": True}})
    assert reg.classify("m", date(2026, 5, 1)) == Contamination.UNKNOWN


def test_shipped_registry_deepseek_entries_and_direction():
    """The bundled registry carries operator-adopted DeepSeek values (no official
    cutoff exists) using max(sources)+buffer; unresearched models stay UNKNOWN."""
    reg = CutoffRegistry.load()
    # V4: release 2026-04 max + 60d buffer -> mid-2026 dates are contaminated.
    assert reg.classify("deepseek-v4-pro", date(2026, 5, 15)) == Contamination.CONTAMINATED
    assert reg.classify("deepseek-v4-pro", date(2026, 7, 15)) == Contamination.CLEAN
    # V3: conflicting sources resolved to max(Jul 2024)+60d -> Oct 2024 onward clean.
    assert reg.classify("deepseek-v3", date(2024, 8, 15)) == Contamination.CONTAMINATED
    assert reg.classify("deepseek-v3", date(2024, 10, 1)) == Contamination.CLEAN
    # Models with published official cutoffs stay unfilled until read from docs.
    assert reg.classify("gpt-5.5", date(2024, 1, 1)) == Contamination.UNKNOWN
