"""Knowledge-cutoff gate: contamination classification."""

from __future__ import annotations

from datetime import date
from importlib import resources

import pytest

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


def test_registry_load_pins_utf8_not_locale_encoding():
    """Regression: the bundled JSON has non-ASCII (em dashes, Korean notes), so a
    locale-dependent read breaks on cp949 Windows. The file must be genuinely
    non-cp949-decodable (proving the encoding matters) AND load() must still work
    everywhere because it pins UTF-8."""
    raw = resources.files("vts.backtest").joinpath("model_cutoffs.json").read_bytes()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("cp949")            # would be the default on Korean Windows
    # The loader pins utf-8, so it succeeds regardless of the machine's locale.
    assert CutoffRegistry.load().cutoff_for("deepseek-v3") == date(2024, 9, 29)


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


# --- contamination exemption (deterministic non-LLM strategies) ---------------
def test_exempt_model_is_clean_at_any_date():
    """A deterministic non-LLM strategy has no cutoff risk -> CLEAN, never UNKNOWN,
    even though it carries no cutoff date."""
    reg = CutoffRegistry({"m": {"cutoff": None, "verified": False,
                                "contamination_exempt": True}})
    assert reg.is_exempt("m") is True
    # Far past and far future both classify clean — nothing to memorize.
    assert reg.classify("m", date(1990, 1, 1)) == Contamination.CLEAN
    assert reg.classify("m", date(2099, 1, 1)) == Contamination.CLEAN


def test_exempt_segment_status_is_clean_not_unknown():
    """The engine tags folds via segment_status; an exempt model with no cutoff
    must still read CLEAN there (regression: it used to short-circuit UNKNOWN)."""
    reg = CutoffRegistry({"m": {"cutoff": None, "verified": False,
                                "contamination_exempt": True}})
    dates = [date(2024, 1, 1), date(2026, 12, 31)]
    assert reg.segment_status("m", dates) == Contamination.CLEAN


def test_exemption_defaults_off_and_needs_explicit_flag():
    """Absent or falsey -> NOT exempt: the escape hatch can't be entered by default."""
    assert CutoffRegistry({"m": {"cutoff": None, "verified": False}}).is_exempt("m") is False
    assert CutoffRegistry({"m": {"contamination_exempt": False}}).is_exempt("m") is False
    assert CutoffRegistry({}).is_exempt("absent") is False
    # A non-exempt model with no verified cutoff stays UNKNOWN (unchanged behavior).
    assert CutoffRegistry({"m": {"cutoff": None, "verified": False}}).classify(
        "m", date(2024, 7, 1)) == Contamination.UNKNOWN


def test_shipped_registry_gemini_official_jan_2025_cutoffs():
    """Gemini 2.5/3 carry Google-OFFICIAL January-2025 cutoffs; +60d buffer pushes
    the effective cutoff to 2025-04-01, so the clean crypto window starts April 2025.
    gemini-2.0-flash is the older (Aug 2024) deprecated model."""
    reg = CutoffRegistry.load()
    for m in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro"):
        assert reg.cutoff_for(m) == date(2025, 4, 1), m            # 2025-01-31 + 60d
        assert reg.classify(m, date(2025, 3, 15)) == Contamination.CONTAMINATED
        assert reg.classify(m, date(2025, 4, 1)) == Contamination.CONTAMINATED   # boundary
        assert reg.classify(m, date(2025, 4, 2)) == Contamination.CLEAN
    assert reg.cutoff_for("gemini-2.0-flash") == date(2024, 10, 30)  # 2024-08-31 + 60d
    # Newer variants the operator's key exposes are NOT researched -> must stay UNKNOWN.
    assert reg.classify("gemini-3.5-flash", date(2026, 1, 1)) == Contamination.UNKNOWN


def test_shipped_registry_momentum_reference_is_exempt():
    """The bundled deterministic momentum model (model_id 'fake-momentum') ships
    exempt, so offline harness runs certify as clean instead of unknown."""
    reg = CutoffRegistry.load()
    assert reg.is_exempt("fake-momentum") is True
    assert reg.classify("fake-momentum", date(2025, 3, 1)) == Contamination.CLEAN
    assert reg.segment_status("fake-momentum", [date(2025, 3, 1)]) == Contamination.CLEAN
    # The exemption is scoped to that one entry — LLM entries are NOT exempt.
    assert reg.is_exempt("deepseek-v4-pro") is False
