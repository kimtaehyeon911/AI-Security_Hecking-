"""Shared test helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> datetime:
    """Construct a tz-aware UTC datetime succinctly."""
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)
