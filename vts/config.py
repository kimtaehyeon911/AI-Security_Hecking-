"""Runtime configuration — the chosen environment values, all env-overridable.

These are the "적정값" (reasonable defaults) fixed for this build; every field can
be overridden by an environment variable so nothing is hard-wired. API keys are
NOT stored here — they are read from the environment by the adapters (see
``.env.example``) and must never be committed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from pydantic import BaseModel, Field


def _env_float(name: str, value: str) -> float:
    """Parse a float env var, raising a clear, named error (not a raw ValueError)."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value for {name}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return parsed

# Binance spot profile (user decision, Step 6 follow-up): crypto has no
# fundamentals/filings — the fundamentals analyst degrades to a no-op and the
# thesis rests on price/news/sentiment. Equities remain available via
# VTS_ASSET_CLASS=us_equity + VTS_DATA_SOURCE=alpha_vantage.
_DEFAULT_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class Settings(BaseModel):
    """Immutable runtime settings."""

    model_config = {"frozen": True}

    # --- asset / venue -------------------------------------------------------
    asset_class: str = Field(default="crypto_spot", description="crypto_spot | us_equity | kr_equity")
    data_source: str = Field(default="binance", description="Backtest data vendor.")
    universe: list[str] = Field(default_factory=lambda: list(_DEFAULT_UNIVERSE))

    # --- LLM (Step 2/3 wiring) ----------------------------------------------
    llm_provider: str = "deepseek"
    deep_think_llm: str = "deepseek-v4-pro"
    quick_think_llm: str = "deepseek-v4-flash"
    temperature: float = 0.0
    monthly_budget_usd: float = 30.0

    # --- cadence -------------------------------------------------------------
    cadence: str = Field(default="1d", description="Decision cadence: 1d | 4h.")

    # --- storage -------------------------------------------------------------
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".vts")

    @property
    def store_path(self) -> Path:
        return self.data_dir / "pit_store.sqlite"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from ``VTS_*`` environment variables (falling back to defaults)."""
        env = os.environ
        raw: dict[str, object] = {}
        if "VTS_ASSET_CLASS" in env:
            raw["asset_class"] = env["VTS_ASSET_CLASS"]
        if "VTS_DATA_SOURCE" in env:
            raw["data_source"] = env["VTS_DATA_SOURCE"]
        if env.get("VTS_UNIVERSE"):
            raw["universe"] = [s.strip().upper() for s in env["VTS_UNIVERSE"].split(",") if s.strip()]
        if "VTS_LLM_PROVIDER" in env:
            raw["llm_provider"] = env["VTS_LLM_PROVIDER"]
        if "VTS_DEEP_THINK_LLM" in env:
            raw["deep_think_llm"] = env["VTS_DEEP_THINK_LLM"]
        if "VTS_QUICK_THINK_LLM" in env:
            raw["quick_think_llm"] = env["VTS_QUICK_THINK_LLM"]
        if "VTS_TEMPERATURE" in env:
            raw["temperature"] = _env_float("VTS_TEMPERATURE", env["VTS_TEMPERATURE"])
        if "VTS_MONTHLY_BUDGET_USD" in env:
            raw["monthly_budget_usd"] = _env_float(
                "VTS_MONTHLY_BUDGET_USD", env["VTS_MONTHLY_BUDGET_USD"]
            )
        if "VTS_CADENCE" in env:
            raw["cadence"] = env["VTS_CADENCE"]
        if "VTS_DATA_DIR" in env:
            raw["data_dir"] = Path(env["VTS_DATA_DIR"])
        return cls(**raw)


def load_settings() -> Settings:
    """Convenience loader used by entry points."""
    return Settings.from_env()
