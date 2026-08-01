"""CLI smoke tests — every command exercised offline (stubbed source, tmp data dir)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import utc

from vts import cli
from vts.pit.schema import OHLCVBar
from vts.pit.store import PointInTimeStore
from vts.sources.fake import InMemorySource


def _bars(symbol, start, n, p0, daily):
    out, price = [], p0
    for i in range(n):
        t = (start + timedelta(days=i)).replace(hour=21)
        out.append(OHLCVBar(symbol=symbol, event_time=t, knowledge_time=t, source="t",
                            open=price, high=price, low=price, close=price,
                            volume=1_000_000))
        price *= 1.0 + daily
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VTS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VTS_UNIVERSE", "UP,DN")
    monkeypatch.delenv("VTS_KILL_SWITCH", raising=False)
    return tmp_path


@pytest.fixture
def stub_source(monkeypatch):
    source = InMemorySource(
        ohlcv=_bars("UP", utc(2025, 1, 1), 60, 100.0, +0.01)
        + _bars("DN", utc(2025, 1, 1), 60, 100.0, -0.01),
    )
    monkeypatch.setattr(cli, "_make_source", lambda settings: source)
    return source


def test_ingest_then_status(env, stub_source, capsys):
    assert cli.main(["ingest", "--start", "2025-01-01", "--end", "2025-03-01"]) == 0
    out = capsys.readouterr().out
    assert "UP: ohlcv=59" in out

    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "UP: 59 bars" in out


def test_backtest_writes_report_and_gates(env, stub_source, tmp_path, capsys):
    cli.main(["ingest", "--start", "2025-01-01", "--end", "2025-03-01"])
    capsys.readouterr()
    report_path = tmp_path / "report.md"
    rc = cli.main(["backtest", "--start", "2025-02-01", "--end", "2025-02-25",
                   "--report", str(report_path)])
    text = report_path.read_text(encoding="utf-8")
    assert "## Gate:" in text and "CAGR" in text and "60/40" in text
    assert rc in (0, 1)                       # exit code IS the gate verdict


def test_paper_runs_and_logs_shortfall(env, stub_source, capsys):
    cli.main(["ingest", "--start", "2025-01-01", "--end", "2025-03-01"])
    capsys.readouterr()
    rc = cli.main(["paper", "--start", "2025-02-01", "--end", "2025-02-20"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "processed" in out and "implementation shortfall" in out
    assert (env / "paper_state.json").exists()


def test_reset_halt_paper(env, stub_source, capsys):
    cli.main(["ingest", "--start", "2025-01-01", "--end", "2025-03-01"])
    cli.main(["paper", "--start", "2025-02-01", "--end", "2025-02-10"])
    capsys.readouterr()
    # Latch a halt by hand, then clear it through the CLI.
    from vts.paper.state import PaperState

    path = env / "paper_state.json"
    st = PaperState.load(path)
    st.halted = True
    st.halt_reasons.append("daily_loss_limit: test")
    st.save(path)

    assert cli.main(["reset-halt", "--scope", "paper", "--operator", "ops@test"]) == 0
    st2 = PaperState.load(path)
    assert st2.halted is False
    assert any("reset by ops@test" in r for r in st2.halt_reasons)


def test_live_dry_run_cycle_with_simulated_broker(env, stub_source, monkeypatch, capsys):
    cli.main(["ingest", "--start", "2025-01-01", "--end", "2025-03-01"])
    capsys.readouterr()
    from vts.live.broker import SimulatedBroker

    monkeypatch.setattr("vts.live.binance_broker.BinanceBroker",
                        lambda *a, **k: SimulatedBroker(
                            cash=1_000_000.0,
                            prices={"UP": 100.0, "DN": 100.0}))
    rc = cli.main(["live", "--date", "2025-02-10", "--capital", "100"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out and "submitted=0" in out


def test_backtest_without_store_fails_cleanly(env, capsys):
    rc = cli.main(["backtest", "--start", "2025-02-01", "--end", "2025-02-25"])
    assert rc == 2
    assert "ingest" in capsys.readouterr().err
