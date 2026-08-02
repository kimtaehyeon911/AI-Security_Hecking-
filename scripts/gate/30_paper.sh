#!/usr/bin/env bash
# Gate 3 — paper trading (same code path as backtest & live). Forward data is
# definitionally clean, so this is where a V4-era model is validated. Run it
# DAILY for >= 8 weeks; the loop is restart-safe and idempotent. The gate to
# watch is the implementation-shortfall gap: if it keeps widening, the blocker
# is execution friction, not the model — do NOT go live.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_py
load_dotenv

say "Gate 3: paper loop ($MODEL)  [$PAPER_START .. $PAPER_END]"
info "run this daily; state persists in $VTS_DATA_DIR/paper_state.json"

"$PY" -m vts paper --model "$MODEL" --start "$PAPER_START" --end "$PAPER_END" \
  || die "paper loop failed"

say "paper state"
"$PY" -m vts status

# Surface the latest shortfall gap in bps for a quick day-over-day read.
"$PY" - <<'PY' || true
from vts.config import load_settings
from vts.paper.state import PaperState
s = load_settings()
p = s.data_dir / "paper_state.json"
if not p.exists():
    print("   (no paper state yet)"); raise SystemExit(0)
st = PaperState.load(p)
log = st.shortfall_log
if not log:
    print("   (no shortfall entries yet)"); raise SystemExit(0)
last = log[-1]
gap = last.get("gap")
bps = last.get("gap_bps_of_capital")
print(f"   latest shortfall gap: {gap:,.2f} ({bps:.1f} bps of capital)" if gap is not None
      else "   latest shortfall gap: n/a")
if len(log) >= 2 and log[0].get('gap') is not None and gap is not None:
    trend = "WIDENING ⚠" if gap > log[0]['gap'] else "stable/narrowing"
    print(f"   gap trend since first logged day: {trend}")
print(f"   halted={st.halted}  days_processed={len(st.processed_dates)}")
PY

ok "Gate 3 — paper cycle recorded. Repeat daily for >= 8 weeks before any live gate."
