#!/usr/bin/env bash
# Gate 1 — pull point-in-time data into the store for the validation window.
# Binance klines are PUBLIC, so this needs no API key — but it DOES need Binance
# egress (blocked in the CI sandbox; run this where api.binance.com is reachable).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_py
load_dotenv

say "Gate 1: ingest $VTS_UNIVERSE  [$INGEST_START .. $INGEST_END]"
info "data dir: $VTS_DATA_DIR (kept outside the repo)"

"$PY" -m vts ingest --start "$INGEST_START" --end "$INGEST_END" \
  || die "ingest failed — is Binance egress reachable from here? (sandbox blocks it)"

say "store overview"
"$PY" -m vts status

# A store with almost no bars means egress silently returned little; catch it here
# rather than in a misleading backtest. The bar threshold is derived from the
# window IN PYTHON (no GNU-only `date -d`, so the script is portable to BSD/macOS).
"$PY" - "$BACKTEST_START" "$BACKTEST_END" <<'PY' || die "too few bars ingested — check egress/window"
import sys
from datetime import datetime, timedelta, timezone
from vts.config import load_settings
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore
start = datetime.strptime(sys.argv[1], "%Y-%m-%d")
end = datetime.strptime(sys.argv[2], "%Y-%m-%d")
# Require at least ~1/3 of the window's calendar days as bars for the WORST symbol
# — a coarse "egress didn't silently return almost nothing" tripwire.
need = max((end - start).days // 3, 1)
s = load_settings()
if not s.store_path.exists():
    print(f"store not found at {s.store_path}", file=sys.stderr); sys.exit(1)
store = PointInTimeStore(s.store_path)
clock = AsOfClock.at(datetime.now(timezone.utc) + timedelta(days=1))
worst = min((len(store.get_ohlcv(sym, clock)) for sym in s.universe), default=0)
print(f"   fewest bars for any symbol: {worst} (need >= {need})")
sys.exit(0 if worst >= need else 1)
PY

ok "Gate 1 — data ingested for the validation window"
