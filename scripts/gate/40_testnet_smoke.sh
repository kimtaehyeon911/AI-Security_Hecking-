#!/usr/bin/env bash
# Gate 4 — testnet smoke. The adapter defaults to testnet.binance.vision, so no
# real funds are at risk. Even a dry-run needs testnet keys (the 1% cap and order
# pre-validation are computed against a real signed account snapshot).
#
# Two phases:
#   1) dry-run cycle (default): plans orders, submits NOTHING.
#   2) armed cycle (only if SMOKE_GO_LIVE=1): sends real orders TO THE TESTNET,
#      requiring the exact arming token. Verify order round-trip / cancel / flat.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_py
load_dotenv

require_env BINANCE_API_KEY
require_env BINANCE_API_SECRET
info "using testnet keys (adapter base URL defaults to testnet.binance.vision)"

# Kill switch must NOT be engaged, or every cycle just latches a liquidation.
if [ -n "${VTS_KILL_SWITCH:-}" ]; then
  die "VTS_KILL_SWITCH is set ('$VTS_KILL_SWITCH') — unset it before a smoke test"
fi

say "Gate 4 phase 1: DRY-RUN cycle (submits nothing), capital=$LIVE_CAPITAL"
"$PY" -m vts live --model "$MODEL" --capital "$LIVE_CAPITAL" \
  || die "dry-run live cycle failed (keys valid? testnet reachable?)"

if [ "${SMOKE_GO_LIVE:-0}" != "1" ]; then
  ok "Gate 4 (dry-run only) — set SMOKE_GO_LIVE=1 to send real TESTNET orders"
  exit 0
fi

say "Gate 4 phase 2: ARMED cycle on TESTNET (real orders, no real funds)"
if [ "${VTS_LIVE_TRADING_ARMED:-}" != "I_UNDERSTAND_THE_RISK" ]; then
  die "arming token not set. Run: export VTS_LIVE_TRADING_ARMED=I_UNDERSTAND_THE_RISK"
fi
"$PY" -m vts live --model "$MODEL" --capital "$LIVE_CAPITAL" --go-live \
  || die "armed testnet cycle failed"

say "post-cycle state (check sleeve/halt/dust)"
"$PY" -m vts status

cat <<'NOTE'

   Manually verify on the testnet account:
     - orders appeared and filled/cancelled as expected
     - `vts status` shows the sleeve you expect (or flat)
     - engage the kill switch once to confirm liquidation + latch:
         export VTS_KILL_SWITCH=1 && python -m vts live --go-live
       then clear the latch with an audited reset:
         python -m vts reset-halt --scope live --operator "your-name"
NOTE
ok "Gate 4 — testnet smoke complete"
