#!/usr/bin/env bash
# Gate 2 — walk-forward backtest on a CUTOFF-CLEAN window, with two hard checks:
#   (a) the window is clean vs the model's effective cutoff (check_window.py), and
#   (b) the strategy beats BOTH benchmarks after costs AND certifies as clean.
# Only (a)+(b) together let a result count as validated (mandate: Step 2/3).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_py
load_dotenv

# The contamination gate keys on the model that could have memorized outcomes:
# the deterministic momentum model is 'fake-momentum' (exempt); the LLM graph is
# whatever VTS_DEEP_THINK_LLM names (must be a verified registry entry).
if [ "$MODEL" = "momentum" ]; then
  CHECK_ID="fake-momentum"
else
  CHECK_ID="$VTS_DEEP_THINK_LLM"
fi

say "Gate 2a: window is cutoff-clean for '$CHECK_ID'?  [$BACKTEST_START .. $BACKTEST_END]"
"$PY" scripts/gate/check_window.py "$CHECK_ID" "$BACKTEST_START" "$BACKTEST_END" \
  || die "window is not cutoff-clean — move BACKTEST_START later (see message above)"

report="$VTS_DATA_DIR/backtest_${CHECK_ID}_${BACKTEST_START}_${BACKTEST_END}.md"
say "Gate 2b: backtest ($MODEL) → $report"

set +e
"$PY" -m vts backtest --model "$MODEL" \
  --start "$BACKTEST_START" --end "$BACKTEST_END" --report "$report"
rc=$?
set -e

case "$rc" in
  0) info "engine gate: PASS (beat every benchmark after costs)";;
  1) die "engine gate: FAIL — strategy did NOT beat every benchmark after costs. \
Do NOT retune on this same window and re-score (forbidden); change hypothesis or data.";;
  2) die "not enough bars for this window — run 10_ingest.sh first";;
  *) die "backtest exited $rc";;
esac

# Belt-and-suspenders: the report must SAY clean AND PASS (exit code already
# implies PASS, but we assert the printed contamination line too).
grep -q 'Contamination: \*\*clean\*\*' "$report" \
  || die "report is not certifiable-clean (contamination != clean) — window is reference-only"
grep -q '## Gate: PASS' "$report" || die "report does not show Gate: PASS"

say "report tail"
tail -n 12 "$report" || true
ok "Gate 2 — validated: cutoff-clean window, beats both benchmarks after costs"
