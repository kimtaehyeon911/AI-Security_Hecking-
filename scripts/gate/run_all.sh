#!/usr/bin/env bash
# Run the offline-safe gates in order, stopping at the first failure.
# Gate 4 (testnet smoke) is NOT run here: it needs testnet keys and sends orders,
# so you run it deliberately (scripts/gate/40_testnet_smoke.sh) after 0–3 pass.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

for step in 00_setup.sh 10_ingest.sh 20_backtest_validation.sh 30_paper.sh; do
  echo
  echo "################################################################"
  echo "# $step"
  echo "################################################################"
  bash "$here/$step" || { echo "STOPPED at $step" >&2; exit 1; }
done

echo
echo "All offline gates (0–3) passed. Next, MANUALLY:"
echo "  1) scripts/gate/40_testnet_smoke.sh   (needs testnet keys)"
echo "  2) run 30_paper.sh daily for >= 8 weeks; watch the shortfall gap"
echo "  3) only then consider production — see scripts/gate/README.md §Go-live"
