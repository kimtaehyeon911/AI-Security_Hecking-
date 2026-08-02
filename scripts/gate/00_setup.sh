#!/usr/bin/env bash
# Gate 0 — environment + green test suite. Nothing downstream is trustworthy
# until every test passes, so this is the first gate.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

say "Gate 0: build venv + install + run the full test suite"

if [ ! -x "$PY" ]; then
  info "creating venv at $VENV (Python 3.12)"
  command -v uv >/dev/null 2>&1 || die "uv not found — install uv (https://docs.astral.sh/uv/) first"
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$PY" -e ".[dev]"
fi

info "python: $("$PY" --version)"
info "running pytest (all must pass)"
"$PY" -m pytest -q || die "test suite is red — fix before proceeding to any real-money gate"

ok "Gate 0 — venv ready and test suite green"
