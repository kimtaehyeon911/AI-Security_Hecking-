# Shared helpers for the gate scripts. Sourced, never executed directly.
# Every gate script does:  set -euo pipefail; source "$(dirname "$0")/lib.sh"

# Resolve repo root (this file lives in scripts/gate/) and cd there so every
# relative path (.venv, vts/, scripts/) is stable regardless of caller cwd.
_GATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_GATE_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Load config (defaults, all overridable from the environment).
# shellcheck source=/dev/null
source "$_GATE_DIR/config.env"

PY="$REPO_ROOT/$VENV/bin/python"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
ok()   { printf '\033[1;32mPASS\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

require_py() {
  [ -x "$PY" ] || die "venv python not found at $PY — run scripts/gate/00_setup.sh first"
}

# Fail loudly if a required env var (e.g. an API key loaded from .env) is empty.
require_env() {
  local name="$1"
  [ -n "${!name:-}" ] || die "$name is empty — set it in .env (see .env.example) and re-source"
}

# Load .env into the environment if present (KEY=VALUE lines, ignoring comments).
# Keys are never printed. Never commit .env.
load_dotenv() {
  local f="$REPO_ROOT/.env"
  [ -f "$f" ] || { info "no .env found (public data needs no keys; LLM/live do)"; return 0; }
  set -a
  # shellcheck source=/dev/null
  source "$f"
  set +a
  info "loaded .env"
}
