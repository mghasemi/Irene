#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$ROOT_DIR/.venv/bin/notebooklm"

if [[ ! -x "$CLI" ]]; then
  echo "Error: notebooklm CLI not found at $CLI"
  echo "Install first: $ROOT_DIR/.venv/bin/python -m pip install \"notebooklm-py[browser]\""
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/notebooklm_py.sh auth-check
  scripts/notebooklm_py.sh login
  scripts/notebooklm_py.sh list
  scripts/notebooklm_py.sh use <notebook_id_or_prefix>
  scripts/notebooklm_py.sh ask "<question>"
  scripts/notebooklm_py.sh status

Examples:
  scripts/notebooklm_py.sh use acde4920
  scripts/notebooklm_py.sh ask "Summarize key assumptions and optimization methods."
EOF
}

cmd="${1:-}"
case "$cmd" in
  auth-check)
    "$CLI" auth check --test
    ;;
  login)
    "$CLI" login
    ;;
  list)
    "$CLI" list
    ;;
  use)
    if [[ $# -lt 2 ]]; then
      echo "Error: missing notebook id/prefix"
      usage
      exit 1
    fi
    "$CLI" use "$2"
    ;;
  ask)
    if [[ $# -lt 2 ]]; then
      echo "Error: missing question"
      usage
      exit 1
    fi
    shift
    "$CLI" ask "$*"
    ;;
  status)
    "$CLI" status
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Error: unknown command '$cmd'"
    usage
    exit 1
    ;;
esac
