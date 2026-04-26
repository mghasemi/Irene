#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
CLIENT="$ROOT_DIR/.github/skills/simplerag-memory/scripts/simplerag_client.sh"
FAILOVER_TARGET="${SIMPLERAG_FAILOVER_TARGET:-http://192.168.1.70:7000}"

export SIMPLERAG_OUTPUT=raw
export SIMPLERAG_PRIMARY_URL="http://127.0.0.1:1"
export SIMPLERAG_FALLBACK_URL="$FAILOVER_TARGET"

echo "Forcing primary endpoint failure at $SIMPLERAG_PRIMARY_URL"
echo "Expecting fallback to $SIMPLERAG_FALLBACK_URL ..."
OUTPUT="$(bash "$CLIENT" groups)"
echo "$OUTPUT"

echo "Failover test passed."
