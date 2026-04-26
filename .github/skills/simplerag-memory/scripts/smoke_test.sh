#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
CLIENT="$ROOT_DIR/.github/skills/simplerag-memory/scripts/simplerag_client.sh"
TEST_GROUP="${SIMPLERAG_TEST_GROUP:-skill-smoke-tests}"
export SIMPLERAG_OUTPUT=raw
STAMP="$(date +%Y%m%d%H%M%S)"
DOC_ID="skill-smoke-$STAMP"
TEXT="Smoke test note created at $STAMP. Remember that cosine distance is the retrieval metric."
QUERY_TEXT="What do I know about cosine distance?"

echo "[1/4] Storing test note..."
STORE_OUTPUT="$(bash "$CLIENT" store "$TEXT" "$TEST_GROUP" "$DOC_ID")"
echo "$STORE_OUTPUT"

echo "[2/4] Querying test note..."
QUERY_OUTPUT="$(bash "$CLIENT" query "$QUERY_TEXT" "$TEST_GROUP" 3)"
echo "$QUERY_OUTPUT"

if ! printf '%s' "$QUERY_OUTPUT" | grep -q "$DOC_ID"; then
  echo "Smoke test failed: expected query results to reference $DOC_ID" >&2
  exit 1
fi

echo "[3/4] Listing groups..."
GROUPS_OUTPUT="$(bash "$CLIENT" groups)"
echo "$GROUPS_OUTPUT"

if ! printf '%s' "$GROUPS_OUTPUT" | grep -q "$TEST_GROUP"; then
  echo "Smoke test failed: expected groups to include $TEST_GROUP" >&2
  exit 1
fi

echo "[4/4] Deleting test note..."
DELETE_OUTPUT="$(bash "$CLIENT" delete "$DOC_ID")"
echo "$DELETE_OUTPUT"

echo "Smoke test passed."
