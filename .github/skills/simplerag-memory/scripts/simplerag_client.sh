#!/usr/bin/env bash
set -euo pipefail

PRIMARY_BASE_URL="${SIMPLERAG_PRIMARY_URL:-http://192.168.1.70:7000}"
FALLBACK_BASE_URL="${SIMPLERAG_FALLBACK_URL:-http://mghasemi.ddns.net:7000}"
BASE_URL="${SIMPLERAG_BASE_URL:-}"
OUTPUT_MODE="${SIMPLERAG_OUTPUT:-pretty}"

usage() {
  cat <<'EOF'
Usage:
  simplerag_client.sh health
  simplerag_client.sh groups
  simplerag_client.sh store <text> [group_id] [doc_id]
  simplerag_client.sh query <query> [group_id] [k] [ingested_after]
  simplerag_client.sh delete <doc_id>
EOF
}

emit_json() {
  local payload="$1"

  if [[ "$OUTPUT_MODE" == "raw" ]]; then
    printf '%s\n' "$payload"
    return 0
  fi

  if command -v jq >/dev/null 2>&1; then
    printf '%s\n' "$payload" | jq .
    return 0
  fi

  printf '%s\n' "$payload"
}

probe() {
  local url="$1"
  curl --silent --show-error --fail --max-time 5 "$url/health" >/dev/null
}

resolve_base_url() {
  if [[ -n "$BASE_URL" ]]; then
    printf '%s\n' "$BASE_URL"
    return 0
  fi

  if probe "$PRIMARY_BASE_URL" 2>/dev/null; then
    printf '%s\n' "$PRIMARY_BASE_URL"
    return 0
  fi

  if probe "$FALLBACK_BASE_URL" 2>/dev/null; then
    printf '%s\n' "$FALLBACK_BASE_URL"
    return 0
  fi

  echo "Hosted SimpleRAG service is unreachable. Tried: $PRIMARY_BASE_URL and $FALLBACK_BASE_URL" >&2
  return 1
}

post_json() {
  local base_url="$1"
  local path="$2"
  local payload="$3"

  curl --silent --show-error --fail \
    -H "Content-Type: application/json" \
    -X POST \
    "$base_url$path" \
    -d "$payload"
}

delete_call() {
  local base_url="$1"
  local doc_id="$2"

  curl --silent --show-error --fail \
    -X DELETE \
    "$base_url/documents/$doc_id"
}

action="${1:-}"
if [[ -z "$action" ]]; then
  usage
  exit 1
fi
shift || true

base_url="$(resolve_base_url)"

if [[ "$OUTPUT_MODE" != "raw" ]]; then
  echo "Using SimpleRAG endpoint: $base_url" >&2
fi

case "$action" in
  health)
    emit_json "$(curl --silent --show-error --fail "$base_url/health")"
    ;;
  groups)
    emit_json "$(curl --silent --show-error --fail "$base_url/groups")"
    ;;
  store)
    text="${1:-}"
    group_id="${2:-workspace}"
    doc_id="${3:-}"
    if [[ -z "$text" ]]; then
      echo "store requires <text>" >&2
      exit 1
    fi
    if [[ -n "$doc_id" ]]; then
      payload=$(printf '{"text": %s, "group_id": %s, "doc_id": %s}' \
        "$(jq -Rn --arg v "$text" '$v')" \
        "$(jq -Rn --arg v "$group_id" '$v')" \
        "$(jq -Rn --arg v "$doc_id" '$v')")
    else
      payload=$(printf '{"text": %s, "group_id": %s}' \
        "$(jq -Rn --arg v "$text" '$v')" \
        "$(jq -Rn --arg v "$group_id" '$v')")
    fi
    emit_json "$(post_json "$base_url" "/ingest" "$payload")"
    ;;
  query)
    query_text="${1:-}"
    group_id="${2:-workspace}"
    k="${3:-5}"
    ingested_after="${4:-}"
    if [[ -z "$query_text" ]]; then
      echo "query requires <query>" >&2
      exit 1
    fi
    if [[ -n "$ingested_after" ]]; then
      payload=$(printf '{"query": %s, "group_id": %s, "k": %s, "ingested_after": %s}' \
        "$(jq -Rn --arg v "$query_text" '$v')" \
        "$(jq -Rn --arg v "$group_id" '$v')" \
        "$k" \
        "$(jq -Rn --arg v "$ingested_after" '$v')")
    else
      payload=$(printf '{"query": %s, "group_id": %s, "k": %s}' \
        "$(jq -Rn --arg v "$query_text" '$v')" \
        "$(jq -Rn --arg v "$group_id" '$v')" \
        "$k")
    fi
    emit_json "$(post_json "$base_url" "/query" "$payload")"
    ;;
  delete)
    doc_id="${1:-}"
    if [[ -z "$doc_id" ]]; then
      echo "delete requires <doc_id>" >&2
      exit 1
    fi
    emit_json "$(delete_call "$base_url" "$doc_id")"
    ;;
  *)
    usage
    exit 1
    ;;
esac
