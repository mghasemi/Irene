# SimpleRAG Memory Skill API Usage

## Base URLs

The skill should prefer the LAN endpoint first:
- `http://192.168.1.70:7000`

If that fails, retry the public endpoint:
- `http://mghasemi.ddns.net:7000`

A future override can be added through an environment variable such as `SIMPLERAG_BASE_URL`.

## Helper Script Requirements
- `curl` for HTTP calls
- `jq` for JSON-safe payload construction

The helper script probes the LAN endpoint first and retries the public endpoint if the LAN host is unavailable.

## Output Modes
- Default mode is `pretty`, which prints the selected endpoint to stderr and pretty-prints JSON responses when `jq` is available.
- Set `SIMPLERAG_OUTPUT=raw` for scripts or tests that need exact JSON output.

## Failover Testing
- `failover_test.sh` forces the primary endpoint to fail and validates that the helper retries the fallback endpoint.
- By default it uses the reachable LAN endpoint as the fallback target so the retry mechanism can be tested deterministically.
- To test the public endpoint specifically, run with `SIMPLERAG_FAILOVER_TARGET=http://mghasemi.ddns.net:7000`.

## Supported Actions

### Store
Maps to `POST /ingest`

Request body:
```json
{
  "text": "Remember this note",
  "group_id": "workspace",
  "doc_id": "optional-id"
}
```

Behavior:
- `text` is required.
- `group_id` defaults to `workspace`.
- `doc_id` may be omitted for auto-generation.
- If `doc_id` is reused, the backend re-indexes and replaces previous chunks for that id.

Expected response fields:
- `doc_id`
- `group_id`
- `chunks_created`
- `ingested_at`

### Recall
Maps to `POST /query`

Request body:
```json
{
  "query": "What did I say about embeddings?",
  "group_id": "workspace",
  "k": 5,
  "ingested_after": "2026-04-01T00:00:00Z"
}
```

Behavior:
- `query` is required.
- `group_id` defaults to `workspace`.
- `k` defaults to `5`.
- Only send `ingested_after` when the user asks for recent or time-scoped recall.

Expected response fields:
- `query`
- `group_id`
- `k`
- `results[]` with `id`, `doc_id`, `text`, `group_id`, `ingested_at`, `score`

### List Groups
Maps to `GET /groups`

Expected response:
```json
{
  "groups": [
    {
      "group_id": "workspace",
      "documents": 3
    }
  ]
}
```

### Forget
Maps to `DELETE /documents/{doc_id}`

Behavior:
- Requires a concrete `doc_id`.
- If the user does not provide a `doc_id`, locate the note first. Do not guess.

Expected response fields:
- `doc_id`
- `deleted_chunks`

## Error Handling

If both endpoints fail:
- report that the hosted SimpleRAG service is unreachable
- mention both attempted endpoints
- do not instruct local startup unless the user explicitly asks for a local fallback path

If the API returns a normal error body, surface the `detail` message.

## Formatting Guidance

### Store response
- `Stored in workspace. Doc ID: auth-2026-04-23. 2 chunks created.`

### Recall response
- Return the top 1-3 results.
- Include score and ingestion date when relevant.
- Quote or summarize only the most relevant snippet.

### List response
- Return group names with document counts.

### Forget response
- `Deleted 2 chunks from auth-2026-04-23.`

## Current Limitations
- No endpoint exists to list all documents within a group.
- No bulk deletion.
- No structured tags or custom metadata beyond `doc_id`, `group_id`, and `ingested_at`.
- No dedicated update endpoint beyond re-ingesting with the same `doc_id`.
