---
name: lightrag-query
description: "Use when querying a running LightRAG HTTP service, asking the research graph questions, retrieving referenced context chunks, or running local/global/hybrid/mix LightRAG queries without ingestion."
metadata: {"clawdbot":{"emoji":"🧠","requires":{"bins":["python3"]},"config":{"env":{"LIGHTRAG_URL":{"description":"Primary LightRAG service URL","default":"http://192.168.1.70:9621","required":true},"LIGHTRAG_ALT_URL":{"description":"Fallback LightRAG service URL","default":"http://mghasemi.ddns.net:9621","required":false},"LIGHTRAG_TIMEOUT":{"description":"HTTP timeout in seconds","default":"20","required":false},"LIGHTRAG_API_KEY":{"description":"Optional LightRAG API key for protected deployments","default":"","required":false}}}}}
---
# LightRAG Query Skill

Use this skill to query a running LightRAG server in query-only mode.

## When to Use

- Ask a question against your LightRAG knowledge graph.
- Retrieve answers with references from a running LightRAG API.
- Run local/global/hybrid/mix retrieval modes.
- Retrieve structured data from LightRAG without response generation.
- Stream query chunks for longer outputs.

## Commands

Run the tool directly from this skill folder.

### Query (recommended)

```bash
python3 {baseDir}/lightrag_query_tool.py query "What is the SDP hierarchy?" --mode mix
python3 {baseDir}/lightrag_query_tool.py query "Summarize SONC vs SOS" --mode hybrid --include-references --format json
```

### Structured retrieval only

```bash
python3 {baseDir}/lightrag_query_tool.py query-data "sdp hierarchy" --mode local --top-k 5 --format json
```

### Streaming query

```bash
python3 {baseDir}/lightrag_query_tool.py query-stream "Explain Moment-SOS relaxations" --mode mix
```

### Override URL and fallback

```bash
python3 {baseDir}/lightrag_query_tool.py --url http://192.168.1.70:9621 --alt-url http://mghasemi.ddns.net:9621 query "sdp hierarchy"
```

## Configuration

Optional environment overrides:

- `LIGHTRAG_URL`: defaults to `http://192.168.1.70:9621`
- `LIGHTRAG_ALT_URL`: defaults to `http://mghasemi.ddns.net:9621`
- `LIGHTRAG_TIMEOUT`: defaults to `20`
- `LIGHTRAG_API_KEY`: optional API key for protected services

If you override the primary URL on the CLI with `--url`, fallback is disabled unless you also provide `--alt-url`.

## Output

- `query` returns the answer and optional references.
- `query-data` returns structured retrieval payload.
- `query-stream` prints streamed NDJSON chunks line by line.
- `--format json` returns machine-readable output for downstream tooling.
