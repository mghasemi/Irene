---
name: lightrag-ingest
description: "Use when ingesting new content into the LightRAG knowledge graph: upload PDFs, plain text, arXiv papers by ID, or arbitrary URLs. Also use to check graph ingestion status. Complements lightrag-query which handles retrieval only."
metadata: {"clawdbot":{"emoji":"📥","requires":{"bins":["python3"]},"optional_bins":["pdftotext"],"config":{"env":{"LIGHTRAG_URL":{"description":"Primary LightRAG service URL","default":"http://192.168.1.70:9621","required":true},"LIGHTRAG_ALT_URL":{"description":"Fallback LightRAG service URL","default":"http://mghasemi.ddns.net:9621","required":false},"LIGHTRAG_TIMEOUT":{"description":"HTTP timeout in seconds for ingest calls (longer than query)","default":"60","required":false},"LIGHTRAG_API_KEY":{"description":"Optional API key for protected deployments","default":"","required":false}}}}}
---
# LightRAG Ingestion Skill

Use this skill to push new academic content into the running LightRAG knowledge graph. Ingestion triggers entity-relationship extraction and updates the Neo4j graph — making content available to `lightrag-query` immediately after.

## When to Use

- After academic-research-hub retrieves papers, ingest their text into LightRAG.
- Ingest a PDF directly from disk.
- Ingest an arXiv paper by ID (fetch abstract + full text automatically).
- Ingest a web URL's text content.
- Check how many nodes/entities are in the current graph.

## When Not to Use

- Do not use to query the graph — use lightrag-query.
- Do not use for note storage — use siyuan or research-memory.

## Commands

### Ingest plain text

```bash
python3 {baseDir}/lightrag_ingest_tool.py ingest-text "The Riemann hypothesis states that..."
python3 {baseDir}/lightrag_ingest_tool.py ingest-text "$(cat paper_excerpt.txt)"
```

### Ingest a PDF from disk

```bash
python3 {baseDir}/lightrag_ingest_tool.py ingest-pdf /path/to/paper.pdf
python3 {baseDir}/lightrag_ingest_tool.py ingest-pdf /path/to/paper.pdf --format json
```

### Ingest an arXiv paper by ID

```bash
python3 {baseDir}/lightrag_ingest_tool.py ingest-arxiv 2310.17567
python3 {baseDir}/lightrag_ingest_tool.py ingest-arxiv 2310.17567 --format json
```

### Ingest a URL

```bash
python3 {baseDir}/lightrag_ingest_tool.py ingest-url "https://lightrag.github.io/"
```

### Check graph status

```bash
python3 {baseDir}/lightrag_ingest_tool.py status
python3 {baseDir}/lightrag_ingest_tool.py status --format json
```

## Output

`ingest-arxiv` returns:

```json
{
  "command": "ingest-arxiv",
  "arxiv_id": "2310.17567",
  "title": "Skill-Mix: a Flexible and Expandable...",
  "chars_ingested": 48203,
  "success": true
}
```

`status` returns:

```json
{
  "node_count": 3847,
  "edge_count": 12091,
  "last_updated": "2026-04-25T14:32:00Z",
  "success": true
}
```

## Configuration

- `LIGHTRAG_URL`: defaults to `http://192.168.1.70:9621`.
- `LIGHTRAG_ALT_URL`: defaults to `http://mghasemi.ddns.net:9621`.
- `LIGHTRAG_TIMEOUT`: defaults to `60` (ingestion is slower than querying).
- `LIGHTRAG_API_KEY`: optional bearer token for protected deployments.

## Notes

- PDF text extraction uses `pdftotext` (poppler) if available, otherwise falls back to a pure-Python extractor.
- arXiv ingestion fetches the abstract page text. For LaTeX source, download the `.tar.gz` from arXiv and pipe through `ingest-text`.
