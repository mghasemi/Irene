---
description: "Stage 2: Multi-faceted information gathering — run the Literature Synthesis Agent, ingest papers into LightRAG, populate Zotero, and produce a human-reviewable literature report before computation begins."
---
# Stage 2 — Information Gathering & Literature Synthesis

**Research question**: `{{question}}`

## Step 1 — Check shared-insights and prior failures

```bash
bash .github/skills/simplerag-memory/scripts/simplerag_client.sh query --group shared-insights --text "{{question}}"
python3 .github/skills/research-memory/research_memory_tool.py search "{{question}}"
```

## Step 2 — LightRAG (relational graph, first pass)

```bash
python3 .github/skills/lightrag-query/lightrag_query_tool.py query "{{question}}" --mode hybrid --include-references --format json
```

## Step 3 — ZIMI (encyclopedic grounding)

For each novel term or theorem surfaced in Step 2:

```bash
python3 .github/skills/zimi/zimi_tool.py search "<term>"
python3 .github/skills/zimi/zimi_tool.py retrieve "<article>"
```

## Step 4 — Academic retrieval (arXiv + Semantic Scholar)

```bash
python3 .github/skills/academic-research-hub/scripts/research.py search \
  --query "{{question}}" \
  --sources arxiv semanticscholar \
  --max-results 10
```

For each highly relevant paper (top 5 by relevance):

```bash
# Ingest into LightRAG knowledge graph
python3 .github/skills/lightrag-ingest/lightrag_ingest_tool.py ingest-arxiv <arxiv_id>

# Add to Zotero bibliography
python3 .github/skills/zotero/zotero_tool.py add-paper --arxiv <arxiv_id> --collection "MathAgent"
```

## Step 5 — SimpleRAG recall

```bash
bash .github/skills/simplerag-memory/scripts/simplerag_client.sh query --group workspace --text "{{question}}" -k 5
```

## Step 6 — NotebookLM (if human-uploaded docs are relevant)

```bash
bash scripts/notebooklm_py.sh auth-check
bash scripts/notebooklm_py.sh ask "{{question}}"
```

## Step 7 — SearXNG (fill remaining gaps)

Only if steps 2–6 leave unresolved gaps:

```bash
uv run .github/skills/searxng/scripts/searxng.py "{{question}}" --categories general
```

## Step 8 — Verify LightRAG graph growth

```bash
python3 .github/skills/lightrag-ingest/lightrag_ingest_tool.py status
```

## Step 9 — Reflexion check

Pass the complete synthesis to the Reflexion Agent for certification.
Check: originality, citation completeness, no hallucinated theorems.

## Step 10 — Human gate

Save synthesis report to SiYuan. Mark Vikunja milestone `"STAGE 2: Literature synthesis complete"`.

**STOP**: Await human approval before proceeding to Stage 3.

```bash
python3 .github/skills/vikunja/vikunja_tool.py get-by-title --title "STAGE 2: Literature synthesis complete"
```
