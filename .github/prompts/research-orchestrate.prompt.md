---
description: "Run a multi-source research workflow by orchestrating local skills (LightRAG, ZIMI, Wolfram|Alpha, Lean4, NotebookLM, SearXNG) and return a grounded synthesis."
agent: "Research Orchestrator"
argument-hint: "Research question or topic"
---

Conduct a grounded multi-source research pass for:

`{{question}}`

Workflow:
1. Start with local/offline sources first:
   - LightRAG query: `python3 .github/skills/lightrag-query/lightrag_query_tool.py --url http://192.168.1.70:9621 --alt-url http://mghasemi.ddns.net:9621 query "{{question}}" --mode mix --format json`
   - ZIMI retrieve fallback: `python3 .github/skills/zimi/zimi_tool.py --url http://192.168.1.70:8899 --alt-url http://mghasemi.ddns.net:8899 retrieve "{{question}}" --search-limit 8 --max-length 6000 --format json`
2. If the question is computational, symbolic, or interpretation-sensitive mathematics, query Wolfram|Alpha:
   - Parse check: `python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py validate "{{question}}" --format json`
   - Symbolic verification: `python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{question}}" --profile symbolic --format json`
   - Theorem or statement verification: `python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{question}}" --profile theorem --format json`
3. If machine-checked formalization or proof-strategy search is requested, route through Lean4:
   - Candidate search: `python3 .github/skills/lean4/lean4_tool.py search "{{question}}" --format json`
   - Proof attempt: `python3 .github/skills/lean4/lean4_tool.py prove "{{question}}" --format json`
   - Compilation check: `python3 .github/skills/lean4/lean4_tool.py check --format json`
4. If project-context grounding is needed, query NotebookLM wrapper:
   - `bash scripts/notebooklm_py.sh status`
   - `bash scripts/notebooklm_py.sh ask "{{question}}"`
5. If recency or web corroboration is needed, run SearXNG:
   - `uv run .github/skills/searxng/scripts/searxng.py search "{{question}}" -n 8 --format json`
6. Return:
   - Direct answer summary
   - 3 to 6 key points
   - Evidence table with source and locator (file/path/url)
   - Contradictions/uncertainties (if any)
   - 1 to 3 follow-up questions

If a source is unavailable, continue with remaining sources and state which checks failed.
