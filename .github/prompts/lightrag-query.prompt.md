---
description: "Query the running LightRAG service through the local lightrag-query skill tool and return a concise grounded summary."
---

Use the local LightRAG query wrapper to ask the running research graph.

Workflow:
1. Run `python3 .github/skills/lightrag-query/lightrag_query_tool.py --url http://192.168.1.70:9621 --alt-url http://mghasemi.ddns.net:9621 query "{{question}}" --mode mix --format json`.
2. If the request fails due to access restrictions, retry with `--api-key` using `LIGHTRAG_API_KEY` from environment.
3. If the answer is too broad, rerun with one of these modes based on intent:
   - `--mode local` for entity/detail-focused answers
   - `--mode global` for broader graph-level context
   - `--mode hybrid` or `--mode mix` for balanced retrieval
4. Return:
   - Direct answer summary
   - 3 to 6 key points
   - Referenced source files (if present in the response)
   - 1 to 3 focused follow-up questions

If the service is unavailable on both endpoints, report both URLs as checked and include the top error line.
