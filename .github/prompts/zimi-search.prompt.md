---
description: "Search the ZIMI instance via the local zimi skill tool and return a concise grounded summary with sources."
---

Use the local ZIMI query wrapper to search the offline encyclopedia.

Workflow:
1. Run `python3 .github/skills/zimi/zimi_tool.py --url http://192.168.1.70:8899 --alt-url http://mghasemi.ddns.net:8899 search "{{question}}" --limit 8 --format json`.
2. If search returns no useful results, run `python3 .github/skills/zimi/zimi_tool.py --url http://192.168.1.70:8899 --alt-url http://mghasemi.ddns.net:8899 retrieve "{{question}}" --search-limit 8 --max-length 6000 --format json`.
3. Prefer mathematically relevant sources when multiple hits are close by using `--prefer-source wikipedia_en_mathematics_nopic --prefer-source planetmath.org`.
4. Return:
   - Direct answer summary
   - 3 to 6 key points
   - Top matching entries with title, zim, and path
   - 1 to 3 focused follow-up questions

If the service is unavailable on both endpoints, report both URLs as checked and include the top error line.
