---
description: "Query the local Wolfram|Alpha skill for concise computational answers or structured mathematical verification."
---

Use the local Wolfram|Alpha wrapper to answer or verify a mathematical query.

Workflow:
1. Classify the request:
   - Direct computation, value lookup, simplification, derivative, integral, factorization, or numeric answer: run `python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py answer "{{question}}" --format json`.
   - Verification, ambiguity checking, symbolic identity inspection, theorem or statement interpretation: run `python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py validate "{{question}}" --format json` first.
2. For symbolic or computational verification, run `python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{question}}" --profile symbolic --format json`.
3. For theorem, identity, or statement-style verification, run `python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{question}}" --profile theorem --format json`.
4. If a profiled verification result has no useful pods, retry `verify` without `--profile`.
5. Return:
   - Direct answer summary
   - 3 to 6 key points
   - Any warnings or assumptions that materially affect interpretation
   - 1 to 3 focused follow-up questions

If `WOLFRAM_ALPHA_APPID` is missing, report that explicitly and stop.