---
description: "Check mathematical identities, theorems, and formal statements using Wolfram|Alpha's theorem and symbolic verification modes."
---

Use the local Wolfram|Alpha wrapper to check a mathematical identity, prove or refute a symbolic statement, or look up theorem details.

Assume the input is a theorem name, mathematical identity, or formal claim (e.g., "sin²x + cos²x = 1", "Euler's identity", "Fermat's last theorem").

Workflow:
1. Run `validate` to confirm Wolfram|Alpha parses the statement and to surface any interpretation ambiguities:
   ```
   python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py validate "{{statement}}" --format json
   ```
2. If `validate` returns a `Clash` assumption (meaning WA found multiple possible meanings), resolve it by passing the right token via `--assumption`. For example:
   ```
   python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{statement}}" --profile theorem --assumption "<token>*<type>-<value>" --format json
   ```
   The token, type, and value come from the `assumptions` array in the `validate` output.
3. Run the primary proof check:
   ```
   python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{statement}}" --profile theorem --format json
   ```
4. If the theorem profile returns no useful pods, also run with `--profile symbolic` to capture algebraic verification:
   ```
   python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{statement}}" --profile symbolic --format json
   ```
5. If both profiles return no pods, run without `--profile` for the full result set.

Reporting:
- State whether the claim is confirmed, refuted, or ambiguous based on the pods returned.
- Quote the primary result and any "Statement" or "Associated equation" pods verbatim.
- List every assumption that was active (e.g., degree vs. radian, natural vs. base-10 logarithm, mathematical vs. historical interpretation).
- If Wolfram|Alpha surfaces alternate forms or related theorems, include them as a follow-up.
- End with 1–3 focused follow-up questions the user might want to explore next.

If `WOLFRAM_ALPHA_APPID` is missing, report that explicitly and stop.
