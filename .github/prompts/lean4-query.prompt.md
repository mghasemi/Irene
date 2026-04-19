---
description: "Use Lean4 to search lemmas, test tactics, and machine-check theorem statements in a repository-agnostic way."
agent: "Lean4 Assistant"
argument-hint: "Lean goal, theorem statement, or proof-debugging request"
---

Run a Lean4-focused workflow for:

`{{question}}`

Workflow:
1. Resolve project context:
   - If a Lean project path is provided, pass it with `--project-dir`.
   - Otherwise let the tool auto-detect from current workspace.
2. Classify intent:
   - Compilation/debugging request: run `check`.
   - Lemma discovery or strategy request: run `search`.
   - Direct proof attempt: run `prove`.
   - Small snippet experimentation: run `repl`.
3. Use one or more commands:
   - `python3 .github/skills/lean4/lean4_tool.py check --format json`
   - `python3 .github/skills/lean4/lean4_tool.py search "{{question}}" --format json`
   - `python3 .github/skills/lean4/lean4_tool.py prove "{{question}}" --format json`
4. If proof attempt fails, summarize unresolved goals and run `search` for narrowed subgoals.
5. Return:
   - Direct outcome summary
   - Candidate lemmas/tactics
   - Remaining blockers or unresolved goals
   - Suggested next Lean step

If Lean or Lake is unavailable, report that explicitly and stop.
