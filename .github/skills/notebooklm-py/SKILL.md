---
name: notebooklm-py
description: "Use when users ask to query NotebookLM, add/select/list notebooks, check NotebookLM auth, summarize from a NotebookLM notebook, or run notebooklm-py workflows. Triggers include: Ask NotebookLM, query my NotebookLM notebook, show our notebooks, use notebook <id>, notebooklm login/auth check, and summarize from NotebookLM."
---

# notebooklm-py Skill

Reliable NotebookLM workflow using the local `notebooklm-py` CLI wrapper in this repository.

## Scope

Use this skill for:
- NotebookLM authentication checks and login
- Listing and selecting notebooks
- Asking grounded questions to the active notebook
- Re-running failed NotebookLM queries with clear diagnostics

Do not use this skill for:
- Generic web search
- Local RAG indexing/debugging
- Non-NotebookLM research tasks

## Required Command Path

All commands should use the repository wrapper:

- `bash scripts/notebooklm_py.sh auth-check`
- `bash scripts/notebooklm_py.sh login`
- `bash scripts/notebooklm_py.sh list`
- `bash scripts/notebooklm_py.sh use <id-or-prefix>`
- `bash scripts/notebooklm_py.sh ask "<question>"`
- `bash scripts/notebooklm_py.sh status`

## Standard Workflow

1. Validate auth first:
   - Run `auth-check`.
   - If invalid, run `login`, then run `auth-check` again.

2. Ensure notebook context:
   - Run `status`.
   - If no active notebook, run `list`, then `use <id-or-prefix>`.

3. Execute the query:
   - Run `ask "<question>"`.

4. Return result cleanly:
   - Provide concise summary.
   - Keep source-grounded claims.
   - If answer is long, include a short bullet digest and key assumptions.

## Failure Handling

- If auth fails: instruct login and retry automatically where possible.
- If notebook is not selected: list and select before asking.
- If command exits non-zero: return the exact failing command and stderr summary.
- If query is ambiguous: ask one clarifying question, then rerun.

## Suggested Response Shape

For research summaries, default to:
- Assumptions
- Optimization methods
- Comparison points
- Practical implications for implementation

## Quick Examples

- "Show our notebooks"
  - Run: `bash scripts/notebooklm_py.sh list`

- "Use notebook acde4920"
  - Run: `bash scripts/notebooklm_py.sh use acde4920`

- "Ask NotebookLM: summarize SONC vs SDP"
  - Run: `bash scripts/notebooklm_py.sh ask "Summarize SONC vs SDP for constrained polynomial optimization."`
