---
description: "Ask the active NotebookLM notebook via notebooklm-py and return a concise grounded summary."
---

Use the local notebooklm-py wrapper to query the currently active NotebookLM context.

Workflow:
1. Run `bash scripts/notebooklm_py.sh auth-check`.
2. If auth is invalid, run `bash scripts/notebooklm_py.sh login`, then re-run auth-check.
3. Run `bash scripts/notebooklm_py.sh status` and ensure a notebook is active.
4. Ask this question with the wrapper:
   - `bash scripts/notebooklm_py.sh ask "{{question}}"`
5. Return:
   - Direct answer summary
   - Key assumptions
   - Main methods or steps
   - 2 to 3 follow-up questions

If no notebook is active, run `bash scripts/notebooklm_py.sh list` and ask which notebook ID/prefix to use.
