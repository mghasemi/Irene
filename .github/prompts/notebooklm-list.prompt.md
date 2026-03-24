---
description: "List NotebookLM notebooks via notebooklm-py, show active context, and suggest the next selection command."
---

Use the local notebooklm-py wrapper to inspect available notebooks and current context.

Workflow:
1. Run `bash scripts/notebooklm_py.sh auth-check`.
2. If auth is invalid, run `bash scripts/notebooklm_py.sh login`, then re-run auth-check.
3. Run `bash scripts/notebooklm_py.sh list`.
4. Run `bash scripts/notebooklm_py.sh status`.
5. Return:
   - Notebook list (id, title, owner)
   - Current active notebook (if any)
   - One suggested next command in this format:
     - `bash scripts/notebooklm_py.sh use <id-prefix>`
