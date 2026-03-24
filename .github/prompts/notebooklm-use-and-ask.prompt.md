---
description: "One-shot NotebookLM flow: list notebooks, optionally select by id prefix, then ask a question via notebooklm-py."
---

Use the local notebooklm-py wrapper to run a one-shot query workflow.

Inputs:
- `{{question}}` (required)
- `{{notebook_prefix}}` (optional)

Workflow:
1. Run `bash scripts/notebooklm_py.sh auth-check`.
2. If auth is invalid, run `bash scripts/notebooklm_py.sh login`, then re-run auth-check.
3. Run `bash scripts/notebooklm_py.sh list`.
4. If `{{notebook_prefix}}` is provided and non-empty, run:
   - `bash scripts/notebooklm_py.sh use {{notebook_prefix}}`
5. Run `bash scripts/notebooklm_py.sh status`.
6. Run:
   - `bash scripts/notebooklm_py.sh ask "{{question}}"`

Return:
- Active notebook id/title used
- Direct answer summary
- Key assumptions
- Main methods
- 2 to 3 follow-up query suggestions

If no notebook is active and `{{notebook_prefix}}` is empty, ask the user for an id/prefix from the list before querying.
