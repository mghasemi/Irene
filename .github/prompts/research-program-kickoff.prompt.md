---
description: "Initialize a long-term research program in Vikunja with milestones, task breakdown, and acceptance criteria."
agent: "Research Program Manager"
argument-hint: "Program objective and constraints"
---

Set up a long-term research program for:

`{{question}}`

Workflow:
0. Run the stage planner first for full pipeline decomposition:
   - `.github/prompts/stage-1-plan.prompt.md`
1. Check whether a matching Vikunja project already exists:
   - `python3 .github/skills/vikunja/vikunja_tool.py projects search --query "{{question}}"`
2. If none exists, create one:
   - `python3 .github/skills/vikunja/vikunja_tool.py projects create --title "{{question}}"`
3. Define 3 to 6 milestone tasks (planning, evidence gathering, analysis, writing, validation, submission).
4. For each milestone, add actionable subtasks with:
   - clear deliverable
   - acceptance criterion
   - suggested priority
   - suggested due date window
5. Return:
   - Project selected/created
   - Milestones and task list
   - Dependency chain
   - First 7-day execution plan

If Vikunja is unavailable, return the same structure as a dry-run plan and include the failure reason.
