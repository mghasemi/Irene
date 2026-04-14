---
description: "Run a weekly research program review using Vikunja: progress, blockers, reprioritization, and next-step updates."
agent: "Research Program Manager"
argument-hint: "Project title or focus"
---

Run a weekly review for:

`{{question}}`

Workflow:
1. Resolve target project:
   - `python3 .github/skills/vikunja/vikunja_tool.py projects search --query "{{question}}"`
2. Pull task status and overdue items for the selected project.
3. Summarize:
   - completed this cycle
   - in progress
   - blocked
   - overdue
4. Propose task updates:
   - mark done where evidence is explicit
   - re-prioritize blocked/critical tasks
   - create 3 to 5 next-cycle tasks if gaps exist
5. Return:
   - Program health summary
   - Updated priority queue
   - Blocker mitigation actions
   - Next checkpoint objectives

If no matching project is found, return a concise recommendation to run the kickoff workflow first.
