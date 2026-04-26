---
description: "Stage 1: Strategic planning — decompose a research goal into a Vikunja project with milestones and subtasks, initialize a SiYuan workspace, and obtain human approval before any computation begins."
---
# Stage 1 — Strategic Planning

**Goal**: `{{goal}}`

## Step 1 — Initialize shared-insights context
Query SimpleRAG for any prior insights on this topic before planning.

```bash
bash .github/skills/simplerag-memory/scripts/simplerag_client.sh query --group shared-insights --text "{{goal}}"
```

## Step 2 — Decompose into subtasks
Break `{{goal}}` into a directed acyclic graph of subtasks following this template:

| Phase | Subtask | Depends on |
|---|---|---|
| Literature | Define scope and search terms | — |
| Literature | Retrieve and synthesize papers | Define scope |
| Computation | Validate symbolic prerequisites | Literature |
| Computation | Run empirical experiments | Symbolic prerequisites |
| Verification | Formalize key theorems in Lean4 | Computation |
| Verification | Wolfram|Alpha cross-check | Computation |
| Writing | Notation unification audit | Verification |
| Writing | LaTeX manuscript draft | Notation audit |
| Writing | Bibliography sync (Zotero) | Literature |
| Review | Reflexion pass | Manuscript draft |
| Review | Human sign-off | Reflexion pass |

## Step 3 — Create Vikunja project and milestones

```bash
# Create the project
python3 .github/skills/vikunja/vikunja_tool.py create-project --title "MathAgent: {{goal}}" --description "Automated research pipeline"

# Create milestone tasks (adjust IDs after project creation)
python3 .github/skills/vikunja/vikunja_tool.py create-task --project "MathAgent: {{goal}}" --title "STAGE 1: Planning approved" --priority high
python3 .github/skills/vikunja/vikunja_tool.py create-task --project "MathAgent: {{goal}}" --title "STAGE 2: Literature synthesis complete" --priority high
python3 .github/skills/vikunja/vikunja_tool.py create-task --project "MathAgent: {{goal}}" --title "STAGE 3: Computation report approved" --priority high
python3 .github/skills/vikunja/vikunja_tool.py create-task --project "MathAgent: {{goal}}" --title "STAGE 4: Formal verification certified" --priority high
python3 .github/skills/vikunja/vikunja_tool.py create-task --project "MathAgent: {{goal}}" --title "STAGE 5: Manuscript draft reviewed" --priority high
python3 .github/skills/vikunja/vikunja_tool.py create-task --project "MathAgent: {{goal}}" --title "STAGE 6: Final reflexion sign-off" --priority high
```

## Step 4 — Initialize SiYuan workspace

Create a root note for this research project in SiYuan:

```bash
python3 .github/skills/siyuan/siyuan_tool.py search "MathAgent: {{goal}}"
# If not found, create a new note block with the project overview and task DAG
```

## Step 5 — Human gate

Present the task DAG and Vikunja project link to the human researcher.

**STOP**: Do not proceed to Stage 2 until the human researcher explicitly approves the plan in Vikunja by completing the task `"STAGE 1: Planning approved"`.

Check approval status:
```bash
python3 .github/skills/vikunja/vikunja_tool.py get-by-title --title "STAGE 1: Planning approved"
```
