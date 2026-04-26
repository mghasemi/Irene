---
description: "Federation over Text (FoT) aggregation: collect trace summaries from all agents in a session, distill shared insights, and store them in SimpleRAG for cross-agent retrieval in future tasks."
---
# FoT Aggregation — Federation over Text

**Session / project**: `{{project_name}}`

## Purpose

Federation over Text allows sub-agents working on separate subtasks to share metacognitive insights. A coding shortcut discovered in Stage 3 should immediately benefit the Lean4 agent in Stage 4. This prompt performs the aggregation step.

Run this after every major stage boundary, and as part of Stage 6 final sign-off.

## Step 1 — Collect agent trace summaries from SiYuan

```bash
python3 .github/skills/siyuan/siyuan_tool.py search "{{project_name}}"
python3 .github/skills/siyuan/siyuan_tool.py search "experiment result {{project_name}}"
python3 .github/skills/siyuan/siyuan_tool.py search "LEAN4 CERTIFICATE {{project_name}}"
```

## Step 2 — Collect failure patterns from research-memory

```bash
python3 .github/skills/research-memory/research_memory_tool.py search "coding-failure"
python3 .github/skills/research-memory/research_memory_tool.py search "{{project_name}}"
python3 .github/skills/research-memory/research_memory_tool.py recent --limit 20
```

## Step 3 — Query existing shared-insights (avoid duplication)

```bash
bash .github/skills/simplerag-memory/scripts/simplerag_client.sh query \
  --group shared-insights \
  --text "{{project_name}}" \
  -k 10
```

## Step 4 — Distill new insights

Review the collected traces. Extract insights that are:
- Non-obvious (not a basic mathematical fact)
- Cross-applicable (useful to at least 2 different agent types)
- Actionable (a future agent can use it to make a better decision)

Categories to consider:
- **Computational shortcuts**: SymPy/SageMath optimizations that worked
- **Proof strategies**: Lean4 tactics or lemma patterns that succeeded
- **Dead ends**: Approaches that failed (prevent re-runs)
- **Notation decisions**: Canonical symbol choices for recurring quantities
- **Source quality signals**: Which papers/sources gave the most grounded results

## Step 5 — Store insights to SimpleRAG shared-insights

For each extracted insight:
```bash
bash .github/skills/simplerag-memory/scripts/simplerag_client.sh store \
  --group shared-insights \
  --text "[FoT][{{project_name}}][{{category}}] {{insight_text}}"
```

## Step 6 — Store summary to research-memory

```bash
python3 .github/skills/research-memory/research_memory_tool.py add-idea \
  --content "FoT aggregation for {{project_name}} ({{date}}): {{n}} insights stored. Key findings: {{summary}}" \
  --tags "fot,aggregation,{{project_name}}"
```

## Step 7 — Report

Output a brief FoT report:

```
FoT AGGREGATION REPORT
======================
Project: {{project_name}}
Date: [ISO date]
Traces collected from: SiYuan (N notes), research-memory (M entries)
Insights extracted: K
  - Computational: [n]
  - Proof strategies: [n]
  - Dead ends logged: [n]
  - Notation decisions: [n]
  - Source signals: [n]
Stored to SimpleRAG shared-insights: K items
```

## Usage Pattern

Agents should query shared-insights at the **start** of each task:

```bash
bash .github/skills/simplerag-memory/scripts/simplerag_client.sh query \
  --group shared-insights \
  --text "{{task_description}}" \
  -k 5
```

This ensures every agent benefits from prior sessions without requiring explicit cross-agent communication.
