# MathAgent Workflow Manual

## Purpose

This manual explains how to run the repository's multi-agent research workflow reliably from planning through final sign-off.

Primary goals:
- Keep research outputs grounded and traceable
- Enforce stage gates and human approvals
- Reuse prior insights across sessions
- Produce verifiable computation, proof, and manuscript artifacts

## Core Components

### Coordinator and QA
- Research Orchestrator agent: central coordinator, routing, stage gates, loop detection
- Reflexion Agent: final checker at stage boundaries (logic, notation, originality, citations)

### Specialist Agents
- Literature Synthesis Agent: evidence gathering and synthesis
- Scientific Coding Agent: experiments and computational validation
- Lean4 Assistant: formal proof support
- Manuscript Pipeline Agent: drafter, enhancer, reviewer flow
- Notation Unification Agent: symbol drift detection and glossary normalization
- Research Program Manager: long-horizon planning and weekly reviews

### Stage Prompts
- Stage 1 plan: .github/prompts/stage-1-plan.prompt.md
- Stage 2 gather: .github/prompts/stage-2-gather.prompt.md
- Stage 3 compute: .github/prompts/stage-3-compute.prompt.md
- Stage 4 verify: .github/prompts/stage-4-verify.prompt.md
- Stage 5 write: .github/prompts/stage-5-write.prompt.md
- Stage 6 reflect: .github/prompts/stage-6-reflect.prompt.md
- FoT aggregation: .github/prompts/fot-aggregate.prompt.md

## One-Time Setup Checklist

1. Confirm MCP server config exists and loads:
- .mcp.json

2. Confirm key runtimes are available:
- Python for most tools
- SymPy venv for sympy-mcp if configured in .mcp.json
- Sage environment for sagemath-mcp if configured in .mcp.json

3. Confirm external integrations are configured:
- Vikunja URL/token
- SiYuan URL/token
- Zotero library type, library ID, API key, optional bib output path
- LightRAG endpoints and key if used

4. Reload MCP after config changes:
- Developer: Reload Window (or MCP restart command if available)

## Standard Operating Procedure (End-to-End)

### Stage 0: Program Kickoff
Use:
- .github/prompts/research-program-kickoff.prompt.md

Outputs:
- Program skeleton in Vikunja
- Initial milestone map

### Stage 1: Strategic Planning
Use:
- .github/prompts/stage-1-plan.prompt.md

Actions:
- Pull prior shared insights
- Decompose goal into dependency graph
- Create project and stage milestones in Vikunja
- Initialize SiYuan workspace note

Gate:
- Human must approve Stage 1 milestone before Stage 2

### Stage 2: Literature and Evidence Gathering
Use:
- .github/prompts/stage-2-gather.prompt.md

Actions:
- Run literature synthesis workflow
- Query and ingest from approved sources
- Push citations into Zotero as needed

Gate:
- Reflexion check and human approval before Stage 3

### Stage 3: Computation and Experiments
Use:
- .github/prompts/stage-3-compute.prompt.md

Actions:
- Run symbolic checks and empirical runs
- Compare independent computational paths where possible
- Record outputs for manuscript reuse

Gate:
- Reflexion check and human approval before Stage 4

### Stage 4: Formal Verification
Use:
- .github/prompts/stage-4-verify.prompt.md

Actions:
- Formalize key claims in Lean4
- Cross-check critical numeric statements

Gate:
- Reflexion check and human approval before Stage 5

### Stage 5: Manuscript Production
Use:
- .github/prompts/stage-5-write.prompt.md

Actions:
- Run notation unification
- Run manuscript pipeline passes
- Compile and citation-check outputs

Gate:
- Reflexion check and human approval before Stage 6

### Stage 6: Reflection and Final Sign-Off
Use:
- .github/prompts/stage-6-reflect.prompt.md

Actions:
- Run final Reflexion certification
- Aggregate cross-agent lessons via FoT
- Produce sign-off checklist and close milestones

## Weekly Operations

Use:
- .github/prompts/research-weekly-review.prompt.md

Cycle:
- Review completed, in-progress, blocked, overdue
- Reprioritize tasks and define next cycle goals
- Link to Stage 6 checklist when closing a cycle

## Runtime Policies to Respect

1. Stage boundary gate policy
- Never advance to next stage before Reflexion certificate and human sign-off

2. GoA message passing policy
- Deterministic computation outputs are treated as hard context for downstream agent work

3. Loop detection policy
- Repeated identical tool calls must trigger escalation (do not spin)

4. Provenance policy
- Preserve source traceability for every major claim

## Troubleshooting Guide

### Symptom: Tool appears unavailable after config edit
- Reload VS Code window
- Re-run a direct CLI smoke test with the same interpreter configured in .mcp.json

### Symptom: Sage commands run in fallback mode
- Verify Sage binary visibility in the configured runtime
- Verify sagemath-mcp command in .mcp.json points to the intended environment launcher

### Symptom: Zotero requests fail
- Verify ZOTERO_LIBRARY_ID and ZOTERO_API_KEY values
- Verify pyzotero is installed in the interpreter used by the zotero MCP server

### Symptom: Stage stalls repeatedly
- Check Reflexion output for FAIL/WARN reasons
- Check Vikunja sign-off task state
- Check loop detection escalation notes

## Minimal Smoke Test Set

Run these after major environment/config updates:

1. SymPy test
- Simplify identity expression and verify expected exact result

2. Sage test
- Run matrix eigenvalue command and verify fallback is false

3. Zotero test
- Run search in JSON mode and verify success true

4. Prompt flow test
- Execute Stage 1 planning prompt for a small sample goal and verify milestone creation

## Recommended Team Usage Pattern

1. Start each new project with kickoff + Stage 1 only
2. Do not parallelize stage transitions across the gate boundary
3. Keep manuscript generation late (after compute + verification evidence is stable)
4. Run FoT aggregation at major checkpoints, not only at the end
5. Use weekly review prompt for cadence and backlog hygiene

## File Map

Agents:
- .github/agents/research-orchestrator.agent.md
- .github/agents/reflexion-agent.agent.md
- .github/agents/literature-synthesis.agent.md
- .github/agents/scientific-coding-agent.agent.md
- .github/agents/manuscript-pipeline.agent.md
- .github/agents/notation-unification.agent.md
- .github/agents/research-program-manager.agent.md

Prompts:
- .github/prompts/research-program-kickoff.prompt.md
- .github/prompts/research-weekly-review.prompt.md
- .github/prompts/stage-1-plan.prompt.md
- .github/prompts/stage-2-gather.prompt.md
- .github/prompts/stage-3-compute.prompt.md
- .github/prompts/stage-4-verify.prompt.md
- .github/prompts/stage-5-write.prompt.md
- .github/prompts/stage-6-reflect.prompt.md
- .github/prompts/fot-aggregate.prompt.md

Configuration:
- .mcp.json
- adr-config.yml
- scripts/adr-wrapper.sh
