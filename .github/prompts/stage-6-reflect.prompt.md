---
description: "Stage 6: Final reflexion, Federation over Text aggregation, and human sign-off. Runs the Reflexion Agent for a comprehensive final pass, aggregates cross-agent insights via FoT into SimpleRAG, and presents the researcher with a complete sign-off checklist."
---
# Stage 6 — Agentic Self-Reflection & Final Sign-Off

**Project**: `{{project_name}}`

## Step 1 — Comprehensive Reflexion Agent pass

Run the Reflexion Agent against the complete pipeline output (all stages):

For each stage output (2–5):
1. **Logical consistency**: Lean4 certificates still compile; no numeric claim contradicts Wolfram|Alpha.
2. **Notation consistency**: `drift_count: 0` on the final manuscript.
3. **Originality**: No section is a verbatim duplicate of a single source.
4. **Citation completeness**: All `\cite{}` keys present in `.bib`.

```bash
# Final notation audit
python3 .github/skills/latex-manuscript/latex_tool.py notation-audit manuscript/draft.tex --format json

# Final compile
python3 .github/skills/latex-manuscript/latex_tool.py compile manuscript/draft.tex

# Final Lean4 proof integrity check
python3 .github/skills/lean4/lean4_tool.py check "{{lean4_proof_block}}"

# Final Wolfram|Alpha spot-check on key numeric claim
python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{key_numeric_claim}}" --profile symbolic
```

## Step 2 — Federation over Text (FoT) aggregation

Collect trace summaries from all stages stored in SiYuan:

```bash
python3 .github/skills/siyuan/siyuan_tool.py search "MathAgent: {{project_name}}"
```

Distill cross-agent insights — non-obvious shortcuts, failure patterns, reusable lemmas, symbolic results with broad applicability. Store each insight to SimpleRAG `shared-insights`:

```bash
bash .github/skills/simplerag-memory/scripts/simplerag_client.sh store \
  --group shared-insights \
  --text "{{insight_text}}"
```

Also store a compact summary to research-memory for long-term retention:

```bash
python3 .github/skills/research-memory/research_memory_tool.py add-idea \
  --content "FoT summary for {{project_name}}: {{summary}}" \
  --tags "fot,summary,{{project_name}}"
```

## Step 3 — Generate final status report

Compile a comprehensive report covering:
- Stage 1–6 completion status
- All Reflexion certificates (pass/warn/fail)
- FoT insights extracted
- Outstanding issues (if any)
- Final PDF location

Save to SiYuan and link to Vikunja.

## Step 4 — Sign-off checklist

Present to human researcher:

```
FINAL SIGN-OFF CHECKLIST
=========================
Project: {{project_name}}

[ ] Stage 2 Literature report: CERTIFIED / WARNING / FAIL
[ ] Stage 3 Computation report: CERTIFIED / WARNING / FAIL
[ ] Stage 4 Formal verification: CERTIFIED (no sorry) / PARTIAL / FAIL
[ ] Stage 5 Manuscript: compiled PDF exists / notation drift = 0 / all citations resolved
[ ] FoT insights stored: N items in shared-insights
[ ] Outstanding blocking issues: [list or NONE]

HUMAN DECISION REQUIRED:
[ ] Approve for submission / archiving
[ ] Request revision of: [stage]
[ ] Halt pipeline and escalate: [reason]
```

## Step 5 — Mark final Vikunja milestone

```bash
python3 .github/skills/vikunja/vikunja_tool.py complete-task --title "STAGE 6: Final reflexion sign-off"
```

Upon human approval, mark all stage milestones complete:

```bash
python3 .github/skills/vikunja/vikunja_tool.py complete-task --title "STAGE 1: Planning approved"
# ... repeat for stages 2–5
```

## Constraints
- Do not mark any stage complete unless the Reflexion Agent issued a CERTIFIED or WARNING (not FAIL) verdict.
- Do not archive or submit the manuscript without explicit human sign-off.
- FoT insights must be stored **before** sign-off so they benefit future projects.
