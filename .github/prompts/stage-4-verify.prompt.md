---
description: "Stage 4: Formal verification — formalize theorems in Lean4, cross-validate with Wolfram|Alpha, and produce machine-checkable proof certificates before any manuscript section is drafted."
---
# Stage 4 — Formal Verification

**Theorem / claim to verify**: `{{theorem}}`

## Step 1 — Check shared-insights for prior proof strategies

```bash
bash .github/skills/simplerag-memory/scripts/simplerag_client.sh query --group shared-insights --text "{{theorem}}"
```

## Step 2 — Search Mathlib for existing lemmas

```bash
python3 .github/skills/lean4/lean4_tool.py search "{{theorem}}"
```

Scan the results for lemmas that can shorten the proof. Note any that directly imply the theorem.

## Step 3 — Attempt proof (construction-verification paradigm)

First, write the Lean4 statement, then attempt the proof:

```bash
python3 .github/skills/lean4/lean4_tool.py check "
theorem my_theorem : {{lean4_statement}} := by
  sorry
"
```

Then iteratively replace `sorry` with tactics. Use the REPL for interactive exploration:

```bash
python3 .github/skills/lean4/lean4_tool.py repl "
#check {{lemma_name}}
example : {{subgoal}} := by exact?
"
```

Attempt full proof:
```bash
python3 .github/skills/lean4/lean4_tool.py prove "{{lean4_statement}}"
```

## Step 4 — Wolfram|Alpha secondary validation

For any numeric corollary of the theorem:
```bash
python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{numeric_corollary}}" --profile symbolic
python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{theorem_statement}}" --profile theorem
```

## Step 5 — Record proof certificate

If Lean4 confirms success (no `sorry`, no errors):

```bash
python3 .github/skills/research-memory/research_memory_tool.py add-idea \
  --content "LEAN4 CERTIFICATE: {{theorem}} — verified. Proof: {{proof_sketch}}" \
  --tags "lean4,certificate,stage-4"
```

Store the full proof in SiYuan for manuscript reference.

## Step 6 — Handle proof failure

If the proof cannot be completed:
1. Document the exact Lean4 error state.
2. Split the theorem into smaller lemmas and return to Step 3 for each.
3. If stuck after 3 iterations, save partial progress and escalate to human via Vikunja with `priority: high`.

```bash
python3 .github/skills/vikunja/vikunja_tool.py create-task \
  --title "PROOF STUCK: {{theorem}}" \
  --priority high \
  --description "Lean4 state: {{error}}"
```

## Step 7 — Reflexion check

Pass the Lean4 certificate and Wolfram|Alpha validation to the Reflexion Agent.
Key check: the proof compiles with no `sorry` placeholders.

## Step 8 — Human gate

Save verification report to SiYuan. Mark Vikunja milestone `"STAGE 4: Formal verification certified"`.

**STOP**: Await human approval before proceeding to Stage 5.

```bash
python3 .github/skills/vikunja/vikunja_tool.py get-by-title --title "STAGE 4: Formal verification certified"
```
