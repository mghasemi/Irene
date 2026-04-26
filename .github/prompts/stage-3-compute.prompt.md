---
description: "Stage 3: Scientific coding and symbolic computation — run SymPy/SageMath for exact symbolic results, execute empirical experiments in the secure sandbox, cross-validate with Wolfram|Alpha, and produce a computation report."
---
# Stage 3 — Scientific Coding & Symbolic Computation

**Claim / experiment to validate**: `{{claim}}`

## Step 1 — Check failure log (never re-run known failures)

```bash
python3 .github/skills/scientific-coding/scientific_coding_tool.py failures --recent 20
```

## Step 2 — Check shared-insights for shortcuts

```bash
bash .github/skills/simplerag-memory/scripts/simplerag_client.sh query --group shared-insights --text "{{claim}}"
```

## Step 3 — Symbolic computation (exact, no floating-point)

For algebraic / calculus tasks:
```bash
python3 .github/skills/sympy-mcp/sympy_tool.py solve "{{expression}}" --var x
python3 .github/skills/sympy-mcp/sympy_tool.py diff "{{expression}}" --var x
python3 .github/skills/sympy-mcp/sympy_tool.py integrate "{{expression}}" --var x
python3 .github/skills/sympy-mcp/sympy_tool.py dsolve "{{ode}}" --func f
```

For advanced algebraic structures, arbitrary precision, or number fields:
```bash
python3 .github/skills/sagemath-mcp/sagemath_tool.py matrix "{{sage_script}}"
python3 .github/skills/sagemath-mcp/sagemath_tool.py ring-ops "{{sage_script}}"
python3 .github/skills/sagemath-mcp/sagemath_tool.py precision-arith "{{sage_script}}"
```

## Step 4 — Wolfram|Alpha cross-validation

For every non-trivial numeric or symbolic result from Step 3:
```bash
python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py verify "{{claim}}" --profile symbolic
python3 .github/skills/wolfram-alpha/wolfram_alpha_tool.py answer "{{numeric_query}}"
```

## Step 5 — Empirical experiment (sandbox)

Implement the experiment in Python. Run in the secure sandbox:

```bash
# Write experiment to a file first, then run
python3 .github/skills/scientific-coding/scientific_coding_tool.py run \
  --file /path/to/experiment.py \
  --label "{{experiment_label}}"
```

Or inline for short snippets:
```bash
python3 .github/skills/scientific-coding/scientific_coding_tool.py run \
  --code "import numpy as np; ..." \
  --label "{{experiment_label}}"
```

## Step 6 — Convert key results to LaTeX

```bash
python3 .github/skills/sympy-mcp/sympy_tool.py latex "{{expression}}"
```

Store the LaTeX strings for use in Stage 5 manuscript drafting.

## Step 7 — Log insights to research-memory

```bash
python3 .github/skills/research-memory/research_memory_tool.py add-idea \
  --content "{{key_finding}}" \
  --tags "stage-3,computation"
```

## Step 8 — Reflexion check

Pass all symbolic results and experiment outputs to the Reflexion Agent.
Key checks: numeric claims cross-validated by Wolfram|Alpha; no model-generated arithmetic.

## Step 9 — Human gate

Save computation report to SiYuan. Mark Vikunja milestone `"STAGE 3: Computation report approved"`.

**STOP**: Await human approval before proceeding to Stage 4.

```bash
python3 .github/skills/vikunja/vikunja_tool.py get-by-title --title "STAGE 3: Computation report approved"
```
