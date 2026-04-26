---
description: "Stage 5: Manuscript generation — run the Manuscript Pipeline Agent (Drafter → Enhancer → Reviewer), enforce notation unification, sync Zotero bibliography, and compile a final LaTeX PDF."
---
# Stage 5 — Notation Unification & Manuscript Generation

**Manuscript target**: `{{manuscript_title}}`

## Step 1 — Sync bibliography

```bash
python3 .github/skills/zotero/zotero_tool.py sync-bib \
  --collection "MathAgent" \
  --output manuscript/refs.bib
```

Confirm all expected papers are present:
```bash
python3 .github/skills/zotero/zotero_tool.py search "{{key_paper}}"
```

## Step 2 — Notation unification (before any drafting)

Run the Notation Unification Agent on any existing draft or notes:

```bash
python3 .github/skills/latex-manuscript/latex_tool.py auto-glossary manuscript/draft.tex \
  --output manuscript/preamble_glossary.tex
python3 .github/skills/latex-manuscript/latex_tool.py notation-audit manuscript/draft.tex --format json
```

**STOP if drift_count > 0**: Resolve all notation drifts before proceeding to Drafter.

## Step 3 — Drafter pass

Invoke the Manuscript Pipeline Agent in Drafter mode. Provide:
- Verified symbolic results from Stage 3
- Lean4 proof certificates from Stage 4
- Literature synthesis report from Stage 2
- Notation glossary from Step 2
- `manuscript/refs.bib` from Step 1

```bash
# After draft is generated, verify it compiles
python3 .github/skills/latex-manuscript/latex_tool.py compile manuscript/draft.tex
```

**Gate**: Advance only if `errors: []`.

## Step 4 — Enhancer pass

Invoke Manuscript Pipeline Agent in Enhancer mode on the compiled draft.

```bash
# Recompile after enhancement
python3 .github/skills/latex-manuscript/latex_tool.py compile manuscript/draft.tex

# Re-audit notation after rewrites
python3 .github/skills/latex-manuscript/latex_tool.py notation-audit manuscript/draft.tex --format json
```

**Gate**: `errors: []` AND `drift_count: 0`.

## Step 5 — Reviewer pass

Invoke Manuscript Pipeline Agent in Reviewer mode.

```bash
# AST structural check
python3 .github/skills/latex-manuscript/latex_tool.py ast-check manuscript/draft.tex --format json
```

Manual checks to perform:
- Every `\cite{key}` verified against `manuscript/refs.bib`
- Every theorem in `\section{Main Results}` has a `\begin{proof}` or Lean4 cross-reference
- No section is a near-duplicate of a single source (query LightRAG for each major claim)

## Step 6 — Final compile

```bash
python3 .github/skills/latex-manuscript/latex_tool.py compile manuscript/draft.tex --compiler pdflatex
```

## Step 7 — Reflexion check

Pass the compiled manuscript and reviewer report to the Reflexion Agent.
All 4 checks: Logical ✓, Notation ✓, Originality ✓, Citations ✓.

## Step 8 — Human gate

Save manuscript draft and reviewer report to SiYuan. Mark Vikunja milestone `"STAGE 5: Manuscript draft reviewed"`.

**STOP**: Await human approval before proceeding to Stage 6.

```bash
python3 .github/skills/vikunja/vikunja_tool.py get-by-title --title "STAGE 5: Manuscript draft reviewed"
```
