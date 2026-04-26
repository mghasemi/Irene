---
name: latex-manuscript
description: "Use when compiling LaTeX documents, applying AST-aware diff patches, auditing mathematical notation for consistency, generating a \\newcommand glossary from a .tex file, or detecting undefined macros and unmatched braces."
metadata: {"clawdbot":{"emoji":"📄","requires":{"bins":["python3"]},"optional_bins":["pdflatex","lualatex"],"config":{"env":{"LATEX_COMPILER":{"description":"LaTeX compiler binary: pdflatex or lualatex","default":"pdflatex","required":false},"LATEX_GLOSSARY_GROUP":{"description":"SimpleRAG group name for the master notation glossary","default":"math-notation-glossary","required":false},"SIMPLERAG_URL":{"description":"SimpleRAG API base URL for glossary retrieval","default":"http://192.168.1.70:7000","required":false}}}}}
---
# LaTeX Manuscript Skill

Use this skill to compile, audit, and patch LaTeX manuscripts. It prevents the "execution illusion" — documents that look correct but fail to compile — and enforces notation consistency across long documents.

## When to Use

- Compile a `.tex` file and capture structured error output.
- Check a `.tex` file for unmatched braces, undefined macros, or notation drift.
- Apply a diff-based patch to a `.tex` source without manual editing.
- Auto-generate a `\newcommand` preamble from recurring symbols in a document.
- Audit all math-mode variables against the master glossary stored in SimpleRAG.

## When Not to Use

- Do not use for symbolic computation — use sympy-mcp or sagemath-mcp.
- Do not use for bibliography management — use zotero.

## Commands

### Compile

```bash
python3 {baseDir}/latex_tool.py compile /path/to/paper.tex
python3 {baseDir}/latex_tool.py compile /path/to/paper.tex --compiler lualatex --format json
```

### AST structural check (no compilation required)

```bash
python3 {baseDir}/latex_tool.py ast-check /path/to/paper.tex
python3 {baseDir}/latex_tool.py ast-check /path/to/paper.tex --format json
```

### Notation audit against master glossary

```bash
python3 {baseDir}/latex_tool.py notation-audit /path/to/paper.tex
python3 {baseDir}/latex_tool.py notation-audit /path/to/paper.tex --glossary-group math-notation-glossary --format json
```

### Auto-generate \newcommand glossary preamble

```bash
python3 {baseDir}/latex_tool.py auto-glossary /path/to/paper.tex
python3 {baseDir}/latex_tool.py auto-glossary /path/to/paper.tex --output /path/to/preamble.tex
```

### Apply diff patch

```bash
python3 {baseDir}/latex_tool.py diff-patch /path/to/paper.tex --patch /path/to/changes.diff
python3 {baseDir}/latex_tool.py diff-patch /path/to/paper.tex --old "\\alpha" --new "\\theta" --scope "section:3"
```

## Output

`compile` returns:

```json
{
  "command": "compile",
  "file": "/path/to/paper.tex",
  "success": false,
  "errors": [
    {"line": 47, "message": "Undefined control sequence \\myfunc"},
    {"line": 103, "message": "Missing $ inserted"}
  ],
  "warnings": [],
  "output_pdf": null
}
```

`notation-audit` returns a list of drift events:

```json
{
  "drifts": [
    {"symbol": "\\alpha", "first_use": "section:1", "redefined_as": "\\theta", "section": "section:3"}
  ]
}
```

## Configuration

- `LATEX_COMPILER`: `pdflatex` (default) or `lualatex`.
- `LATEX_GLOSSARY_GROUP`: SimpleRAG group holding the master notation glossary. Default: `math-notation-glossary`.
- `SIMPLERAG_URL`: SimpleRAG endpoint for glossary retrieval.
