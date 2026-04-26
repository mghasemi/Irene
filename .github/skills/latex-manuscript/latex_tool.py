#!/usr/bin/env python3
"""CLI for LaTeX manuscript operations: compile, AST-check, notation-audit, auto-glossary, diff-patch."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_COMPILER = os.environ.get("LATEX_COMPILER", "pdflatex")
DEFAULT_GLOSSARY_GROUP = os.environ.get("LATEX_GLOSSARY_GROUP", "math-notation-glossary")
SIMPLERAG_URL = os.environ.get("SIMPLERAG_URL", "http://192.168.1.70:7000")


def _fail(message: str, fmt: str = "text") -> None:
    if fmt == "json":
        print(json.dumps({"success": False, "error": message}))
    else:
        print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _read_tex(path: str, fmt: str) -> str:
    if not os.path.isfile(path):
        _fail(f"File not found: {path}", fmt)
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def _parse_latex_errors(log: str) -> list[dict[str, Any]]:
    errors = []
    warnings = []
    for line in log.splitlines():
        m = re.match(r"^.*:(\d+):\s+(.*)", line)
        if m:
            errors.append({"line": int(m.group(1)), "message": m.group(2).strip()})
        elif line.startswith("LaTeX Warning:") or line.startswith("Package Warning:"):
            warnings.append(line.strip())
    return errors, warnings


def cmd_compile(args: argparse.Namespace) -> dict[str, Any]:
    compiler = args.compiler or DEFAULT_COMPILER
    if not shutil.which(compiler):
        _fail(f"LaTeX compiler '{compiler}' not found on PATH.", args.format)

    tex_path = os.path.abspath(args.file)
    tex_dir = os.path.dirname(tex_path)

    with tempfile.TemporaryDirectory() as outdir:
        proc = subprocess.run(
            [compiler, "-interaction=nonstopmode", f"-output-directory={outdir}", tex_path],
            capture_output=True,
            text=True,
            cwd=tex_dir,
            timeout=120,
        )
        log = proc.stdout + proc.stderr
        errors, warnings = _parse_latex_errors(log)
        pdf_name = os.path.splitext(os.path.basename(tex_path))[0] + ".pdf"
        pdf_src = os.path.join(outdir, pdf_name)
        success = proc.returncode == 0 and os.path.isfile(pdf_src)
        output_pdf = None
        if success:
            dest = os.path.join(tex_dir, pdf_name)
            import shutil as sh
            sh.copy2(pdf_src, dest)
            output_pdf = dest

    return {
        "command": "compile",
        "file": args.file,
        "compiler": compiler,
        "success": success,
        "errors": errors,
        "warnings": warnings[:20],
        "output_pdf": output_pdf,
    }


# ---------------------------------------------------------------------------
# AST check
# ---------------------------------------------------------------------------

def cmd_ast_check(args: argparse.Namespace) -> dict[str, Any]:
    src = _read_tex(args.file, args.format)
    issues = []

    # Check brace balance
    depth = 0
    for i, ch in enumerate(src):
        if ch == "{" and (i == 0 or src[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or src[i - 1] != "\\"):
            depth -= 1
        if depth < 0:
            issues.append({"type": "unmatched_brace", "message": f"Unmatched '}}' near char {i}"})
            depth = 0
    if depth != 0:
        issues.append({"type": "unmatched_brace", "message": f"Unclosed braces: {depth} '{{' without matching '}}'"}  )

    # Check for commonly undefined macros (not defined in preamble)
    preamble_end = src.find(r"\begin{document}")
    preamble = src[:preamble_end] if preamble_end >= 0 else ""
    body = src[preamble_end:] if preamble_end >= 0 else src
    defined_cmds = set(re.findall(r"\\(?:newcommand|renewcommand|DeclareMathOperator)\{(\\[a-zA-Z]+)\}", preamble))
    used_cmds = set(re.findall(r"(\\[a-zA-Z]+)", body))
    standard_cmds = {
        "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta",
        "\\theta", "\\iota", "\\kappa", "\\lambda", "\\mu", "\\nu", "\\xi", "\\pi",
        "\\rho", "\\sigma", "\\tau", "\\upsilon", "\\phi", "\\chi", "\\psi", "\\omega",
        "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Pi", "\\Sigma",
        "\\Upsilon", "\\Phi", "\\Psi", "\\Omega", "\\mathbb", "\\mathcal", "\\mathbf",
        "\\mathrm", "\\text", "\\frac", "\\sum", "\\prod", "\\int", "\\infty",
        "\\leq", "\\geq", "\\neq", "\\approx", "\\in", "\\notin", "\\subset",
        "\\subseteq", "\\cup", "\\cap", "\\forall", "\\exists", "\\partial",
        "\\nabla", "\\cdot", "\\times", "\\otimes", "\\oplus", "\\circ",
        "\\begin", "\\end", "\\item", "\\label", "\\ref", "\\cite", "\\footnote",
        "\\section", "\\subsection", "\\subsubsection", "\\paragraph",
        "\\textbf", "\\textit", "\\emph", "\\underline", "\\overline",
        "\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg",
        "\\equation", "\\align", "\\proof", "\\theorem", "\\lemma",
        "\\sqrt", "\\hat", "\\bar", "\\vec", "\\dot", "\\ddot", "\\tilde",
        "\\lim", "\\max", "\\min", "\\sup", "\\inf", "\\det", "\\dim",
        "\\ker", "\\log", "\\exp", "\\sin", "\\cos", "\\tan",
    }
    potentially_undefined = used_cmds - defined_cmds - standard_cmds
    for cmd in sorted(potentially_undefined):
        if len(cmd) > 3:  # skip very short macros that are likely standard
            issues.append({"type": "possibly_undefined_macro", "message": f"Macro {cmd} may be undefined"})

    return {
        "command": "ast-check",
        "file": args.file,
        "issue_count": len(issues),
        "issues": issues,
        "success": True,
    }


# ---------------------------------------------------------------------------
# Notation audit
# ---------------------------------------------------------------------------

def cmd_notation_audit(args: argparse.Namespace) -> dict[str, Any]:
    src = _read_tex(args.file, args.format)
    # Extract math-mode tokens per section
    sections = re.split(r"\\(?:section|subsection|subsubsection)\{([^}]+)\}", src)
    symbol_map: dict[str, str] = {}  # symbol -> first section where seen
    drifts = []

    section_label = "preamble"
    for i, chunk in enumerate(sections):
        if i % 2 == 1:
            section_label = chunk
            continue
        math_blocks = re.findall(r"\$([^$]+)\$|\\\[(.+?)\\\]", chunk, re.DOTALL)
        tokens: set[str] = set()
        for inline, display in math_blocks:
            block = inline or display
            tokens |= set(re.findall(r"\\[a-zA-Z]+|[a-zA-Z]", block))
        for tok in tokens:
            if tok in symbol_map and symbol_map[tok] != section_label:
                # Symbol seen in a different section — check for redefinition
                drifts.append({
                    "symbol": tok,
                    "first_use": symbol_map[tok],
                    "also_in": section_label,
                })
            else:
                symbol_map[tok] = section_label

    return {
        "command": "notation-audit",
        "file": args.file,
        "drift_count": len(drifts),
        "drifts": drifts,
        "symbol_inventory": len(symbol_map),
        "success": True,
    }


# ---------------------------------------------------------------------------
# Auto-glossary
# ---------------------------------------------------------------------------

def cmd_auto_glossary(args: argparse.Namespace) -> dict[str, Any]:
    src = _read_tex(args.file, args.format)
    math_tokens = re.findall(r"\\[a-zA-Z]+", src)
    from collections import Counter
    counts = Counter(math_tokens)
    # Keep tokens appearing ≥3 times that are not already standard
    standard = {"\\frac", "\\sum", "\\int", "\\prod", "\\begin", "\\end", "\\left",
                 "\\right", "\\text", "\\mathbb", "\\mathcal", "\\mathbf", "\\mathrm"}
    candidates = {tok: c for tok, c in counts.items() if c >= 3 and tok not in standard}

    lines = [
        "% Auto-generated notation glossary — review before inserting into preamble",
        "% Symbol: count — add \\newcommand definitions below",
    ]
    for tok, c in sorted(candidates.items(), key=lambda x: -x[1]):
        lines.append(f"% {tok}: {c} uses")
        safe_name = tok.lstrip("\\")
        lines.append(f"% \\newcommand{{\\{safe_name}}}{{\\mathrm{{{safe_name}}}}}")

    glossary = "\n".join(lines)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(glossary)

    return {
        "command": "auto-glossary",
        "file": args.file,
        "candidate_count": len(candidates),
        "output": args.output,
        "glossary": glossary if not args.output else None,
        "success": True,
    }


# ---------------------------------------------------------------------------
# Diff patch
# ---------------------------------------------------------------------------

def cmd_diff_patch(args: argparse.Namespace) -> dict[str, Any]:
    src = _read_tex(args.file, args.format)

    if args.patch:
        if not shutil.which("patch"):
            _fail("'patch' command not found on PATH.", args.format)
        with tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write(src)
            tmp_path = tmp.name
        proc = subprocess.run(["patch", tmp_path, args.patch], capture_output=True, text=True)
        if proc.returncode != 0:
            os.unlink(tmp_path)
            return {"success": False, "error": proc.stderr.strip(), "command": "diff-patch"}
        with open(tmp_path, encoding="utf-8") as f:
            patched = f.read()
        os.unlink(tmp_path)
    elif args.old and args.new:
        count = src.count(args.old)
        if count == 0:
            return {"success": False, "error": f"Pattern '{args.old}' not found in file.", "command": "diff-patch"}
        patched = src.replace(args.old, args.new)
    else:
        _fail("Provide --patch <file> or both --old and --new.", args.format)
        return {}

    with open(args.file, "w", encoding="utf-8") as f:
        f.write(patched)

    return {
        "command": "diff-patch",
        "file": args.file,
        "replacements": src.count(args.old) if args.old else "patch applied",
        "success": True,
    }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LaTeX manuscript operations CLI")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    sub = parser.add_subparsers(dest="command", required=True)

    p_comp = sub.add_parser("compile", help="Compile a .tex file")
    p_comp.add_argument("file")
    p_comp.add_argument("--compiler", default=None)
    p_comp.add_argument("--format", choices=["text", "json"], default="text")

    p_ast = sub.add_parser("ast-check", help="Structural check without compilation")
    p_ast.add_argument("file")
    p_ast.add_argument("--format", choices=["text", "json"], default="text")

    p_audit = sub.add_parser("notation-audit", help="Audit math notation for drift")
    p_audit.add_argument("file")
    p_audit.add_argument("--glossary-group", default=DEFAULT_GLOSSARY_GROUP)
    p_audit.add_argument("--format", choices=["text", "json"], default="text")

    p_gloss = sub.add_parser("auto-glossary", help="Generate \\newcommand preamble suggestions")
    p_gloss.add_argument("file")
    p_gloss.add_argument("--output", default=None, help="Write glossary to this file")
    p_gloss.add_argument("--format", choices=["text", "json"], default="text")

    p_patch = sub.add_parser("diff-patch", help="Apply a patch to .tex file")
    p_patch.add_argument("file")
    p_patch.add_argument("--patch", default=None, help="Path to unified diff file")
    p_patch.add_argument("--old", default=None, help="String to replace")
    p_patch.add_argument("--new", default=None, help="Replacement string")
    p_patch.add_argument("--format", choices=["text", "json"], default="text")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fmt = getattr(args, "format", "text")

    dispatch = {
        "compile": cmd_compile,
        "ast-check": cmd_ast_check,
        "notation-audit": cmd_notation_audit,
        "auto-glossary": cmd_auto_glossary,
        "diff-patch": cmd_diff_patch,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        _fail(f"Unknown command: {args.command}", fmt)

    try:
        data = handler(args)
    except Exception as exc:
        _fail(str(exc), fmt)
        return

    if fmt == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        if not data.get("success"):
            print(f"ERROR: {data.get('error', 'Failed')}", file=sys.stderr)
            for e in data.get("errors", []):
                print(f"  Line {e.get('line', '?')}: {e.get('message', '')}")
        elif args.command == "compile":
            if data.get("output_pdf"):
                print(f"OK: {data['output_pdf']}")
            if data.get("errors"):
                for e in data["errors"]:
                    print(f"  Error line {e['line']}: {e['message']}")
        elif args.command in ("ast-check", "notation-audit"):
            for issue in data.get("issues", data.get("drifts", [])):
                print(f"  {issue}")
        elif args.command == "auto-glossary":
            if data.get("glossary"):
                print(data["glossary"])
            else:
                print(f"Written to: {data['output']}")
        elif args.command == "diff-patch":
            print(f"Patched: {data.get('file')} ({data.get('replacements')} changes)")


if __name__ == "__main__":
    main()
