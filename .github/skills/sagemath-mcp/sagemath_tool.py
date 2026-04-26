#!/usr/bin/env python3
"""CLI for SageMath advanced algebraic and numerical computation.

Falls back to SymPy for supported operations when `sage` is not available.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_TIMEOUT = int(os.environ.get("SAGE_TIMEOUT", "60"))
SAGE_BIN = shutil.which("sage")


def _fail(message: str, fmt: str = "text") -> None:
    if fmt == "json":
        print(json.dumps({"success": False, "error": message}))
    else:
        print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _run_sage(script: str, fmt: str = "text") -> dict[str, Any]:
    """Execute a Sage script string and return structured output."""
    if not SAGE_BIN:
        return _sympy_fallback(script, fmt)

    with tempfile.NamedTemporaryFile(suffix=".sage", mode="w", delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [SAGE_BIN, tmp_path],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Sage timed out after {DEFAULT_TIMEOUT}s", "fallback": False}
    finally:
        os.unlink(tmp_path)

    return {
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "exit_code": proc.returncode,
        "fallback": False,
        "success": proc.returncode == 0,
    }


def _sympy_fallback(script: str, fmt: str) -> dict[str, Any]:
    """Best-effort SymPy fallback when sage is not installed."""
    try:
        import sympy  # noqa: PLC0415
    except ImportError:
        return {
            "success": False,
            "error": "Neither sage nor sympy is available. Install one of them.",
            "fallback": True,
        }

    # Execute the script in a restricted namespace with sympy pre-imported
    ns: dict[str, Any] = {"__builtins__": {}, **{k: getattr(sympy, k) for k in dir(sympy) if not k.startswith("_")}}
    output_lines: list[str] = []

    def _print(*a: Any, **_kw: Any) -> None:
        output_lines.append(" ".join(str(x) for x in a))

    ns["print"] = _print
    try:
        exec(script, ns)  # noqa: S102
    except Exception as exc:
        return {"success": False, "error": str(exc), "fallback": True}

    return {
        "stdout": "\n".join(output_lines),
        "stderr": "",
        "exit_code": 0,
        "fallback": True,
        "fallback_engine": "sympy",
        "success": True,
    }


def cmd_ring_ops(args: argparse.Namespace) -> dict[str, Any]:
    result = _run_sage(args.script, args.format)
    result["command"] = "ring-ops"
    result["script"] = args.script
    return result


def cmd_matrix(args: argparse.Namespace) -> dict[str, Any]:
    result = _run_sage(args.script, args.format)
    result["command"] = "matrix"
    result["script"] = args.script
    return result


def cmd_precision_arith(args: argparse.Namespace) -> dict[str, Any]:
    result = _run_sage(args.script, args.format)
    result["command"] = "precision-arith"
    result["script"] = args.script
    return result


def cmd_number_field(args: argparse.Namespace) -> dict[str, Any]:
    result = _run_sage(args.script, args.format)
    result["command"] = "number-field"
    result["script"] = args.script
    return result


def cmd_run_script(args: argparse.Namespace) -> dict[str, Any]:
    if not SAGE_BIN:
        _fail("sage is not installed. Cannot run script files without SageMath.", args.format)
    try:
        proc = subprocess.run(
            [SAGE_BIN, args.file],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Sage timed out after {DEFAULT_TIMEOUT}s", "command": "run-script"}
    return {
        "command": "run-script",
        "file": args.file,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "exit_code": proc.returncode,
        "fallback": False,
        "success": proc.returncode == 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SageMath computation CLI (with SymPy fallback)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("ring-ops", "matrix", "precision-arith", "number-field"):
        p = sub.add_parser(name.replace("-", "_").replace("_", "-"), help=f"Run a {name} Sage script")
        p.add_argument("script", help="Sage/Python script string to execute")
        p.add_argument("--format", choices=["text", "json"], default="text")

    p_run = sub.add_parser("run-script", help="Execute a .sage script file")
    p_run.add_argument("file", help="Path to .sage file")
    p_run.add_argument("--format", choices=["text", "json"], default="text")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fmt = getattr(args, "format", "text")

    dispatch = {
        "ring-ops": cmd_ring_ops,
        "matrix": cmd_matrix,
        "precision-arith": cmd_precision_arith,
        "number-field": cmd_number_field,
        "run-script": cmd_run_script,
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
        print(json.dumps(data, ensure_ascii=False))
    else:
        if data.get("stdout"):
            print(data["stdout"])
        if data.get("stderr"):
            print(data["stderr"], file=sys.stderr)
        if not data.get("success"):
            print(f"ERROR: {data.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)
        if data.get("fallback"):
            print(f"(fallback: {data.get('fallback_engine', 'sympy')})", file=sys.stderr)


if __name__ == "__main__":
    main()
