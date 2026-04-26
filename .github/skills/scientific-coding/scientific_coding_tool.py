#!/usr/bin/env python3
"""Secure Python execution sandbox for scientific experiments.

Enforces CPU and memory limits. Auto-logs successes to SiYuan and failures
to research-memory to prevent redundant re-runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any


SANDBOX_CPU_SECONDS = int(os.environ.get("SANDBOX_CPU_SECONDS", "30"))
SANDBOX_MEM_MB = int(os.environ.get("SANDBOX_MEM_MB", "512"))
SIYUAN_URL = os.environ.get("SIYUAN_URL", "")
SIYUAN_TOKEN = os.environ.get("SIYUAN_TOKEN", "")
RESEARCH_MEMORY_DB = os.environ.get("RESEARCH_MEMORY_DB", "")


def _fail(message: str, fmt: str = "text") -> None:
    if fmt == "json":
        print(json.dumps({"success": False, "error": message}))
    else:
        print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Sandbox wrapper script
# ---------------------------------------------------------------------------

_SANDBOX_WRAPPER = """\
import resource, sys, os

cpu = int(os.environ.get('SANDBOX_CPU_SECONDS', '30'))
mem_mb = int(os.environ.get('SANDBOX_MEM_MB', '512'))

resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))

# Block socket creation (best-effort)
import builtins
_real_import = builtins.__import__
def _blocked_import(name, *args, **kwargs):
    mod = _real_import(name, *args, **kwargs)
    if name == 'socket':
        import socket as _sock
        _orig_socket = _sock.socket
        class _BlockedSocket(_orig_socket):
            def connect(self, *a, **kw):
                raise PermissionError("Network access is blocked in sandbox")
            def sendto(self, *a, **kw):
                raise PermissionError("Network access is blocked in sandbox")
        _sock.socket = _BlockedSocket
    return mod
builtins.__import__ = _blocked_import

# Execute user script
with open(sys.argv[1]) as f:
    code = f.read()
exec(compile(code, sys.argv[1], 'exec'), {'__name__': '__main__'})
"""


def _run_sandboxed(code: str, fmt: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        wrapper_path = os.path.join(tmpdir, "_sandbox_runner.py")
        user_path = os.path.join(tmpdir, "_user_code.py")

        with open(wrapper_path, "w") as f:
            f.write(_SANDBOX_WRAPPER)
        with open(user_path, "w") as f:
            f.write(code)

        env = os.environ.copy()
        env["SANDBOX_CPU_SECONDS"] = str(SANDBOX_CPU_SECONDS)
        env["SANDBOX_MEM_MB"] = str(SANDBOX_MEM_MB)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, wrapper_path, user_path],
                capture_output=True,
                text=True,
                timeout=SANDBOX_CPU_SECONDS + 5,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"CPU time limit exceeded ({SANDBOX_CPU_SECONDS}s)",
                "stdout": "",
                "stderr": "",
                "cpu_seconds_used": SANDBOX_CPU_SECONDS,
                "exit_code": -1,
            }
        elapsed = time.monotonic() - start

        return {
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "exit_code": proc.returncode,
            "cpu_seconds_used": round(elapsed, 2),
            "success": proc.returncode == 0,
        }


def _log_failure(label: str, code: str, error: str) -> None:
    if not RESEARCH_MEMORY_DB:
        return
    tool = os.path.join(os.path.dirname(__file__), "..", "research-memory", "research_memory_tool.py")
    if not os.path.isfile(tool):
        return
    content = f"[sandbox failure] label={label}\nError: {error}\n---\n{code[:500]}"
    subprocess.run(
        [sys.executable, tool, "add-idea", "--content", content, "--tags", "coding-failure"],
        capture_output=True,
    )


def _log_siyuan(label: str, stdout: str) -> str | None:
    if not SIYUAN_URL or not SIYUAN_TOKEN:
        return None
    tool = os.path.join(os.path.dirname(__file__), "..", "siyuan", "siyuan_tool.py")
    if not os.path.isfile(tool):
        return None
    content = f"**Experiment: {label}**\n\n```\n{stdout[:2000]}\n```"
    proc = subprocess.run(
        [sys.executable, tool, "create", "--content", content],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(proc.stdout)
        return data.get("block_id")
    except Exception:
        return None


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.file:
        if not os.path.isfile(args.file):
            _fail(f"File not found: {args.file}", args.format)
        with open(args.file, encoding="utf-8") as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        _fail("Provide --code or --file.", args.format)
        return {}

    label = args.label or (os.path.basename(args.file) if args.file else "inline")
    result = _run_sandboxed(code, args.format)
    result["command"] = "run"
    result["label"] = label

    if result["success"]:
        block_id = _log_siyuan(label, result["stdout"])
        result["siyuan_block_id"] = block_id
    else:
        _log_failure(label, code, result.get("stderr", result.get("error", "")))
        result["failure_logged"] = bool(RESEARCH_MEMORY_DB)

    return result


def cmd_test(args: argparse.Namespace) -> dict[str, Any]:
    if not os.path.isfile(args.file):
        _fail(f"Test file not found: {args.file}", args.format)

    # Wrap in pytest call within sandbox
    code = f"import subprocess, sys\nresult = subprocess.run([sys.executable, '-m', 'pytest', '{args.file}', '-v'], capture_output=False)\nsys.exit(result.returncode)"
    label = f"pytest:{os.path.basename(args.file)}"
    result = _run_sandboxed(code, args.format)
    result["command"] = "test"
    result["label"] = label
    return result


def cmd_save_to_siyuan(args: argparse.Namespace) -> dict[str, Any]:
    if not SIYUAN_URL:
        _fail("SIYUAN_URL is not set.", args.format)
    block_id = _log_siyuan(args.label, args.content or f"Result saved from: {args.label}")
    return {"command": "save-to-siyuan", "label": args.label, "block_id": block_id, "success": block_id is not None}


def cmd_failures(args: argparse.Namespace) -> dict[str, Any]:
    if not RESEARCH_MEMORY_DB:
        _fail("RESEARCH_MEMORY_DB is not set.", args.format)
    tool = os.path.join(os.path.dirname(__file__), "..", "research-memory", "research_memory_tool.py")
    if not os.path.isfile(tool):
        _fail(f"research_memory_tool.py not found at {tool}", args.format)
    cmd = [sys.executable, tool, "search", "coding-failure"]
    if args.recent:
        cmd += ["--limit", str(args.recent)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"command": "failures", "output": proc.stdout.strip(), "success": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scientific coding sandbox CLI")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Execute Python code in sandbox")
    grp = p_run.add_mutually_exclusive_group()
    grp.add_argument("--code", default=None)
    grp.add_argument("--file", default=None)
    p_run.add_argument("--label", default=None)
    p_run.add_argument("--format", choices=["text", "json"], default="text")

    p_test = sub.add_parser("test", help="Run pytest on a test file in sandbox")
    p_test.add_argument("--file", required=True)
    p_test.add_argument("--format", choices=["text", "json"], default="text")

    p_siy = sub.add_parser("save-to-siyuan", help="Save a labelled result to SiYuan")
    p_siy.add_argument("--label", required=True)
    p_siy.add_argument("--content", default=None)
    p_siy.add_argument("--format", choices=["text", "json"], default="text")

    p_fail = sub.add_parser("failures", help="Show recent coding failures from research-memory")
    p_fail.add_argument("--recent", type=int, default=10)
    p_fail.add_argument("--label", default=None)
    p_fail.add_argument("--format", choices=["text", "json"], default="text")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fmt = getattr(args, "format", "text")

    dispatch = {
        "run": cmd_run,
        "test": cmd_test,
        "save-to-siyuan": cmd_save_to_siyuan,
        "failures": cmd_failures,
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
            sys.exit(1)
        if args.command == "run":
            if data.get("stdout"):
                print(data["stdout"])
            if data.get("stderr"):
                print(data["stderr"], file=sys.stderr)
            print(f"Exit: {data.get('exit_code')}  CPU: {data.get('cpu_seconds_used')}s", file=sys.stderr)
        elif args.command == "failures":
            print(data.get("output", ""))


if __name__ == "__main__":
    main()
