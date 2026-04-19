#!/usr/bin/env python3
"""Generic Lean4 command wrapper for checks, search, and proof attempts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT = 60.0
PROJECT_MARKERS = ("lakefile.lean", "lakefile.toml", "lakefile.json", "lean-toolchain")
DEFAULT_SCRATCH_PROJECT_NAME = "Lean4Scratch"
TACTIC_CANDIDATES = (
    "first | exact? | aesop?",
    "simp",
    "norm_num",
    "ring",
    "linarith",
    "nlinarith",
    "positivity",
    "omega",
    "try polyrith",
)


@dataclass
class ToolConfig:
    lean_bin: str
    lake_bin: str
    timeout: float
    strict_project: bool
    allow_scratch: bool
    scratch_root: Path
    progress: bool
    format: str


def parse_bool(text: str | None, default: bool = False) -> bool:
    if text is None:
        return default
    lowered = text.strip().lower()
    return lowered in ("1", "true", "yes", "on")


def resolve_binary(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    fallback = Path.home() / ".elan" / "bin" / name
    if fallback.exists():
        return str(fallback)
    raise RuntimeError(f"Missing required executable: {name}")


def is_project_dir(path: Path) -> bool:
    return any((path / marker).exists() for marker in PROJECT_MARKERS)


def discover_project(start: Path) -> Path | None:
    current = start.resolve()
    while True:
        if is_project_dir(current):
            return current
        if current.parent == current:
            return None
        current = current.parent


def resolve_project_dir(args: argparse.Namespace) -> tuple[Path, str]:
    if args.project_dir:
        project = Path(args.project_dir).expanduser().resolve()
        return project, "cli"

    env_dir = os.getenv("LEAN4_PROJECT_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve(), "env"

    cwd = Path.cwd()
    discovered = discover_project(cwd)
    if discovered is not None:
        return discovered, "auto"

    return cwd, "cwd"


def run_command(command: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def decode_maybe_bytes(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def progress_log(config: ToolConfig, message: str) -> None:
    if not config.progress:
        return
    print(f"[lean4_tool] {message}", file=sys.stderr, flush=True)


def record_phase(phases: list[dict[str, str]], phase: str, status: str, detail: str = "") -> None:
    phases.append({"phase": phase, "status": status, "detail": detail})


def result_payload(
    *,
    command: list[str],
    cwd: Path,
    process: subprocess.CompletedProcess[str],
    project_source: str,
    operation: str,
) -> dict[str, Any]:
    status = "ok" if process.returncode == 0 else "error"
    return {
        "operation": operation,
        "status": status,
        "exit_code": process.returncode,
        "project_dir": str(cwd),
        "project_source": project_source,
        "command": command,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def render_payload(payload: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(payload, indent=2, ensure_ascii=True)
    lines = [
        f"operation: {payload['operation']}",
        f"status: {payload['status']}",
        f"exit_code: {payload['exit_code']}",
        f"project_dir: {payload['project_dir']} ({payload['project_source']})",
        f"command: {' '.join(payload['command'])}",
    ]
    if payload.get("stdout"):
        lines.append("stdout:")
        lines.append(payload["stdout"].rstrip())
    if payload.get("stderr"):
        lines.append("stderr:")
        lines.append(payload["stderr"].rstrip())
    return "\n".join(lines)


def init_scratch_project(config: ToolConfig, phases: list[dict[str, str]]) -> Path:
    scratch_root = config.scratch_root.expanduser().resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_project = scratch_root / DEFAULT_SCRATCH_PROJECT_NAME
    progress_log(config, f"scratch root: {scratch_root}")
    record_phase(phases, "scratch_root", "ok", str(scratch_root))

    if is_project_dir(scratch_root):
        active_project = scratch_root
        progress_log(config, "using in-place scratch project")
        record_phase(phases, "scratch_discovery", "ok", "in-place")
    elif is_project_dir(scratch_project):
        active_project = scratch_project
        progress_log(config, f"using nested scratch project: {scratch_project}")
        record_phase(phases, "scratch_discovery", "ok", "nested")
    else:
        progress_log(config, "initializing scratch Lake+Mathlib project")
        record_phase(phases, "scratch_init", "start", str(scratch_project))
        init_command = [config.lake_bin, "init", DEFAULT_SCRATCH_PROJECT_NAME, "math"]
        init_result = run_command(init_command, scratch_root, max(config.timeout, 300.0))
        if init_result.returncode != 0 and "package already initialized" not in (init_result.stderr or ""):
            record_phase(phases, "scratch_init", "error", (init_result.stderr or init_result.stdout or "lake init failed").strip())
            message = (init_result.stderr or init_result.stdout or "lake init failed").strip()
            raise RuntimeError(f"Unable to initialize Lean scratch project: {message}")
        record_phase(phases, "scratch_init", "ok", "initialized")

        if is_project_dir(scratch_project):
            active_project = scratch_project
            progress_log(config, f"initialized nested scratch project: {scratch_project}")
            record_phase(phases, "scratch_discovery", "ok", "nested_post_init")
        elif is_project_dir(scratch_root):
            active_project = scratch_root
            progress_log(config, "initialized in-place scratch project")
            record_phase(phases, "scratch_discovery", "ok", "in_place_post_init")
        else:
            record_phase(phases, "scratch_discovery", "error", "project not found after init")
            raise RuntimeError("Unable to locate scratch Lean project after initialization")

    ready_marker = active_project / ".lean4_tool_ready"
    if not ready_marker.exists():
        progress_log(config, "warming dependencies: lake exe cache get")
        record_phase(phases, "scratch_cache", "start", str(active_project))
        cache_command = [config.lake_bin, "exe", "cache", "get"]
        cache_result = run_command(cache_command, active_project, max(config.timeout, 300.0))
        if cache_result.returncode == 0:
            record_phase(phases, "scratch_cache", "ok", "cache fetched")
        else:
            record_phase(phases, "scratch_cache", "warn", "cache unavailable, continuing to build")

        progress_log(config, "building scratch project: lake build")
        record_phase(phases, "scratch_build", "start", str(active_project))
        build_command = [config.lake_bin, "build"]
        build_result = run_command(build_command, active_project, max(config.timeout, 600.0))
        if build_result.returncode != 0:
            record_phase(phases, "scratch_build", "error", (build_result.stderr or build_result.stdout or "lake build failed").strip())
            message = (build_result.stderr or build_result.stdout or "lake build failed").strip()
            raise RuntimeError(f"Unable to prepare Lean scratch project: {message}")
        record_phase(phases, "scratch_build", "ok", "build complete")

        ready_marker.write_text("ready\n", encoding="utf-8")
        progress_log(config, "scratch project ready")
        record_phase(phases, "scratch_ready", "ok", str(ready_marker))
    else:
        progress_log(config, "scratch project already prepared")
        record_phase(phases, "scratch_ready", "ok", "already_prepared")

    return active_project


def ensure_project_if_needed(project_dir: Path, config: ToolConfig, needs_project: bool) -> tuple[Path, str, list[dict[str, str]]]:
    phases: list[dict[str, str]] = []
    if not needs_project:
        record_phase(phases, "project_resolution", "ok", "project_not_required")
        return project_dir, "unchanged", phases
    if is_project_dir(project_dir):
        record_phase(phases, "project_resolution", "ok", f"existing_project:{project_dir}")
        return project_dir, "unchanged", phases
    if config.allow_scratch and not config.strict_project:
        record_phase(phases, "project_resolution", "start", "using_scratch_fallback")
        scratch_project = init_scratch_project(config, phases)
        record_phase(phases, "project_resolution", "ok", f"scratch_project:{scratch_project}")
        return scratch_project, "scratch", phases
    record_phase(phases, "project_resolution", "error", "no_project_and_scratch_disabled")
    return project_dir, "error", phases


def write_temp_lean(project_dir: Path, content: str) -> Path:
    tmp_dir = project_dir / ".lean4_tmp"
    tmp_dir.mkdir(exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="session_", suffix=".lean", dir=str(tmp_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.write_text(content, encoding="utf-8")
    return tmp_path


def check_command(args: argparse.Namespace, config: ToolConfig, project_dir: Path, project_source: str) -> dict[str, Any]:
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = (project_dir / file_path).resolve()
        command = [config.lake_bin, "env", config.lean_bin, str(file_path)]
    else:
        command = [config.lake_bin, "build"]
    process = run_command(command, project_dir, config.timeout)
    return result_payload(
        command=command,
        cwd=project_dir,
        process=process,
        project_source=project_source,
        operation="check",
    )


def repl_command(args: argparse.Namespace, config: ToolConfig, project_dir: Path, project_source: str) -> dict[str, Any]:
    content = args.code
    if args.mathlib:
        content = "import Mathlib\n\n" + content
    tmp_path = write_temp_lean(project_dir, content)
    try:
        if is_project_dir(project_dir):
            command = [config.lake_bin, "env", config.lean_bin, str(tmp_path)]
        else:
            command = [config.lean_bin, str(tmp_path)]
        process = run_command(command, project_dir, config.timeout)
        return result_payload(
            command=command,
            cwd=project_dir,
            process=process,
            project_source=project_source,
            operation="repl",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def search_command(args: argparse.Namespace, config: ToolConfig, project_dir: Path, project_source: str) -> dict[str, Any]:
    snippet = "\n".join(
        (
            "import Mathlib",
            "",
            f"#check {args.query}",
            "",
            f"example : {args.query} := by",
            "  exact?",
            "",
            f"example : {args.query} := by",
            "  apply?",
            "",
        )
    )
    tmp_path = write_temp_lean(project_dir, snippet)
    try:
        command = [config.lake_bin, "env", config.lean_bin, str(tmp_path)]
        process = run_command(command, project_dir, config.timeout)
        payload = result_payload(
            command=command,
            cwd=project_dir,
            process=process,
            project_source=project_source,
            operation="search",
        )
        payload["query"] = args.query
        return payload
    finally:
        tmp_path.unlink(missing_ok=True)


def prove_command(args: argparse.Namespace, config: ToolConfig, project_dir: Path, project_source: str) -> dict[str, Any]:
    tactic_block = "\n  | ".join(TACTIC_CANDIDATES)
    snippet = "\n".join(
        (
            "import Mathlib",
            "",
            f"example : {args.statement} := by",
            "  first",
            f"  | {tactic_block}",
            "",
        )
    )
    tmp_path = write_temp_lean(project_dir, snippet)
    try:
        command = [config.lake_bin, "env", config.lean_bin, str(tmp_path)]
        process = run_command(command, project_dir, config.timeout)
        payload = result_payload(
            command=command,
            cwd=project_dir,
            process=process,
            project_source=project_source,
            operation="prove",
        )
        payload["statement"] = args.statement
        payload["tactics"] = list(TACTIC_CANDIDATES)
        return payload
    finally:
        tmp_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic Lean4 helper")
    parser.add_argument("--project-dir", help="Lean project directory (Lake root)")
    parser.add_argument("--timeout", type=float, help="Command timeout in seconds")
    parser.add_argument("--strict-project", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-scratch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scratch-root", help="Scratch project root directory")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--format", choices=("json", "text"), default="json")

    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Run Lean/Lake checks")
    check_parser.add_argument("file", nargs="?", help="Optional Lean file to check")

    repl_parser = subparsers.add_parser("repl", help="Run ad hoc Lean code")
    repl_parser.add_argument("code", help="Lean code snippet")
    repl_parser.add_argument("--mathlib", action=argparse.BooleanOptionalAction, default=True)

    search_parser = subparsers.add_parser("search", help="Search candidate lemmas for a target proposition")
    search_parser.add_argument("query", help="Term or proposition to inspect")

    prove_parser = subparsers.add_parser("prove", help="Attempt automated proof tactics")
    prove_parser.add_argument("statement", help="Proposition statement to attempt")

    return parser


def build_config(args: argparse.Namespace) -> ToolConfig:
    timeout_raw = args.timeout if args.timeout is not None else os.getenv("LEAN4_TIMEOUT", str(DEFAULT_TIMEOUT))
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid LEAN4_TIMEOUT value: {timeout_raw}") from exc

    strict_project = (
        args.strict_project
        if args.strict_project is not None
        else parse_bool(os.getenv("LEAN4_STRICT_PROJECT"), default=False)
    )

    allow_scratch = (
        args.allow_scratch
        if args.allow_scratch is not None
        else parse_bool(os.getenv("LEAN4_ALLOW_SCRATCH"), default=True)
    )

    scratch_root_raw = args.scratch_root or os.getenv("LEAN4_SCRATCH_ROOT", str(Path.home() / ".cache" / "lean4_tool"))
    scratch_root = Path(scratch_root_raw).expanduser()

    progress = (
        args.progress
        if args.progress is not None
        else parse_bool(os.getenv("LEAN4_PROGRESS"), default=True)
    )

    return ToolConfig(
        lean_bin=resolve_binary("lean"),
        lake_bin=resolve_binary("lake"),
        timeout=timeout,
        strict_project=strict_project,
        allow_scratch=allow_scratch,
        scratch_root=scratch_root,
        progress=progress,
        format=args.format,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    phases: list[dict[str, str]] = []
    try:
        config = build_config(args)
        project_dir, project_source = resolve_project_dir(args)
        if not project_dir.exists():
            raise RuntimeError(f"Project directory does not exist: {project_dir}")

        needs_project = args.command in ("search", "prove") or (args.command == "check" and args.file is None)
        project_dir, project_adjustment, phase_updates = ensure_project_if_needed(project_dir, config, needs_project)
        phases.extend(phase_updates)
        if project_adjustment == "error":
            raise RuntimeError("No Lean project markers found. Set LEAN4_PROJECT_DIR or pass --project-dir to a Lake project.")
        if project_adjustment == "scratch":
            project_source = "scratch"
            progress_log(config, f"resolved project via scratch fallback: {project_dir}")

        if args.command == "check":
            payload = check_command(args, config, project_dir, project_source)
        elif args.command == "repl":
            payload = repl_command(args, config, project_dir, project_source)
        elif args.command == "search":
            payload = search_command(args, config, project_dir, project_source)
        elif args.command == "prove":
            payload = prove_command(args, config, project_dir, project_source)
        else:
            parser.error(f"Unsupported command: {args.command}")
            return 2

        payload["phases"] = phases
        print(render_payload(payload, config.format))
        return 0 if payload.get("exit_code") == 0 else 1
    except subprocess.TimeoutExpired as exc:
        payload = {
            "operation": args.command if hasattr(args, "command") else "unknown",
            "status": "timeout",
            "exit_code": -1,
            "project_dir": str(resolve_project_dir(args)[0]) if hasattr(args, "project_dir") else str(Path.cwd()),
            "project_source": "unknown",
            "command": exc.cmd if isinstance(exc.cmd, list) else [str(exc.cmd)],
            "stdout": decode_maybe_bytes(exc.stdout),
            "stderr": decode_maybe_bytes(exc.stderr) or f"Command timed out after {exc.timeout} seconds",
            "phases": phases,
        }
        print(render_payload(payload, getattr(args, "format", "json")))
        return 1
    except RuntimeError as exc:
        payload = {
            "operation": args.command if hasattr(args, "command") else "unknown",
            "status": "error",
            "exit_code": 1,
            "project_dir": str(Path.cwd()),
            "project_source": "unknown",
            "command": [],
            "stdout": "",
            "stderr": str(exc),
            "phases": phases,
        }
        print(render_payload(payload, getattr(args, "format", "json")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
