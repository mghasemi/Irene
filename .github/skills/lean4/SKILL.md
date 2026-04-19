---
name: lean4
description: "Use when checking Lean4 files, searching Mathlib lemmas, exploring goal states, or attempting machine-checked proofs in a Lean project."
metadata: {"clawdbot":{"emoji":"\u03bb","requires":{"bins":["python3"]},"config":{"env":{"LEAN4_PROJECT_DIR":{"description":"Default Lean project directory. Used when --project-dir is not provided.","default":"","required":false},"LEAN4_TIMEOUT":{"description":"Lean command timeout in seconds","default":"60","required":false},"LEAN4_STRICT_PROJECT":{"description":"Require a Lake/Lean project for commands that import Mathlib","default":"false","required":false},"LEAN4_ALLOW_SCRATCH":{"description":"Allow auto-creation of a scratch Lake+Mathlib project when no project is found","default":"true","required":false},"LEAN4_SCRATCH_ROOT":{"description":"Root folder used for the auto-created scratch project","default":"~/.cache/lean4_tool","required":false},"LEAN4_PROGRESS":{"description":"Emit progress logs to stderr during scratch initialization and routing","default":"true","required":false}}}}}
---
# Lean4 Verification And Proof Support

Use this skill for repository-agnostic Lean4 workflows. This skill does not encode domain-specific theorem families.

## When to Use

- Check whether Lean files compile.
- Search for candidate Mathlib lemmas for a target statement.
- Probe a Lean goal and inspect tactic suggestions.
- Attempt a first-pass proof with standard automation tactics.

## Commands

Run the tool directly from this skill folder.

### Check a file or project

```bash
python3 {baseDir}/lean4_tool.py check --project-dir /path/to/project
python3 {baseDir}/lean4_tool.py check Theorems/L_T1.lean --project-dir /path/to/project
```

### Run ad hoc Lean code

```bash
python3 {baseDir}/lean4_tool.py repl "#check Nat.succ"
python3 {baseDir}/lean4_tool.py repl "example : 2 + 2 = 4 := by decide" --mathlib
```

### Search for lemma candidates

```bash
python3 {baseDir}/lean4_tool.py search "forall x : R, x * x >= 0"
python3 {baseDir}/lean4_tool.py search "Nat.succ"
python3 {baseDir}/lean4_tool.py search "forall a b : Nat, a + b = b + a" --allow-scratch
```

### Attempt an automated proof sketch

```bash
python3 {baseDir}/lean4_tool.py prove "forall x : \u211d, x^2 >= 0"
python3 {baseDir}/lean4_tool.py prove "forall a b : \u211d, a + b = b + a"
```

## Output

- Commands return structured JSON by default.
- Use `--format text` for a compact human-readable summary.
- JSON payload includes the resolved project directory, command, exit code, stdout, stderr, and a simple status label.
- JSON payload also includes `phases`, a machine-readable list of lifecycle events (for example project resolution, scratch init/cache/build, and readiness state).

## Configuration

- `LEAN4_PROJECT_DIR`: optional default project path.
- `LEAN4_TIMEOUT`: command timeout in seconds (default `60`).
- `LEAN4_STRICT_PROJECT`: when `true`, commands requiring project context fail if no project root is found.
- `LEAN4_ALLOW_SCRATCH`: when `true`, `search`/`prove` create and use a scratch Lean+Mathlib project when needed.
- `LEAN4_SCRATCH_ROOT`: root directory for the scratch project (default `~/.cache/lean4_tool`).
- `LEAN4_PROGRESS`: when `true`, emits phase logs to stderr (`init`, `cache`, `build`, and project resolution).

## Notes

- The skill is designed to be reusable across repos.
- Domain-specific routing should live in prompts/agents, not in this tool.
