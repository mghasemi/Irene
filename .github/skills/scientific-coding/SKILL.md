---
name: scientific-coding
description: "Use when running Python experiments, simulations, or data analyses in a secure resource-limited sandbox. Captures stdout, stderr, and execution metadata. Auto-logs results to SiYuan and records failure context in research-memory to prevent redundant re-runs."
metadata: {"clawdbot":{"emoji":"🧪","requires":{"bins":["python3"]},"config":{"env":{"SANDBOX_CPU_SECONDS":{"description":"CPU time limit per run in seconds","default":"30","required":false},"SANDBOX_MEM_MB":{"description":"Memory limit per run in megabytes","default":"512","required":false},"SIYUAN_URL":{"description":"SiYuan API URL for auto-logging results","default":"","required":false},"SIYUAN_TOKEN":{"description":"SiYuan API token","default":"","required":false},"RESEARCH_MEMORY_DB":{"description":"Path to research_memory SQLite DB for failure logging","default":"","required":false}}}}}
---
# Scientific Coding Sandbox

Use this skill to execute Python code for empirical experimentation in a resource-limited, isolated environment. The sandbox prevents runaway processes and network access. Results are auto-logged to SiYuan; failures are recorded to research-memory to prevent redundant re-runs.

## When to Use

- Test a numerical algorithm or optimization heuristic.
- Generate statistical distributions, empirical data, or plots.
- Run a pytest suite against a newly generated module.
- Validate symbolic results from sympy-mcp or sagemath-mcp empirically.

## When Not to Use

- Do not use for symbolic computation — use sympy-mcp or sagemath-mcp.
- Do not use to compile LaTeX — use latex-manuscript.
- Do not use for tasks requiring network access (the sandbox blocks outbound connections).

## Commands

### Run inline code

```bash
python3 {baseDir}/scientific_coding_tool.py run --code "import math; print(math.factorial(20))"
python3 {baseDir}/scientific_coding_tool.py run --code "$(cat experiment.py)" --format json
```

### Run a script file

```bash
python3 {baseDir}/scientific_coding_tool.py run --file /path/to/experiment.py
python3 {baseDir}/scientific_coding_tool.py run --file /path/to/experiment.py --label "gradient-descent-v1"
```

### Run pytest in sandbox

```bash
python3 {baseDir}/scientific_coding_tool.py test --file /path/to/test_module.py
python3 {baseDir}/scientific_coding_tool.py test --file /path/to/test_module.py --format json
```

### Save last result to SiYuan

```bash
python3 {baseDir}/scientific_coding_tool.py save-to-siyuan --label "gradient-descent-v1" --notebook "MathAgent Experiments"
```

### Check failure log (what has already been tried)

```bash
python3 {baseDir}/scientific_coding_tool.py failures --recent 10
python3 {baseDir}/scientific_coding_tool.py failures --label "gradient-descent"
```

## Output

`run` returns:

```json
{
  "command": "run",
  "label": "gradient-descent-v1",
  "stdout": "Converged in 47 iterations. Loss: 0.00012",
  "stderr": "",
  "exit_code": 0,
  "cpu_seconds_used": 1.3,
  "memory_mb_peak": 24,
  "success": true,
  "siyuan_block_id": "20260425143200-abc123"
}
```

On resource limit exceeded:

```json
{
  "success": false,
  "error": "CPU time limit exceeded (30s)",
  "failure_logged": true
}
```

## Resource Limits

The sandbox enforces:
- CPU time: `SANDBOX_CPU_SECONDS` (default 30 s) via `resource.RLIMIT_CPU`.
- Virtual memory: `SANDBOX_MEM_MB` (default 512 MB) via `resource.RLIMIT_AS`.
- No outbound network: sandbox runs with `--network none` semantics (blocks socket creation).

## Auto-Logging

- On **success**: result is posted to SiYuan if `SIYUAN_URL` and `SIYUAN_TOKEN` are set.
- On **failure**: error + code snippet are stored in `research_memory` under category `coding-failure` if `RESEARCH_MEMORY_DB` is set. The orchestrator queries this log before re-running to avoid identical failed attempts.
