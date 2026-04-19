---
name: wolfram-alpha
description: "Use when querying Wolfram|Alpha for concise computational answers, validating whether a mathematical input is understood, or inspecting pods, assumptions, and warnings to verify mathematical statements."
metadata: {"clawdbot":{"emoji":"∫","requires":{"bins":["python3"]},"config":{"env":{"WOLFRAM_ALPHA_APPID":{"description":"Wolfram|Alpha AppID used for Research_mgh or another registered app","default":"","required":true},"WOLFRAM_ALPHA_TIMEOUT":{"description":"HTTP timeout in seconds","default":"20","required":false},"WOLFRAM_ALPHA_RESULT_URL":{"description":"Short answer endpoint","default":"https://api.wolframalpha.com/v1/result","required":false},"WOLFRAM_ALPHA_QUERY_URL":{"description":"Full results endpoint","default":"https://api.wolframalpha.com/v2/query","required":false},"WOLFRAM_ALPHA_VALIDATE_URL":{"description":"Validation endpoint","default":"https://api.wolframalpha.com/v2/validatequery","required":false}}}}}
---
# Wolfram|Alpha Knowledge And Verification

Use this skill when you want Wolfram|Alpha to act as a computational reference or a verification aid for mathematical statements.

## When to Use

- Get a short computational answer for a math query.
- Inspect pods, assumptions, warnings, and alternate interpretations for a mathematical statement.
- Check whether a query is likely to be understood before asking for the full result.
- Verify whether Wolfram|Alpha changed the interpretation of an ambiguous expression.

## When Not to Use

- Do not treat Wolfram|Alpha output as a formal proof.
- Do not rely on concise answers alone when interpretation ambiguity matters.
- Do not use this skill for image rendering in phase 1; use text and structured output first.

## Commands

Run the tool directly from this skill folder.

### Short answer

```bash
python3 {baseDir}/wolfram_alpha_tool.py answer "integrate x^2"
python3 {baseDir}/wolfram_alpha_tool.py answer "prime factors of 360" --format json
```

### Structured verification

```bash
python3 {baseDir}/wolfram_alpha_tool.py verify "sin(30)" --profile symbolic
python3 {baseDir}/wolfram_alpha_tool.py verify "log 20" --profile symbolic --format json
python3 {baseDir}/wolfram_alpha_tool.py verify "12/11/1996" --assumption "DateOrder_**Day.Month.Year--"
python3 {baseDir}/wolfram_alpha_tool.py verify "Pythagorean theorem" --profile theorem
python3 {baseDir}/wolfram_alpha_tool.py verify "pi" --profile symbolic --podstate "DecimalApproximation__More digits"
```

### Parse-only validation

```bash
python3 {baseDir}/wolfram_alpha_tool.py validate "x^2 + y^2 = 1"
python3 {baseDir}/wolfram_alpha_tool.py validate "log 0.5" --format json
```

## Verification Guidance

The `verify` command is the default high-level workflow.

1. Run the original query without extra assumptions first.
2. Inspect warnings and assumptions before trusting a result for an ambiguous statement.
3. Use `--profile symbolic` for compact computational and algebraic checks.
4. Use `--profile theorem` for statement-style inputs where definition or property pods matter.
5. Re-run with explicit `--assumption` or `--podstate` tokens when the interpretation matters.
6. Prefer `--format json` when downstream tooling needs to inspect the exact interpretation.

## Configuration

Environment-first credential handling:

- `WOLFRAM_ALPHA_APPID`: required AppID.
- `WOLFRAM_ALPHA_TIMEOUT`: defaults to `20`.
- `WOLFRAM_ALPHA_RESULT_URL`: defaults to `https://api.wolframalpha.com/v1/result`.
- `WOLFRAM_ALPHA_QUERY_URL`: defaults to `https://api.wolframalpha.com/v2/query`.
- `WOLFRAM_ALPHA_VALIDATE_URL`: defaults to `https://api.wolframalpha.com/v2/validatequery`.

Local setup example for your registered app:

```bash
export WOLFRAM_ALPHA_APPID='K7EULAUY95'
python3 {baseDir}/wolfram_alpha_tool.py answer "integrate x^2"
```

The AppID is documented for your local setup, but it is not embedded as the repository default.

## Output

- `answer` returns a short answer string or a compact JSON wrapper.
- `verify` returns normalized result metadata, pods, assumptions, warnings, and sources.
- `validate` returns parse-only success metadata with warnings and assumptions when available.
- `verify --profile symbolic` biases output toward compact computational pods.
- `verify --profile theorem` biases output toward statement and property-style pods.

Use `--format json` whenever another tool or agent needs stable machine-readable output.