---
name: sympy-mcp
description: "Use when performing exact symbolic mathematics: solving equations, computing derivatives/integrals, solving ODEs, factoring, simplifying, or converting expressions to LaTeX. Returns exact closed-form results with no floating-point error."
metadata: {"clawdbot":{"emoji":"∂","requires":{"bins":["python3"],"pypackages":["sympy"]},"config":{"env":{}}}}
---
# SymPy Symbolic Computation

Use this skill for any exact symbolic mathematics. SymPy runs locally — no network or API key required.

## When to Use

- Solve algebraic or transcendental equations exactly.
- Compute derivatives, indefinite or definite integrals.
- Solve ordinary differential equations (ODEs).
- Factor or expand polynomials over the rationals.
- Simplify trigonometric, exponential, or logarithmic expressions.
- Convert an expression to a clean LaTeX string for manuscript insertion.

## When Not to Use

- Do not use for floating-point numerics — use SageMath's precision arithmetic for that.
- Do not use for abstract algebraic structures (rings, fields, modules) — use sagemath-mcp.
- Do not use for formal proofs — use Lean4.

## Commands

Run the tool from this skill folder.

### Solve an equation

```bash
python3 {baseDir}/sympy_tool.py solve "x**2 - 4"
python3 {baseDir}/sympy_tool.py solve "x**2 + 2*x + 1" --var x --format json
```

### Differentiate

```bash
python3 {baseDir}/sympy_tool.py diff "x**3 * sin(x)" --var x
python3 {baseDir}/sympy_tool.py diff "exp(-x**2)" --var x --n 2
```

### Integrate

```bash
python3 {baseDir}/sympy_tool.py integrate "sin(x)**2" --var x
python3 {baseDir}/sympy_tool.py integrate "exp(-x**2)" --var x --limits "-oo oo"
```

### Solve ODE

```bash
python3 {baseDir}/sympy_tool.py dsolve "f(x).diff(x) - f(x)"
python3 {baseDir}/sympy_tool.py dsolve "f(x).diff(x,2) + f(x)" --func f
```

### Simplify / factor / expand

```bash
python3 {baseDir}/sympy_tool.py simplify "sin(x)**2 + cos(x)**2"
python3 {baseDir}/sympy_tool.py factor "x**3 - x**2 + x - 1"
python3 {baseDir}/sympy_tool.py expand "(x+1)**4"
```

### Convert to LaTeX

```bash
python3 {baseDir}/sympy_tool.py latex "Integral(sin(x)**2, (x, 0, pi))"
python3 {baseDir}/sympy_tool.py latex "sqrt(2)/2 + I"
```

## Output

Every command returns a plain-text result plus a `latex` field. Use `--format json` to get a structured object:

```json
{
  "command": "solve",
  "expression": "x**2 - 4",
  "result": "[-2, 2]",
  "latex": "\\left[ -2, \\  2\\right]",
  "success": true
}
```

## Installation

```bash
pip install sympy
```

No environment variables required.
