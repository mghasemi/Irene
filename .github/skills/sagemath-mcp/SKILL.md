---
name: sagemath-mcp
description: "Use for advanced algebraic structures (rings, fields, groups, number fields), arbitrary-precision arithmetic, matrix theory (eigenvalues, Jordan form), or any computation requiring SageMath's mathematical rigour. Falls back to SymPy when SageMath is absent."
metadata: {"clawdbot":{"emoji":"🔢","requires":{"bins":["python3"]},"optional_bins":["sage"],"config":{"env":{"SAGE_TIMEOUT":{"description":"Timeout in seconds for sage subprocess calls","default":"60","required":false}}}}}
---
# SageMath Symbolic & Algebraic Computation

Use this skill when you need SageMath's advanced mathematical environments: arbitrary-precision fields, abstract algebra, number theory, or matrix computations beyond SymPy's scope.

## When to Use

- Construct and manipulate polynomial rings, quotient rings, or finite fields.
- Compute eigenvalues, Jordan canonical forms, or Smith normal forms of matrices.
- Work with algebraic number fields, Galois groups, or integer lattices.
- Perform arithmetic in a `RealField(prec)` or `ComplexField(prec)` with arbitrary precision.
- Run SageMath scripts that are too large for inline `sage -c`.

## When Not to Use

- For basic calculus (derivatives, integrals, ODEs), prefer sympy-mcp — it is faster.
- For formal proofs, use lean4.
- For external data retrieval, use lightrag-query or academic-research-hub.

## Fallback Behaviour

If `sage` is not on `PATH`, the tool falls back to SymPy for supported operations and returns a `"fallback": true` flag in the JSON output. Install SageMath for full functionality.

## Commands

### Ring and field operations

```bash
python3 {baseDir}/sagemath_tool.py ring-ops "R.<x> = QQ[]; f = x^4 - 1; print(f.factor())"
python3 {baseDir}/sagemath_tool.py ring-ops "GF(7^2)" --format json
```

### Matrix operations

```bash
python3 {baseDir}/sagemath_tool.py matrix "A = matrix(QQ, [[1,2],[3,4]]); print(A.eigenvalues())"
python3 {baseDir}/sagemath_tool.py matrix "A = matrix(ZZ, [[6,4],[1,3]]); print(A.jordan_form())" --format json
```

### Arbitrary-precision arithmetic

```bash
python3 {baseDir}/sagemath_tool.py precision-arith "RR = RealField(100); print(RR(pi))"
python3 {baseDir}/sagemath_tool.py precision-arith "RR = RealField(200); print(RR(2).sqrt())"
```

### Number field operations

```bash
python3 {baseDir}/sagemath_tool.py number-field "K.<a> = NumberField(x^2 - 2); print(K.discriminant())"
python3 {baseDir}/sagemath_tool.py number-field "K.<z> = CyclotomicField(5); print(K.galois_group())"
```

### Run an arbitrary Sage script file

```bash
python3 {baseDir}/sagemath_tool.py run-script /path/to/script.sage
python3 {baseDir}/sagemath_tool.py run-script /path/to/script.sage --format json
```

## Output

```json
{
  "command": "matrix",
  "script": "A = matrix(QQ, [[1,2],[3,4]]); print(A.eigenvalues())",
  "stdout": "[-0.3722813..., 5.3722813...]",
  "stderr": "",
  "fallback": false,
  "success": true
}
```

## Configuration

- `SAGE_TIMEOUT`: subprocess timeout for `sage -c` calls. Defaults to `60` seconds.

## Installation

```bash
# Ubuntu / Debian
sudo apt install sagemath
# or via conda
conda install -c conda-forge sage
```
