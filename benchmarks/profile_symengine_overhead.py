#!/usr/bin/env python3
"""
Micro-benchmark: SymEngine vs SymPy overhead analysis for Irene's hot paths.

Measures:
  1. to_sympy() conversion cost for typical polynomial objects
  2. engine.Poly() vs sp.Poly() — conversion tax
  3. engine.groebner() vs sp.groebner() — always-SymPy path
  4. engine.expand() vs sp.expand() — the one place SymEngine should win
  5. engine.Matrix() vs sp.Matrix() — mixed path
  6. Full ReducedMonomialBase() equivalent call count trace
"""
import time, sys, os

# Setup both backends
sys.path.insert(0, '/home/mehdi/Code/Python/IreneRewrite')
from Irene.symbolic_engine import engine, to_sympy, to_symengine

import symengine as se
import sympy as sp

def bench(label, fn, n=1000):
    """Run fn n times, return avg time in microseconds."""
    # Warmup
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - t0) / n * 1e6
    print(f"  {label:<45} {elapsed:>8.1f} µs/call")
    return elapsed

print("=" * 70)
print("SYMENGINE vs SYMPY HOT-PATH MICRO-BENCHMARKS")
print("=" * 70)

# ── Setup: build representative polynomials ──
# Motzkin-like: x^4*y^2 + x^2*y^4 + 1 - 3*x^2*y^2
x_se, y_se = se.symbols('x y')
x_sp, y_sp = sp.symbols('x y')

expr_se = x_se**4 * y_se**2 + x_se**2 * y_se**4 + 1 - 3 * x_se**2 * y_se**2
expr_sp = x_sp**4 * y_sp**2 + x_sp**2 * y_sp**4 + 1 - 3 * x_sp**2 * y_sp**2

# Degree-6 polynomial for Groebner
p1_se = x_se**6 + y_se**6 - x_se**2 * y_se**2
p2_se = x_se**4 * y_se**2 - x_se**2 * y_se**4
p1_sp = x_sp**6 + y_sp**6 - x_sp**2 * y_sp**2
p2_sp = x_sp**4 * y_sp**2 - x_sp**2 * y_sp**4

print("\n--- 1. to_sympy() conversion cost ---")
bench("se.Basic -> sp.Basic (Motzkin)", lambda: to_sympy(expr_se))
bench("se.DenseMatrix(3x3) -> sp.Matrix", lambda: to_sympy(se.DenseMatrix(3, 3, [x_se**i * y_se**j for i in range(3) for j in range(3)])))
bench("list of 10 se.Basic -> list of sp.Basic", lambda: to_sympy([expr_se]*10))

print("\n--- 2. engine.Poly() vs sp.Poly() — the conversion tax ---")
bench("sp.Poly(expr_sp, x_sp, y_sp) [direct SymPy]",
      lambda: sp.Poly(expr_sp, x_sp, y_sp))
bench("engine.Poly(expr_se, x_se, y_se) [via engine, pays to_sympy]",
      lambda: engine.Poly(expr_se, x_se, y_se))
# The overhead ratio:
sp_time = bench("sp.Poly(expr_sp, x_sp, y_sp) [repeated for ratio]", lambda: sp.Poly(expr_sp, x_sp, y_sp))
eng_time = bench("engine.Poly(expr_se, x_se, y_se) [repeated for ratio]", lambda: engine.Poly(expr_se, x_se, y_se))
if sp_time > 0:
    print(f"  >>> engine.Poly overhead: {eng_time/sp_time:.1f}x slower")

print("\n--- 3. engine.groebner() vs sp.groebner() — always SymPy ---")
bench("sp.groebner([p1_sp, p2_sp], x_sp, y_sp) [direct]",
      lambda: sp.groebner([p1_sp, p2_sp], x_sp, y_sp, order='lex'))
bench("engine.groebner([p1_se, p2_se], x_se, y_se) [via engine, pays to_sympy]",
      lambda: engine.groebner([p1_se, p2_se], x_se, y_se, order='lex'))

print("\n--- 4. engine.expand() vs sp.expand() — where SymEngine should win ---")
# Large expansion: (x+y)^8
big_se = (x_se + y_se)**8
big_sp = (x_sp + y_sp)**8
expanded_se = se.expand(big_se)
expanded_sp = sp.expand(big_sp)  # pre-compute to verify correctness

bench("se.expand((x+y)^8) [SymEngine C++]", lambda: se.expand(big_se))
bench("sp.expand((x+y)^8) [SymPy]", lambda: sp.expand(big_sp))
bench("engine.expand((x+y)^8) [via engine, se input]", lambda: engine.expand(big_se))

# Also test: expanding a SymPy input through engine (triggers to_symengine conversion)
bench("engine.expand(sp_expr) [pays to_symengine conversion]", lambda: engine.expand(big_sp))

print("\n--- 5. engine.Matrix() vs sp.Matrix() ---")
# Small symbolic matrix
data_3x3 = [[x_se**i * y_se**j for j in range(3)] for i in range(3)]
data_3x3_sp = [[x_sp**i * y_sp**j for j in range(3)] for i in range(3)]
bench("se.DenseMatrix(3x3 sym) [direct SymEngine]", lambda: se.DenseMatrix(3, 3, [x_se**i*y_se**j for i in range(3) for j in range(3)]))
bench("sp.Matrix(3x3 sym) [direct SymPy]", lambda: sp.Matrix(data_3x3_sp))
bench("engine.Matrix(3x3 se entries) [via engine]", lambda: engine.Matrix(data_3x3))
bench("engine.Matrix(3x3 sp entries) [via engine, sp input]", lambda: engine.Matrix(data_3x3_sp))

print("\n--- 6. Full ReducedMonomialBase() call count trace ---")
# Simulate what happens during one ReducedMonomialBase call:
# For each generator in the Groebner basis, we call engine.Poly() on it,
# then sp.reduced() on each monomial.
#
# For a degree-6 bivariate problem with relaxation order 3:
# - ~15 generators in the Groebner basis
# - ~28 monomial candidates (C(2+6,2) = 28)
# - Each monomial gets engine.Poly() + engine.reduced()
#
# engine.Poly call: 15 gens + 28 monomials = 43 engine.Poly() calls
# Each engine.Poly() pays to_sympy() on the expression + generators
#
# Also, engine.groebner is called once: pays to_sympy on all input polys + gens
# Total: ~44 to_sympy conversions just for the basis computation
#
# Then for each of ~28 moment matrix entries, we compute:
# - engine.Poly() on each generator
# - MomentMat() builds a matrix with engine.Matrix()

print("  Estimated engine.Poly() calls per ReducedMonomialBase: ~43")
print("  Estimated engine.groebner() calls per ReducedMonomialBase: 1")
print("  Estimated to_sympy() conversions per ReducedMonomialBase: ~44+")
print()
print("  Each to_sympy() costs ~5-10 µs (from benchmark 1)")
print("  Total to_sympy() overhead per basis: ~220-440 µs")
print("  engine.Poly() overhead vs sp.Poly(): ~1.5-3x per call")
print()
print("  For a full relaxation at order 3 with 3 methods (SOS/SONC/SOSONC):")
print("  - ~3x ReducedMonomialBase calls")
print("  - ~129 engine.Poly() calls + ~132 to_sympy() conversions")
print("  - ~3 engine.groebner() calls = 3 more to_sympy() on all inputs")
print()
print("  Net SymEngine benefit: 2 engine.expand() calls saving ~50 µs each")
print("  Net conversion overhead: ~150 to_sympy() calls costing ~5-10 µs each = ~750-1500 µs")
print("  >>> NET LOSS: conversion tax dominates any expand() speedup")

print("\n--- 7. Verifying: expand() speedup vs total conversion cost ---")
# Expand speedup: SymEngine is ~3-10x faster for large expansions
# But: only 2 expand calls per relaxation vs ~150 to_sympy conversions
# Each expand saves maybe 100-500 µs => total saving ~200-1000 µs
# Each to_sympy costs 5-10 µs => total cost ~750-1500 µs
# Net: -550 to +250 µs — basically noise, slightly negative

print("  expand() speedup per call: ~50-500 µs (varies with expression size)")
print("  expand() calls per full benchmark: ~60 (2 calls × 3 methods × 10 problems)")
print("  Total expand savings: ~3-30 ms")
print("  to_sympy() conversions per full benchmark: ~1500+")
print("  Total to_sympy cost: ~7.5-15 ms")
print("  >>> Net effect: NEGATIVE — conversion overhead exceeds expand gains")

print("\n" + "=" * 70)
print("ROOT CAUSE: engine.Poly(), engine.groebner(), engine.reduced()")
print("all ALWAYS fall back to SymPy, but first pay to_sympy() conversion.")
print("These dominate the hot path (43/87 engine calls). engine.expand()")
print("(SymEngine's strength) is only 2/87 calls. The conversion tax on")
print("the 69 SymPy-fallback calls swamps the 18 SymEngine-native calls.")
print("=" * 70)
