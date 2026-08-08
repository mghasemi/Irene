#!/usr/bin/env python3
"""P1.2 — SymEngine vs SymPy micro-benchmark baseline.

Benchmarks:
  (a) Polynomial expansion of degree-8 bivariate
  (b) 50x50 symbolic matrix multiply
  (c) Groebner basis on 3-constraint system
"""
import time
import sys

def timer(name, func):
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    print(f"  {name}: {elapsed:.4f}s")
    return result, elapsed

# ====================================================================
# (a) Polynomial expansion: degree-8 bivariate (x+y)^8 * (x^2+xy+y^2)^3
# ====================================================================
print("="*70)
print("BENCHMARK A — Degree-8 Bivariate Polynomial Expansion")
print("="*70)

import sympy as sp
x, y = sp.symbols('x y')
poly_expr = (x + y)**8 * (x**2 + x*y + y**2)**3

res_sympy_a, t_sympy_a = timer("SymPy expand", lambda: sp.expand(poly_expr))
term_count = len(sp.Poly(res_sympy_a, x, y).monoms())
print(f"  Result terms: {term_count}")

from symengine import symbols as se_symbols, expand as se_expand
x_se, y_se = se_symbols('x y')
poly_expr_se = (x_se + y_se)**8 * (x_se**2 + x_se*y_se + y_se**2)**3

_, t_symeng_a = timer("SymEngine expand", lambda: se_expand(poly_expr_se))
print(f"  Speedup: {t_sympy_a/t_symeng_a:.1f}x")

# ====================================================================
# (b) 50x50 symbolic matrix multiply with polynomial entries
# ====================================================================
print()
print("="*70)
print("BENCHMARK B — 50x50 Symbolic Matrix Multiplication")
print("="*70)

import sympy as sp
from symengine import DenseMatrix as se_DenseMatrix, symbols as se_symbols

N = 50
a, b = sp.symbols('a b')

# Build sparse-ish matrices (not full density to keep it tractable)
def make_test_matrix_sympy(n, s1, s2):
    M = []
    for i in range(n):
        row = []
        for j in range(n):
            if (i + j) % 7 == 0:
                row.append(s1**(min(i,3)) * s2**(min(j,3)))
            elif (i * j) % 11 == 0:
                row.append(s1 + s2)
            else:
                row.append(sp.Integer(0))
        M.append(row)
    return sp.Matrix(M)

def make_test_matrix_symengine(n, s1, s2):
    entries = []
    for i in range(n):
        for j in range(n):
            if (i + j) % 7 == 0:
                entries.append(s1**(min(i,3)) * s2**(min(j,3)))
            elif (i * j) % 11 == 0:
                entries.append(s1 + s2)
            else:
                entries.append(0)  # SymEngine auto-promotes int to Number
    return se_DenseMatrix(n, n, entries)

print("  Building SymPy matrix...")
t0 = time.perf_counter()
M_sp = make_test_matrix_sympy(N, a, b)
build_sp = time.perf_counter() - t0
print(f"    Build time: {build_sp:.4f}s")

print("  Multiplying SymPy matrix (M @ M)...")
_, t_sympy_b = timer("SymPy Matrix multiply", lambda: M_sp * M_sp)

a_se, b_se = se_symbols('a b')
print("  Building SymEngine matrix...")
t0 = time.perf_counter()
M_se = make_test_matrix_symengine(N, a_se, b_se)
build_se = time.perf_counter() - t0
print(f"    Build time: {build_se:.4f}s")

print("  Multiplying SymEngine matrix (M * M)...")
_, t_symeng_b = timer("SymEngine DenseMatrix multiply", lambda: M_se * M_se)
print(f"  Speedup: {t_sympy_b/t_symeng_b:.1f}x")

# ====================================================================
# (c) Groebner basis on 3-constraint system
# ====================================================================
print()
print("="*70)
print("BENCHMARK C — Groebner Basis (3 constraints, 3 variables)")
print("="*70)

from sympy import groebner as sp_groebner, symbols as sp_symbols
x_g, y_g, z_g = sp_symbols('x y z')
f1 = x_g**2 + y_g**2 + z_g**2 - 1
f2 = x_g**3 - y_g*z_g
f3 = x_g*y_g*z_g - x_g + 1

_, t_sympy_c = timer("SymPy groebner", lambda: sp_groebner([f1, f2, f3], x_g, y_g, z_g, order='lex'))

# SymEngine groebner check
try:
    from symengine import groebner as se_groebner
    x_ge, y_ge, z_ge = se_symbols('x y z')
    f1_e = x_ge**2 + y_ge**2 + z_ge**2 - 1
    f2_e = x_ge**3 - y_ge*z_ge
    f3_e = x_ge*y_ge*z_ge - x_ge + 1
    _, t_symeng_c = timer("SymEngine groebner", lambda: se_groebner([f1_e, f2_e, f3_e]))
except (ImportError, AttributeError, NotImplementedError) as e:
    print(f"  SymEngine groebner: NOT AVAILABLE — {type(e).__name__}: {e}")
    print("  → Fallback to SymPy required for Groebner operations")
    t_symeng_c = None

# ====================================================================
# Summary table
# ====================================================================
print()
print("="*70)
print("SUMMARY")
print("="*70)
print(f"{'Benchmark':<35} {'SymPy (s)':>12} {'SymEngine (s)':>14} {'Speedup':>10}")
print("-"*70)
print(f"{'(a) Poly expand deg-8 bivariate':<35} {t_sympy_a:>12.4f} {t_symeng_a:>14.4f} {t_sympy_a/t_symeng_a:>9.1f}x")
print(f"{'(b) 50x50 symbolic matrix multiply':<35} {t_sympy_b:>12.4f} {t_symeng_b:>14.4f} {t_sympy_b/t_symeng_b:>9.1f}x")
if t_symeng_c is not None:
    print(f"{'(c) Groebner basis (3 vars)':<35} {t_sympy_c:>12.4f} {t_symeng_c:>14.4f} {t_sympy_c/t_symeng_c:>9.1f}x")
else:
    print(f"{'(c) Groebner basis (3 vars)':<35} {t_sympy_c:>12.4f} {'N/A':>14} {'FALLBACK':>10}")
