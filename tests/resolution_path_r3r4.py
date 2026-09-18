"""Resolution path — Strategies 3 & 4 (reduced generators + combined).

Phase 1 found: Higher order alone does NOT help (o=3 identical to baseline -3.7855).
Phase 2 found: Box |x|<=1.5, o=3 => -1.065 (gap +0.92) is the best bounding strategy.

This script tests:
- S3: Remove derivative variables from generators (no ADE machinery at all)
- S4: Combined — reduced generators + Box |x|<=1.5 bounds (best of both worlds)
"""

import time
from math import exp, sqrt

import numpy as np
from scipy.optimize import minimize

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxation_api import RelaxationEngine
from Irene.relaxations import RelaxationConfig


# ====================================================================
# SciPy reference
# ====================================================================
def solve_with_scipy():
    def objective(point):
        xv, yv = point
        return 1.0 - xv**2 + yv

    def constraint_values(point):
        xv, yv = point
        radicand = xv * yv + 1.0
        root = sqrt(max(radicand, 0.0))
        exponential = exp(np.clip(xv * yv, -700.0, 700.0))
        return np.array([
            radicand,
            root + xv / 2.0 - xv**2 + 2.0 * yv,
            2.0 - yv**2,
            yv - xv * exponential + 1.0,
        ])

    constraints = {"type": "ineq", "fun": constraint_values}
    starts = [
        (xv, yv)
        for xv in np.linspace(-3.0, 3.0, 7)
        for yv in np.linspace(-sqrt(2.0), sqrt(2.0), 5)
        if xv * yv + 1.0 >= 0.0
    ]
    results = []
    for start in starts:
        r = minimize(objective, np.asarray(start, dtype=float),
                     method="SLSQP",
                     bounds=[(-10., 10.), (-sqrt(2.), sqrt(2.))],
                     constraints=constraints,
                     options={"ftol": 1e-10, "maxiter": 1000})
        if r.success and np.min(constraint_values(r.x)) >= -1e-7:
            results.append(r)
    return min(results, key=lambda item: item.fun) if results else None


# ====================================================================
# Strategy 3: No derivative variables (pure algebraic formulation)
# ====================================================================
def make_sga_3gen():
    """Reduced 3-generator setup: only x, y, u.

    Removes v, d_vx, d_vy, d_vu entirely — no ADE machinery.
    The exponential constraint y - x*v + 1 >= 0 is dropped (v not present).
    This tests whether the algebraic structure alone can certify nonnegativity.
    """
    names = ["x", "y", "u"]
    sg = CommutativeSemigroup(names)
    sga = SemigroupAlgebra(sg)

    x, y, u = (sga[n] for n in names)

    relations = [u**2 - (x * y + 1)]

    prog = OptimizationProblem(sga, relations=relations)
    prog.set_objective(1 - x**2 + y)
    prog.add_constraints([
        u + x / 2 - x**2 + 2 * y,
        2 - y**2,
    ])

    return prog


# ====================================================================
# Strategy 4a: Reduced generators + Box bounds on x (no v variable)
# ====================================================================
def make_sga_3gen_boxed(box_x):
    """Reduced 3-generator setup with explicit box constraint on x.

    Tests whether boxing x alone helps when derivative variables are removed.
    """
    names = ["x", "y", "u"]
    sg = CommutativeSemigroup(names)
    sga = SemigroupAlgebra(sg)

    x, y, u = (sga[n] for n in names)

    relations = [u**2 - (x * y + 1)]

    prog = OptimizationProblem(sga, relations=relations)
    prog.set_objective(1 - x**2 + y)
    prog.add_constraints([
        u + x / 2 - x**2 + 2 * y,
        2 - y**2,
        box_x - x,   # x <= box_x
        box_x + x,   # -x <= box_x
    ])

    return prog


# ====================================================================
# Strategy 4b: Full 7-generator with Box bounds (best bounding from Phase 2)
# ====================================================================
def make_sga_7gen_boxed(box_x, v_min, v_max):
    """Full 7-generator setup with ADE + explicit box on x and bounds on v.

    Combines the full differential structure with the best bounding strategy
    found in Phase 2 (Box |x|<=1.5).
    """
    names = ["x", "y", "u", "v", "d_vx", "d_vy", "d_vu"]
    sg = CommutativeSemigroup(names)
    sga = SemigroupAlgebra(sg)

    gens = {g.ext_rep[0].name: g for g in sg.generators}
    x, y, u, v, d_vx, d_vy, d_vu = (sga[n] for n in names)

    # Derivatives
    sga.add_derivative({gens["x"]: 1, gens["y"]: 0, gens["u"]: 0, gens["v"]: y * v})
    sga.add_derivative({gens["x"]: 0, gens["y"]: 1, gens["u"]: 0, gens["v"]: x * v})
    sga.add_derivative({gens["x"]: d_vx, gens["y"]: d_vy, gens["u"]: d_vu, gens["v"]: 1})

    relations = [
        u**2 - (x * y + 1),
        v * y**2 * d_vx - (y - x * y * v * d_vy),
        v * x**2 * d_vy - (x - x * y * v * d_vx),
        2 * u * d_vu - (y * d_vx + x * d_vy),
    ]

    prog = OptimizationProblem(sga, relations=relations)
    prog.set_objective(1 - x**2 + y)
    prog.add_constraints([
        u + x / 2 - x**2 + 2 * y,
        2 - y**2,
        y - x * v + 1,
        box_x - x,   # x <= box_x
        box_x + x,   # -x <= box_x
        v - v_min,        # v >= v_min
        v_max - v,        # v <= v_max
    ])

    return prog


# ====================================================================
# Strategy 4c: 4-generator (x,y,u,v) with Box bounds — no ADE derivatives
# ====================================================================
def make_sga_4gen_boxed(box_x, v_min, v_max):
    """4-generator setup with box on x and explicit bounds on v.

    No derivative variables or ADE machinery — just algebraic lifts + bounding.
    This is the cleanest test of whether bounding alone closes the gap.
    """
    names = ["x", "y", "u", "v"]
    sg = CommutativeSemigroup(names)
    sga = SemigroupAlgebra(sg)

    x, y, u, v = (sga[n] for n in names)

    relations = [u**2 - (x * y + 1)]

    prog = OptimizationProblem(sga, relations=relations)
    prog.set_objective(1 - x**2 + y)
    prog.add_constraints([
        u + x / 2 - x**2 + 2 * y,
        2 - y**2,
        y - x * v + 1,
        box_x - x,   # x <= box_x
        box_x + x,   # -x <= box_x
        v - v_min,        # v >= v_min
        v_max - v,        # v <= v_max
    ])

    return prog


# ====================================================================
# Runner
# ====================================================================
def run_variant(label, make_prog_fn, order):
    """Run a single variant and return (value, elapsed_s, status)."""
    print(f"\n{'=' * 70}")
    print(f"{label} (order={order})")
    print(f"{'=' * 70}")

    t0 = time.time()
    try:
        prog = make_prog_fn()
        config = RelaxationConfig(
            reduction_method="border_basis",
            quotient_basis="groebner",
            monomial_pruning=True,
            sparsity_detection=True,
            verbose_reduction=False,
        )
        engine = RelaxationEngine(prog, order=order, solver="cvxopt",
                                  verbosity=0, config=config)
        result = engine.solve("sos")
        elapsed = time.time() - t0
        print(f"  SDP lower bound: {result.value:.8f}")
        print(f"  Status: {result.status} ({result.message})")
        print(f"  Time: {elapsed:.1f}s")
        return result.value, elapsed, result.status
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Error after {elapsed:.1f}s: {e}")
        return None, elapsed, "error"


def main():
    scipy_result = solve_with_scipy()

    if not scipy_result:
        print("ERROR: SciPy reference failed. Aborting.")
        return

    upper_bound = scipy_result.fun
    print("=" * 70)
    print("RESOLUTION PATH — Strategies 3 & 4")
    print(f"SciPy feasible upper bound: {upper_bound:.8f}")
    print(f"Baseline (full ADE, o=2):   -3.78550585  gap=+3.6356")

    results = []

    # --- Strategy 3: No derivative variables ---
    val, t, st = run_variant("S3: Pure algebraic (x,y,u), o=2", make_sga_3gen, 2)
    results.append(("Pure alg, o=2", val, t))

    val, t, st = run_variant("S3: Pure algebraic (x,y,u), o=3", make_sga_3gen, 3)
    results.append(("Pure alg, o=3", val, t))

    # --- Strategy 4a: Reduced + boxed x ---
    for bx in [2.0, 1.5]:
        label = f"S4a: Box |x|<={bx}, no deriv"
        val, t, st = run_variant(label, lambda b=bx: make_sga_3gen_boxed(b), 2)
        results.append((f"Box {bx}, no deriv, o=2", val, t))

    # --- Strategy 4b: Full ADE + best bounding from Phase 2 ---
    # Best Phase 2 result: Box |x|<=1.5, v in [0.120, 8.342], o=3 => -1.065
    val, t, st = run_variant("S4b: Full ADE + Box |x|<=1.5", 
                             lambda: make_sga_7gen_boxed(1.5, 0.12, 8.34), 3)
    results.append(("Full ADE+Box 1.5, o=3", val, t))

    # --- Strategy 4c: 4-gen + best bounding (no ADE) ---
    for bx in [1.5, 2.0]:
        label = f"S4c: 4gen Box |x|<={bx}"
        val, t, st = run_variant(label, 
                                 lambda b=bx: make_sga_4gen_boxed(b, 0.12, 8.34), 2)
        results.append((f"4gen+Box {bx}, o=2", val, t))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("COMBINED RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  SciPy feasible upper bound:   {upper_bound:.8f}")
    print(f"  Baseline (full ADE, o=2):     -3.78550585  gap=+3.6356")
    
    best_val = None
    best_label = ""
    for label, val, t in results:
        if val is not None:
            gap = upper_bound - val
            marker = " <-- BEST" if (best_val is None or val > best_val) else ""
            print(f"  {label:30s} {val:>12.8f}  gap={gap:+.4f}  ({t:.0f}s){marker}")
            if best_val is None or val > best_val:
                best_val = val
                best_label = label
        else:
            print(f"  {label:30s} FAILED")

    if best_val is not None:
        improvement = -3.78550585 - best_val
        gap_remaining = upper_bound - best_val
        print(f"\n  Best strategy: {best_label}")
        print(f"  Improvement over baseline: +{improvement:.4f}")
        print(f"  Remaining gap to SciPy:    {gap_remaining:+.4f}")


if __name__ == "__main__":
    main()
