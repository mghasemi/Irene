"""Resolution path experiments for sqrt/exp differential lift gap.

Previous session (2026-08-13) found a structural gap:
  SDP lower bound: -3.7855
  SciPy upper bound: -0.1499
  Gap: +3.6356

Three resolution strategies identified:
1. Higher relaxation order (k >= 4) so ADE relations reduce low-order monomials
2. Explicit bounds on v based on domain of x*y
3. Remove derivative variables if not needed for objective/constraints

This script tests all three strategies and a combined variant.
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
# SciPy reference (same as original script)
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


def make_sga_7gen():
    """Full 7-generator setup with ADE derivatives."""
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
    ])

    return prog


def make_sga_3gen():
    """Reduced 3-generator setup: only x, y, u (no derivative vars)."""
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


def make_sga_4gen_bounded():
    """4-generator setup: x, y, u, v with explicit bounds on v."""
    names = ["x", "y", "u", "v"]
    sg = CommutativeSemigroup(names)
    sga = SemigroupAlgebra(sg)

    x, y, u, v = (sga[n] for n in names)

    # Only algebraic lift relation; no ADE derivative relations
    relations = [u**2 - (x * y + 1)]

    prog = OptimizationProblem(sga, relations=relations)
    prog.set_objective(1 - x**2 + y)
    prog.add_constraints([
        u + x / 2 - x**2 + 2 * y,
        2 - y**2,
        y - x * v + 1,
        # Explicit bounds on v: since |y| <= sqrt(2), and we box x in [-B, B],
        # |x*y| <= B*sqrt(2).  Use B=3 => |xy| <= 4.24 => v in [exp(-4.24), exp(4.24)]
        # Tighter: the constraint y - x*v + 1 >= 0 with v = exp(x*y) means
        # we need v to be positive and bounded.
        v,                    # v >= 0
        75 - v,              # v <= 75 (exp(4.32) ~ 75; generous upper bound)
    ])

    return prog


def run_variant(label, make_prog_fn, order, timeout=600):
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


# ====================================================================
# Main experiment runner
# ====================================================================
def main():
    scipy_result = solve_with_scipy()

    if not scipy_result:
        print("ERROR: SciPy reference failed. Aborting.")
        return

    print("=" * 70)
    print("RESOLUTION PATH EXPERIMENTS")
    print(f"SciPy feasible upper bound: {scipy_result.fun:.8f}")
    print(f"SciPy point: x={scipy_result.x[0]:.6f}, y={scipy_result.x[1]:.6f}")

    results = []

    # --- Strategy 1: Higher order with full ADE (7 generators) ---
    val, t, st = run_variant("S1: Full ADE, o=3", make_sga_7gen, 3)
    results.append(("Full ADE, o=3", val, t))

    val, t, st = run_variant("S1: Full ADE, o=4", make_sga_7gen, 4, timeout=900)
    results.append(("Full ADE, o=4", val, t))

    # --- Strategy 3: Remove derivative variables (3 generators) ---
    val, t, st = run_variant("S3: No deriv vars, o=2", make_sga_3gen, 2)
    results.append(("No deriv, o=2", val, t))

    val, t, st = run_variant("S3: No deriv vars, o=3", make_sga_3gen, 3)
    results.append(("No deriv, o=3", val, t))

    val, t, st = run_variant("S3: No deriv vars, o=4", make_sga_3gen, 4)
    results.append(("No deriv, o=4", val, t))

    # --- Strategy 2: Explicit bounds on v (4 generators) ---
    val, t, st = run_variant("S2: Bounded v, o=2", make_sga_4gen_bounded, 2)
    results.append(("Bounded v, o=2", val, t))

    val, t, st = run_variant("S2: Bounded v, o=3", make_sga_4gen_bounded, 3)
    results.append(("Bounded v, o=3", val, t))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  SciPy feasible upper bound:   {scipy_result.fun:.8f}")
    for label, val, t in results:
        if val is not None:
            gap = scipy_result.fun - val
            print(f"  {label:25s} {val:>12.8f}  gap={gap:+.4f}  ({t:.0f}s)")
        else:
            print(f"  {label:25s} FAILED")


if __name__ == "__main__":
    main()
