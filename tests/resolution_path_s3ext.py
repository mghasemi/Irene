"""Resolution path — Strategy 3 extended (pure algebraic at higher orders) + tighter boxing.

Key finding: Pure algebraic (x,y,u), o=2 => -0.669 (gap +0.52).
The ADE machinery actually hurts the bound.

This script tests:
- S3-ext: Pure algebraic at o=4, o=5 to see if hierarchy converges without ADE noise
- S5: Tighter boxing on x,y with pure algebraic (the feasible region is small)
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
# Pure algebraic (x,y,u) — the winning formulation
# ====================================================================
def make_sga_3gen():
    """Reduced 3-generator setup: only x, y, u."""
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
# Pure algebraic with tighter box on x (SciPy point has x ~ 0.96)
# ====================================================================
def make_sga_3gen_tight_box(box_x):
    """Pure algebraic with tight box around the feasible region."""
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
    print("RESOLUTION PATH — Strategy 3 Extended + Tighter Boxing")
    print(f"SciPy feasible upper bound: {upper_bound:.8f}")
    print(f"Best so far (pure alg, o=2): -0.66873270  gap=+0.5188")

    results = []

    # --- S3 extended: Higher orders with pure algebraic ---
    for order in [4, 5]:
        val, t, st = run_variant(f"S3-ext: Pure alg, o={order}", make_sga_3gen, order)
        results.append((f"Pure alg, o={order}", val, t))

    # --- S5: Tighter boxing around SciPy point (x ~ 0.96) ---
    for bx in [1.2, 1.0, 0.8]:
        label = f"S5: Box |x|<={bx}, pure alg"
        val, t, st = run_variant(label, lambda b=bx: make_sga_3gen_tight_box(b), 2)
        results.append((f"Box {bx}, o=2", val, t))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EXTENDED RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  SciPy feasible upper bound:   {upper_bound:.8f}")
    print(f"  Best so far (pure alg, o=2): -0.66873270  gap=+0.5188")

    best_val = None
    best_label = ""
    for label, val, t in results:
        if val is not None:
            gap = upper_bound - val
            marker = " <-- NEW BEST" if (best_val is None or val > best_val) else ""
            print(f"  {label:30s} {val:>12.8f}  gap={gap:+.4f}  ({t:.0f}s){marker}")
            if best_val is None or val > best_val:
                best_val = val
                best_label = label
        else:
            print(f"  {label:30s} FAILED")

    if best_val is not None:
        improvement_over_baseline = -3.78550585 - best_val
        gap_remaining = upper_bound - best_val
        pct_closed = (improvement_over_baseline / 3.6356) * 100
        print(f"\n  Best strategy: {best_label}")
        print(f"  Improvement over baseline (-3.7855): +{improvement_over_baseline:.4f} ({pct_closed:.1f}% of gap closed)")
        print(f"  Remaining gap to SciPy:             {gap_remaining:+.4f}")


if __name__ == "__main__":
    main()
