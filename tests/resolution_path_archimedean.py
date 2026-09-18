"""Resolution path — Archimedean constraints for each variable.

From session math/20260813_224820_0104d8 (Resolution Path experiments).
Problem: min 1 - x^2 + y s.t. sqrt(xy+1) + x/2 - x^2 + 2y >= 0, |y| <= sqrt(2), y - x*exp(xy) + 1 >= 0

Best previous result: pure algebraic (x,y,u) with box on x only => gap +0.0054 at |x|<=0.92.
This script adds Archimedean constraints for EACH variable to test whether bounding all generators
individually further tightens the SDP relaxation.

Archimedean condition: For each generator g, add B_g - g and B_g + g as constraints (i.e., |g| <= B_g).
"""

import time
from math import sqrt

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
        exponential = np.exp(np.clip(xv * yv, -700.0, 700.0))
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
# Pure algebraic (x,y,u) with per-variable Archimedean boxes
# ====================================================================
def make_sga_3gen_archimedean(box_x=None, box_y=None, box_u=None):
    """Reduced 3-generator setup with optional Archimedean bounds on each variable.

    box_x: bound for |x| (None = no bound)
    box_y: bound for |y| (None = no bound; note 2-y^2 already implies |y|<=sqrt(2))
    box_u: bound for |u| (None = no bound)
    """
    names = ["x", "y", "u"]
    sg = CommutativeSemigroup(names)
    sga = SemigroupAlgebra(sg)

    x, y, u = (sga[n] for n in names)

    relations = [u**2 - (x * y + 1)]

    prog = OptimizationProblem(sga, relations=relations)
    prog.set_objective(1 - x**2 + y)
    prog.add_constraints([
        u + x / 2 - x**2 + 2 * y,   # sqrt(xy+1) + x/2 - x^2 + 2y >= 0
        2 - y**2,                    # |y| <= sqrt(2) (already Archimedean for y)
    ])

    if box_x is not None:
        prog.add_constraints([box_x - x, box_x + x])   # |x| <= box_x
    if box_y is not None:
        prog.add_constraints([box_y - y, box_y + y])   # |y| <= box_y (redundant with 2-y^2)
    if box_u is not None:
        prog.add_constraints([box_u - u, box_u + u])   # |u| <= box_u

    return prog


# ====================================================================
# Runner
# ====================================================================
def run_variant(label, make_prog_fn, order):
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
# Main
# ====================================================================
def main():
    scipy_result = solve_with_scipy()
    if not scipy_result:
        print("ERROR: SciPy reference failed. Aborting.")
        return

    ub = scipy_result.fun
    x_opt, y_opt = scipy_result.x
    u_opt = sqrt(x_opt * y_opt + 1)

    print("=" * 70)
    print("RESOLUTION PATH — Archimedean constraints per variable")
    print(f"SciPy upper bound: {ub:.8f}")
    print(f"SciPy optimum: x={x_opt:.6f}, y={y_opt:.6f}, u=sqrt(xy+1)={u_opt:.6f}")
    print(f"Baseline (full ADE, o=2): -3.78550585  gap=+3.6356")
    print()

    results = []

    # --- Baseline: no Archimedean constraints on x or u ---
    label = "No box (baseline)"
    val, t, st = run_variant(label, lambda: make_sga_3gen_archimedean(), 2)
    results.append((label, val, t))

    # --- Box only x (previous best strategy) ---
    for bx in [1.0, 0.92]:
        label = f"Box |x|<={bx}"
        val, t, st = run_variant(label, lambda b=bx: make_sga_3gen_archimedean(box_x=b), 2)
        results.append((label, val, t))

    # --- Box only u (new test) ---
    for bu in [3.0, 2.5, 2.0]:
        label = f"Box |u|<={bu}"
        val, t, st = run_variant(label, lambda b=bu: make_sga_3gen_archimedean(box_u=b), 2)
        results.append((label, val, t))

    # --- Box both x and u (combined Archimedean) ---
    for bx, bu in [(1.0, 3.0), (1.0, 2.5), (0.92, 2.5), (0.92, 2.0)]:
        label = f"Box |x|<={bx}, |u|<={bu}"
        val, t, st = run_variant(label,
                                 lambda b=bx, c=bu: make_sga_3gen_archimedean(box_x=b, box_u=c), 2)
        results.append((label, val, t))

    # --- Box all three variables (full Archimedean per variable) ---
    for bx, by, bu in [(1.0, 1.4, 2.5), (0.92, 1.3, 2.0), (0.96, 1.2, 1.8)]:
        label = f"Box |x|<={bx}, |y|<={by}, |u|<={bu}"
        val, t, st = run_variant(label,
                                 lambda a=bx, b=by, c=bu: make_sga_3gen_archimedean(box_x=a, box_y=b, box_u=c), 2)
        results.append((label, val, t))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("ARCHIMEDEAN PER-VARIABLE SUMMARY")
    print("=" * 70)
    print(f"\n  SciPy upper bound:   {ub:.8f}")
    print(f"  Baseline (no box):   -3.78550585  gap=+3.6356")

    best_val = None
    best_label = ""
    for label, val, t in results:
        if val is not None:
            gap = ub - val
            marker = " <-- BEST" if (best_val is None or val > best_val) else ""
            valid = "[VALID]" if gap >= 0 else "[EXCEEDS UB]"
            print(f"  {label:35s} {val:>12.8f}  gap={gap:+.6f}  ({t:.0f}s){valid}{marker}")
            if best_val is None or val > best_val:
                best_val = val
                best_label = label
        else:
            print(f"  {label:35s} FAILED")

    if best_val is not None:
        improvement = -3.78550585 - best_val
        gap_remaining = ub - best_val
        pct = (improvement / 3.6356) * 100
        print(f"\n  Best: {best_label}")
        print(f"  Improvement over baseline: +{improvement:.4f} ({pct:.1f}% closure)")
        print(f"  Remaining gap to SciPy:    {gap_remaining:+.6f}")


if __name__ == "__main__":
    main()
