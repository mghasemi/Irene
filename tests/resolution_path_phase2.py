"""Resolution path — Phase 2: Tighter bounding strategies.

Phase 1 tests higher order and generator reduction.
Phase 2 focuses on geometric tightening of the exponential lift constraint.

Key insight from previous session:
  The constraint y - x*v + 1 >= 0 is binding at SciPy optimum (x=0.964, y=-0.220).
  With v = exp(x*y), we have v* ≈ 0.809.
  But SDP can set v arbitrarily small, decoupling it from exp(x*y).

Strategies tested here:
A. Box constraint on x (|x| <= B) => tighter bounds on v via |xy| <= B*sqrt(2)
B. Taylor expansion of exp(xy) as polynomial constraints up to order k
C. Combined: box + Taylor + higher order
"""

import time
from math import exp, factorial, sqrt

import numpy as np
from scipy.optimize import minimize

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxation_api import RelaxationEngine
from Irene.relaxations import RelaxationConfig


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


def make_sga_boxed(B=2.0):
    """Box x in [-B, B], add tight v bounds derived from |xy| <= B*sqrt(2)."""
    names = ["x", "y", "u", "v"]
    sg = CommutativeSemigroup(names)
    sga = SemigroupAlgebra(sg)

    x, y, u, v = (sga[n] for n in names)

    # Max |xy| = B * sqrt(2)
    max_xy = B * sqrt(2)
    v_min = exp(-max_xy)  # lower bound on exponential
    v_max = exp(max_xy)   # upper bound

    relations = [u**2 - (x * y + 1)]

    prog = OptimizationProblem(sga, relations=relations)
    prog.set_objective(1 - x**2 + y)
    prog.add_constraints([
        u + x / 2 - x**2 + 2 * y,
        2 - y**2,
        y - x * v + 1,
        B**2 - x**2,          # |x| <= B
        v,                    # v >= 0
        v_max - v,            # v <= exp(B*sqrt(2))
    ])

    return prog, B, max_xy, v_min, v_max


def make_sga_taylor(order_taylor=3):
    """Approximate v ≈ exp(xy) via Taylor expansion constraints.

    exp(z) = 1 + z + z^2/2! + ... + z^n/n!
    So v - (1 + xy + (xy)^2/2 + ...) should be small.

    We add: |v - T_k(xy)| <= epsilon as polynomial constraints.
    """
    names = ["x", "y", "u", "v"]
    sg = CommutativeSemigroup(names)
    sga = SemigroupAlgebra(sg)

    x, y, u, v = (sga[n] for n in names)

    # Taylor polynomial of exp(xy) up to given order
    z = x * y  # shorthand
    taylor = sga.one
    term = sga.one
    for k in range(1, order_taylor + 1):
        term *= z
        taylor += term / factorial(k)

    relations = [u**2 - (x * y + 1)]

    prog = OptimizationProblem(sga, relations=relations)
    prog.set_objective(1 - x**2 + y)
    prog.add_constraints([
        u + x / 2 - x**2 + 2 * y,
        2 - y**2,
        y - x * v + 1,
        # Taylor approximation: constrain v to be close to Taylor polynomial
        # Irene expects >= 0 constraints, so write as 0.5 - (v - T_k)^2 >= 0
        0.5 - (v - taylor)**2,
    ])

    return prog


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
        import traceback
        traceback.print_exc()
        return None, elapsed, "error"


def main():
    scipy_result = solve_with_scipy()

    if not scipy_result:
        print("ERROR: SciPy reference failed. Aborting.")
        return

    print("=" * 70)
    print("RESOLUTION PATH — PHASE 2 (Geometric Tightening)")
    print(f"SciPy feasible upper bound: {scipy_result.fun:.8f}")

    results = []

    # --- Strategy A: Box constraints with decreasing B ---
    for B in [3.0, 2.0, 1.5]:
        prog_fn, B_val, max_xy, v_min, v_max = make_sga_boxed(B)
        label = f"A: Box |x|<={B}, v∈[{v_min:.3f},{v_max:.3f}]"

        for order in [2, 3]:
            val, t, st = run_variant(label, lambda p=prog_fn: p, order)
            results.append((label + f", o={order}", val, t))

    # --- Strategy B: Taylor approximation constraints ---
    for taylor_ord in [2, 3]:
        prog_fn = make_sga_taylor(taylor_ord)
        label = f"B: Taylor exp(xy) to order {taylor_ord}"

        for order in [2, 3]:
            val, t, st = run_variant(label, lambda p=prog_fn: p, order)
            results.append((label + f", o={order}", val, t))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PHASE 2 COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  SciPy feasible upper bound:   {scipy_result.fun:.8f}")
    for label, val, t in results:
        if val is not None:
            gap = scipy_result.fun - val
            print(f"  {label:45s} {val:>12.8f}  gap={gap:+.4f}  ({t:.0f}s)")
        else:
            print(f"  {label:45s} FAILED")


if __name__ == "__main__":
    main()
