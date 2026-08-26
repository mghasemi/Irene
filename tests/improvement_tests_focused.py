"""Focused tests: reproduce prior best, test order 3, and refine boxing strategy."""

import time
from math import sqrt
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
        exponential = np.exp(np.clip(xv * yv, -700.0, 700.0))
        return np.array([radicand, root + xv / 2.0 - xv**2 + 2.0 * yv,
                        2.0 - yv**2, yv - xv * exponential + 1.0])

    constraints = {"type": "ineq", "fun": constraint_values}
    starts = [(xv, yv) for xv in np.linspace(-3.0, 3.0, 7)
              for yv in np.linspace(-sqrt(2.), sqrt(2.), 5)
              if xv * yv + 1.0 >= 0.0]
    results = []
    for start in starts:
        r = minimize(objective, np.asarray(start, dtype=float), method="SLSQP",
                     bounds=[(-10., 10.), (-sqrt(2.), sqrt(2.))],
                     constraints=constraints, options={"ftol": 1e-10, "maxiter": 1000})
        if r.success and np.min(constraint_values(r.x)) >= -1e-7:
            results.append(r)
    return min(results, key=lambda item: item.fun) if results else None


def make_prog(box_x=None, box_y=None, box_u=None):
    names = ["x", "y", "u"]
    sg = CommutativeSemigroup(names)
    sga = SemigroupAlgebra(sg)
    x, y, u = (sga[n] for n in names)
    relations = [u**2 - (x * y + 1)]
    prog = OptimizationProblem(sga, relations=relations)
    prog.set_objective(1 - x**2 + y)
    prog.add_constraints([u + x / 2 - x**2 + 2 * y, 2 - y**2])
    if box_x is not None:
        prog.add_constraints([box_x - x, box_x + x])
    if box_y is not None:
        prog.add_constraints([box_y - y, box_y + y])
    if box_u is not None:
        prog.add_constraints([box_u - u, box_u + u])
    return prog


def run(label, make_fn, order=2, solver="sos", reduction="border_basis"):
    t0 = time.time()
    try:
        engine = RelaxationEngine(make_fn(), order=order, solver="cvxopt", verbosity=0,
                                  config=RelaxationConfig(reduction_method=reduction,
                                                          quotient_basis="groebner",
                                                          monomial_pruning=True,
                                                          sparsity_detection=True))
        result = engine.solve(solver)
        return result.value, time.time() - t0, result.status
    except Exception as e:
        return None, time.time() - t0, f"error: {e}"


def main():
    scipy_res = solve_with_scipy()
    if not scipy_res:
        print("SciPy failed"); return
    ub = scipy_res.fun
    xo, yo = scipy_res.x
    uo = sqrt(xo * yo + 1)

    print(f"SciPy UB: {ub:.8f}  at x={xo:.6f}, y={yo:.6f}, u={uo:.6f}")
    print("=" * 70)

    tests = [
        # Reproduce prior best
        ("box 0.92, sos", lambda: make_prog(box_x=0.92), 2, "sos"),
        ("box 0.94, sos", lambda: make_prog(box_x=0.94), 2, "sos"),
        ("box 0.96, sos", lambda: make_prog(box_x=0.96), 2, "sos"),

        # Order 3 tests (key improvement candidate)
        ("o=3, no box, sos", lambda: make_prog(), 3, "sos"),
        ("o=3, box 1.0, sos", lambda: make_prog(box_x=1.0), 3, "sos"),
        ("o=3, box 0.96, sos", lambda: make_prog(box_x=0.96), 3, "sos"),

        # SOSONC with various boxes (prior best may have been SOSONC)
        ("box 0.92, sosonc_sos", lambda: make_prog(box_x=0.92), 2, "sosonc_sos_first"),
        ("box 0.94, sosonc_sos", lambda: make_prog(box_x=0.94), 2, "sosonc_sos_first"),

        # Multi-box with SOSONC
        ("multi-box tight, sosonc", lambda: make_prog(0.97, 0.35, 1.2), 2, "sosonc_sos_first"),

        # Order 3 + multi-box (most aggressive)
        ("o=3, multi-box, sos", lambda: make_prog(0.98, 0.4, 1.3), 3, "sos"),
    ]

    results = []
    for label, fn, order, solver in tests:
        val, t, st = run(label, fn, order, solver)
        gap = ub - val if val is not None else float('nan')
        valid = "[VALID]" if val is not None and gap >= 0 else "[INVALID/ERR]"
        results.append((label, val, t, gap, valid))
        print(f"  {label:30s} {val:>12.8f}  gap={gap:+.6f}  ({t:.0f}s) {valid}")

    # Find best VALID result
    valid_results = [(l, v, t, g) for l, v, t, g, _ in results if v is not None and g >= 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x[1])
        print(f"\nBest VALID: {best[0]} => {best[1]:.8f} (gap +{best[3]:.6f})")


if __name__ == "__main__":
    main()
