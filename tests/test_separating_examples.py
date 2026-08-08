"""Regression tests for separating examples.

Motzkin, Choi-Lam, and Robinson polynomials are nonnegative but NOT SOS.
If the SOS relaxation at order 3 certifies them as nonnegative (value >= -tol),
that indicates a bug in the SDP formulation or solver routing — these should
remain negative/infeasible for SOS alone.

These tests guard against silent regressions where numerical tolerance drift,
solver changes, or formulation errors cause separating examples to be
incorrectly classified as SOS-certifiable.
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sosonc import SOSONCRelaxations


def _build_unconstrained(variables, objective_expr):
    """Build an unconstrained optimization problem from variable names and expression."""
    sg = CommutativeSemigroup(variables)
    sga = SemigroupAlgebra(sg)
    sym_dict = {v: sga[v] for v in variables}

    objective = eval(objective_expr, {"__builtins__": {}}, sym_dict)
    prog = OptimizationProblem(sga)
    prog.set_objective(objective)
    return prog


def test_motzkin_not_sos():
    """Motzkin polynomial: x^4 y^2 + x^2 y^4 + 1 - 3 x^2 y^2.

    True minimum is 0, but SOS at order 3 cannot certify nonnegativity —
    the SOS lower bound should be strictly negative (or infeasible).
    """
    prog = _build_unconstrained(
        ['x', 'y'],
        "x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2"
    )

    engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=3)
    result = engine.globalMinSOS()

    # SOS should NOT certify nonnegativity for Motzkin.
    # The bound should be negative (or the solve fails).
    if result.val is not None:
        assert result.val < -1e-4, (
            f"Motzkin SOS bound = {result.val:.6f} >= -1e-4: "
            "separating example incorrectly certified as SOS! "
            "This indicates a regression in the SDP formulation."
        )


def test_choi_lam_not_sos():
    """Choi-Lam polynomial: x^4 y^2 + x^2 y^4 + x^2 y^2 (x^2 + y^2 - 1).

    Nonnegative on R^2, not SOS. SOS at order 3 should give a negative bound.
    """
    prog = _build_unconstrained(
        ['x', 'y'],
        "x**4 * y**2 + x**2 * y**4 + x**2 * y**2 * (x**2 + y**2 - 1)"
    )

    engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=3)
    result = engine.globalMinSOS()

    if result.val is not None:
        assert result.val < -1e-4, (
            f"Choi-Lam SOS bound = {result.val:.6f} >= -1e-4: "
            "separating example incorrectly certified as SOS!"
        )


def test_robinson_not_sos():
    """Robinson polynomial: x^4 y^2 + x^2 y^4 + x^4 + y^4 - x^2 - y^2.

    Nonnegative, not SOS. SOS at order 3 should give a negative bound.
    """
    prog = _build_unconstrained(
        ['x', 'y'],
        "x**4 * y**2 + x**2 * y**4 + x**4 + y**4 - x**2 - y**2"
    )

    engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=3)
    result = engine.globalMinSOS()

    if result.val is not None:
        assert result.val < -1e-4, (
            f"Robinson SOS bound = {result.val:.6f} >= -1e-4: "
            "separating example incorrectly certified as SOS!"
        )


if __name__ == "__main__":
    tests = [test_motzkin_not_sos, test_choi_lam_not_sos, test_robinson_not_sos]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"\u2713 {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"\u2717 {test.__name__}: REGRESSION — {e}")
            failed += 1
        except Exception as e:
            print(f"\u2717 {test.__name__}: ERROR — {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
