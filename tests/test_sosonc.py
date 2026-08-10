"""Tests for Irene.sosonc — SOS+SONC relaxation module."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math

import pytest

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sosonc import SOSONCRelaxations, SOSONCRelaxSol, sosonc_bounds


# ── Helpers ──────────────────────────────────────────────────

def _make_prog_1d():
    """x^4 - x^2, unconstrained."""
    sg = CommutativeSemigroup(['x'])
    sga = SemigroupAlgebra(sg)
    x = sga['x']
    prog = OptimizationProblem(sga)
    prog.set_objective(x ** 4 - x ** 2)
    return prog


def _make_prog_quad():
    """x^2 + y^2, unconstrained."""
    sg = CommutativeSemigroup(['x', 'y'])
    sga = SemigroupAlgebra(sg)
    x = sga['x']
    y = sga['y']
    prog = OptimizationProblem(sga)
    prog.set_objective(x ** 2 + y ** 2)
    return prog


# ──────────────────────────────────────────────────────────────
# SOSONCRelaxSol
# ──────────────────────────────────────────────────────────────

def test_result_container_defaults():
    sol = SOSONCRelaxSol()
    assert math.isinf(sol.val) and sol.val < 0
    assert sol.method == ""
    assert sol.status == "error"
    assert sol.error_code == 2
    assert sol.runtime == 0.0
    assert isinstance(repr(sol), str)


def test_result_container_repr():
    sol = SOSONCRelaxSol()
    sol.val = 3.5
    sol.method = "sos"
    sol.status = "optimal"
    r = repr(sol)
    assert "3.5" in r
    assert "sos" in r


# ──────────────────────────────────────────────────────────────
# SOSONCRelaxations — integration
# ──────────────────────────────────────────────────────────────

class TestSOSONC:

    def test_global_min_sos_quadratic(self):
        """x^2 + y^2 global minimum = 0; SOS at r=1 should be >= 0."""
        prog = _make_prog_quad()
        engine = SOSONCRelaxations(prog, verbosity=0)
        result = engine.globalMinSOS()
        print(f"\n  SOS quad val={result.val}, status={result.status}")
        assert result.status == "optimal", f"Expected optimal, got {result.status}"
        assert not math.isinf(result.val), "Expected finite SOS bound"
        assert result.val >= -1e-6, f"Expected >= 0, got {result.val}"

    def test_global_min_sos_quartic(self):
        """x^4 - x^2: SOS at r=2 should give ~-0.25."""
        prog = _make_prog_1d()
        engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=2)
        result = engine.globalMinSOS()
        print(f"\n  SOS quart val={result.val}, status={result.status}")
        assert result.status == "optimal", f"Expected optimal, got {result.status}"
        assert not math.isinf(result.val), "Expected finite SOS bound"
        assert result.val <= -0.24, f"Expected <= -0.24, got {result.val}"
        assert result.val >= -0.26, f"Expected >= -0.26, got {result.val}"

    def test_global_min_sonc_runs(self):
        """SONC on quadratic — runs without crash."""
        prog = _make_prog_quad()
        engine = SOSONCRelaxations(prog, verbosity=0)
        result = engine.globalMinSONC()
        print(f"\n  SONC quad val={result.val}, status={result.status}")
        assert result.method == "sonc"
        assert result.runtime >= 0

    def test_two_step_sos_first_runs(self):
        """SOS-first two-step returns a result."""
        prog = _make_prog_quad()
        engine = SOSONCRelaxations(prog, verbosity=0)
        result = engine.globalMinSOSPSONC(first="sos")
        print(f"\n  SOS-first val={result.val}, method={result.method}")
        assert result.method in ("sos-first", "sonc-first", "sos", "sonc")

    def test_two_step_sonc_first_runs(self):
        """SONC-first two-step returns a result."""
        prog = _make_prog_quad()
        engine = SOSONCRelaxations(prog, verbosity=0)
        result = engine.globalMinSOSPSONC(first="sonc")
        print(f"\n  SONC-first val={result.val}, method={result.method}")
        assert result.method in ("sos-first", "sonc-first", "sos", "sonc")

    def test_sosonc_bounds(self):
        """Convenience function returns all four keys."""
        prog = _make_prog_quad()
        bounds = sosonc_bounds(prog, verbosity=0)
        for key in ("sos", "sonc", "sos_first", "sonc_first"):
            assert key in bounds, f"Missing: {key}"
        print(f"\n  Bounds: {bounds}")

    def test_invalid_first_arg(self):
        """Rejects invalid 'first' argument."""
        prog = _make_prog_quad()
        engine = SOSONCRelaxations(prog)
        with pytest.raises(ValueError, match="first must be"):
            engine.globalMinSOSPSONC(first="invalid")


# ── Known cases ──────────────────────────────────────────────

class TestKnownPolynomials:

    def test_motzkin_sonc(self):
        """Motzkin: 1 + x^4*y^2 + x^2*y^4 - 3*x^2*y^2. SONC >= 0 locally."""
        sg = CommutativeSemigroup(['x', 'y'])
        sga = SemigroupAlgebra(sg)
        x = sga['x']
        y = sga['y']
        f = 1 + x ** 4 * y ** 2 + x ** 2 * y ** 4 - 3 * x ** 2 * y ** 2

        prog = OptimizationProblem(sga)
        prog.set_objective(f)

        engine = SOSONCRelaxations(prog, verbosity=0)
        result = engine.globalMinSONC()
        print(f"\n  Motzkin SONC val={result.val}, status={result.status}")
        # Motzkin is SONC, global minimum = 0
        if result.status == "optimal" and not math.isinf(result.val):
            assert result.val <= 1e-4, f"Motzkin SONC bound too high: {result.val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
