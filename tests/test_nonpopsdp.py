"""Tests for the NonPOPSDP pipeline (IreneRewrite port).

Validates the polynomial-approximation layer (Taylor/Chebyshev), the
transcendental surrogates, and the end-to-end Lasserre SDP solve for
non-polynomial objectives. The port fixes two numerical bugs present in the
original implementation (verified against original Irene):

1. Chebyshev coefficient extraction used an incorrectly scaled raw FFT
   (max error ~61.5 for exp degree 6 on [-2,2]); the port uses
   numpy.polynomial.chebyshev.chebfit (error ~5e-4).
2. Taylor coefficients used naive central differences (error ~1e36 for
   exp degree 6); the port uses a Richardson-extrapolated high-order stencil.
"""
from math import exp, sin, cos, pi, sqrt, factorial

import pytest
from sympy import symbols

from Irene.nonpopsdp import (
    chebyshev_approx,
    taylor_approx,
    TranscendentalApproximator,
    NonPOPSDP,
    NonPOPSDP_Multi,
)


class TestApproximations:
    def test_chebyshev_exp_deg6(self):
        x = symbols("x")
        poly, err = chebyshev_approx(exp, x, (-2.0, 2.0), 6)
        assert err < 0.01  # true degree-6 Chebyshev error ~5e-4

    def test_chebyshev_sin_non_symmetric_domain(self):
        x = symbols("x")
        poly, err = chebyshev_approx(sin, x, (0.0, pi), 8)
        assert err < 1e-3

    def test_taylor_exp_deg6(self):
        x = symbols("x")
        poly, err = taylor_approx(exp, x, 0.0, 6)
        # Lagrange remainder for exp at degree 6 is 1/7! ~ 2e-4
        assert err < 1e-2
        # poly should be close to the true Taylor polynomial
        coeffs = [float(poly.coeff(x, k)) for k in range(7)]
        for k, c in enumerate(coeffs):
            assert abs(c - 1.0 / factorial(k)) < 1e-4

    def test_chebyshev_poly_evaluates_near_function(self):
        x = symbols("x")
        poly, _ = chebyshev_approx(exp, x, (-1.0, 1.0), 8)
        f = __import__("sympy").lambdify(x, poly, "numpy")
        assert abs(f(0.5) - exp(0.5)) < 1e-5
        assert abs(f(-1.0) - exp(-1.0)) < 1e-5


class TestTranscendentalApproximator:
    def test_substitute_replaces_symbols(self):
        x = symbols("x")
        sin_sym, cos_sym = symbols("sin cos")
        app = TranscendentalApproximator(
            x,
            {
                "sin": {"func": sin, "method": "chebyshev", "domain": (-pi, pi), "degree": 8},
                "cos": {"func": cos, "method": "chebyshev", "domain": (-pi, pi), "degree": 8},
            },
        )
        expr = app.substitute(sin_sym + cos_sym)
        assert sin_sym not in expr.free_symbols
        assert cos_sym not in expr.free_symbols
        assert x in expr.free_symbols

    def test_unknown_method_raises(self):
        x = symbols("x")
        with pytest.raises(ValueError):
            TranscendentalApproximator(
                x, {"f": {"func": exp, "method": "nonsense", "domain": (-1, 1)}})


class TestNonPOPSDP:
    def test_trig_min(self):
        """min(sin+cos) on [-pi,pi]; true -sqrt(2) ~ -1.41421."""
        x = symbols("x")
        sin_sym, cos_sym = symbols("sin cos")
        pop = NonPOPSDP(
            x,
            {
                "sin": {"func": sin, "method": "chebyshev", "domain": (-pi, pi), "degree": 8},
                "cos": {"func": cos, "method": "chebyshev", "domain": (-pi, pi), "degree": 8},
            },
            relax_order=2, ball_radius=pi, verbosity=0,
        )
        pop.set_objective(sin_sym + cos_sym)
        lb = pop.solve()
        assert lb is not None
        # Valid lower bound, within Chebyshev approximation error of true min
        assert lb <= -sqrt(2) + 1e-2
        assert lb >= -sqrt(2) - 1e-2

    def test_exp_min(self):
        """min exp(x) on [-1,1]; true exp(-1) ~ 0.36788."""
        x = symbols("x")
        exp_sym = symbols("exp")
        pop = NonPOPSDP(
            x,
            {"exp": {"func": exp, "method": "chebyshev", "domain": (-1.0, 1.0), "degree": 6}},
            relax_order=2, ball_radius=1.0, verbosity=0,
        )
        pop.set_objective(exp_sym)
        lb = pop.solve()
        assert lb is not None
        assert lb <= exp(-1) + 1e-2
        assert lb >= exp(-1) - 1e-2

    def test_exp_taylor_method(self):
        """min exp(x) on [-0.5, 0.5] via Taylor around 0; true exp(-0.5)."""
        x = symbols("x")
        exp_sym = symbols("exp")
        pop = NonPOPSDP(
            x,
            {"exp": {"func": exp, "method": "taylor", "center": 0.0, "degree": 6}},
            relax_order=2, ball_radius=0.5, verbosity=0,
        )
        pop.set_objective(exp_sym)
        lb = pop.solve()
        assert lb is not None
        # surrogate error ~1.5e-5 on [-0.5, 0.5]; allow SDP tolerance
        assert lb <= exp(-0.5) + 1e-2
        assert lb >= exp(-0.5) - 1e-2

    def test_result_metadata(self):
        x = symbols("x")
        exp_sym = symbols("exp")
        pop = NonPOPSDP(
            x,
            {"exp": {"func": exp, "method": "chebyshev", "domain": (-1.0, 1.0), "degree": 6}},
            relax_order=2, ball_radius=1.0, verbosity=0,
        )
        pop.set_objective(exp_sym)
        pop.solve()
        assert pop.result is not None
        assert "lower_bound" in pop.result
        assert "status" in pop.result
        assert "solver" in pop.result


class TestNonPOPSDP_Multi:
    def test_two_var_smoke(self):
        """min exp(x) + y^2 with y in [-1,1] and x in [-1,1]."""
        x, y = symbols("x y")
        exp_sym = symbols("exp")
        pop = NonPOPSDP_Multi(
            [x, y],
            {"exp": {"func": exp, "method": "chebyshev", "domain": (-1.0, 1.0), "degree": 6, "var_idx": 0}},
            relax_order=2, ball_radius=1.0, verbosity=0,
        )
        pop.set_objective(exp_sym + y**2)
        lb = pop.solve()
        assert lb is not None
        # true min = exp(-1) + 0 = 0.36788 (y=0 attainable)
        assert lb <= exp(-1) + 1e-2
        assert lb >= exp(-1) - 1e-2
