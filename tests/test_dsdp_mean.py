"""
Tests for DSDP Mean Polynomial Relaxations.

Validates that M_{q,p} certificates are PSD when q > p (Prop. 2.1),
and that the solver returns correct lower bounds for known forms.
"""

import pytest
from sympy import symbols

from Irene.dsdp import DSDPMeanRelaxation, DSDPRelaxations


class TestMeanCertificateParams:
    """Verify parameter validation aligns with theory (q > p required)."""

    def test_rejects_non_psd_params(self):
        """q <= p must be rejected as non-PSD."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=1, p=2, weights=[1.0, 1.0], verbosity=0)
        certs = dsdp._build_mean_certificate_moments()
        # Should return empty because q=1 <= p=2 violates PSD condition
        assert len(certs) == 0

    def test_accepts_psd_params(self):
        """q > p must produce non-empty certificate constraints."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=2, p=1, weights=[1.0, 1.0], verbosity=0)
        dsdp.SetObjective(x**2 + y**2)
        certs = dsdp._build_mean_certificate_moments()
        # Should produce constraints for valid q > p
        assert len(certs) > 0


class TestChoiLamForm:
    """Choi-Lam form Q = x^4 + y^4 + z^4 + w^4 - 4xyzw is PSD (min = 0)."""

    def test_mean_relaxation_returns_zero(self):
        """M_{1,0} certificate should identify Q as PSD (lower bound ~ 0)."""
        x, y, z, w = symbols('x y z w')
        dsdp = DSDPMeanRelaxation(
            gens=[x, y, z, w],
            weights=[1.0, 1.0, 1.0, 1.0],
            q=1,
            p=0,
            verbosity=0
        )
        dsdp.SetObjective(x**4 + y**4 + z**4 + w**4 - 4*x*y*z*w)
        lb = dsdp.solve(order=2)
        # Q is PSD with minimum 0
        assert abs(float(lb)) < 1e-2


class TestSquareRecovery:
    """Lemma 6.1: M_{2,1} encodes squares."""

    def test_m21_recovers_square(self):
        """q=2, p=1 should capture SOS certificates at depth 1."""
        x, y = symbols('x y')
        dsdp = DSDPMeanRelaxation(
            gens=[x, y],
            weights=[1.0, 1.0],
            q=2,
            p=1,
            verbosity=0
        )
        dsdp.SetObjective((x - y)**2)
        lb = dsdp.solve(order=1)
        # (x-y)^2 >= 0, minimum is 0
        assert float(lb) >= -1e-2


class TestWeightValidation:
    """Weights must be positive and match generator count."""

    def test_rejects_negative_weights(self):
        x, y = symbols('x y')
        with pytest.raises(ValueError, match="positive"):
            DSDPRelaxations([x, y], weights=[1.0, -1.0], verbosity=0)

    def test_rejects_mismatched_weights(self):
        x, y = symbols('x y')
        with pytest.raises(ValueError, match="length"):
            DSDPRelaxations([x, y], weights=[1.0], verbosity=0)
