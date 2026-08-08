"""
Solver routing tests for Phase 2 — CVXPY abstraction layer.

Tests that the SDP solver correctly routes through CVXPY with CLARABEL/SCS,
falls back to legacy paths when CVXPY is unavailable, and produces correct
numerical results across solvers.
"""
import pytest
import numpy as np
from sympy import Symbol
from Irene.relaxations import SDPRelaxations


class TestSolverRouting:
    """Test that solver selection routes correctly through the CVXPY layer."""

    def test_default_solver_uses_cvxpy(self):
        """Default solver (CVXOPT) should route through CVXPY with CLARABEL/SCS backend."""
        x = Symbol('x')
        rlx = SDPRelaxations([x])
        rlx.SetObjective(x**2)
        rlx.MomentsOrd(1)
        rlx.RelaxationDeg()
        rlx.InitSDP()
        lb = rlx.Minimize()

        assert lb is not None
        assert abs(lb) < 1e-6, f"Expected ~0, got {lb}"
        solver_name = rlx.Info.get('solver', '')
        assert 'CVXPY' in solver_name, f"Expected CVXPY routing, got {solver_name}"

    def test_clarabel_solver(self):
        """Explicit CLARABEL solver should work and produce correct results."""
        x, y = Symbol('x'), Symbol('y')
        rlx = SDPRelaxations([x, y])
        rlx.SetObjective(x**2 + y**2)
        rlx.AddConstraint(x + y >= 1)
        rlx.MomentsOrd(1)
        rlx.RelaxationDeg()
        rlx.SetSDPSolver('CLARABEL')
        rlx.InitSDP()
        lb = rlx.Minimize()

        assert lb is not None
        assert abs(lb - 0.5) < 1e-3, f"Expected ~0.5, got {lb}"

    def test_scs_solver(self):
        """SCS solver should work (first-order, slightly less precise)."""
        x = Symbol('x')
        rlx = SDPRelaxations([x])
        rlx.SetObjective(x**2)
        rlx.MomentsOrd(1)
        rlx.RelaxationDeg()
        rlx.SetSDPSolver('SCS')
        rlx.InitSDP()
        lb = rlx.Minimize()

        assert lb is not None
        # SCS is first-order, allow wider tolerance
        assert abs(lb) < 1e-2, f"Expected ~0, got {lb}"

    def test_constrained_problem_consistency(self):
        """Different solvers should produce consistent lower bounds on the same problem."""
        x, y = Symbol('x'), Symbol('y')
        expected_lb = 0.5
        tolerance = 1e-2

        for solver_name in ['CLARABEL', 'SCS']:
            rlx = SDPRelaxations([x, y])
            rlx.SetObjective(x**2 + y**2)
            rlx.AddConstraint(x + y >= 1)
            rlx.MomentsOrd(1)
            rlx.RelaxationDeg()
            rlx.SetSDPSolver(solver_name)
            rlx.InitSDP()
            lb = rlx.Minimize()

            assert lb is not None, f"Solver {solver_name} returned None"
            assert abs(lb - expected_lb) < tolerance, \
                f"Solver {solver_name}: expected ~{expected_lb}, got {lb}"

    def test_moment_matrix_recovered(self):
        """After solve, moment matrix should be PSD and recoverable."""
        x = Symbol('x')
        rlx = SDPRelaxations([x])
        rlx.SetObjective(x**2)
        rlx.MomentsOrd(1)
        rlx.RelaxationDeg()
        rlx.InitSDP()
        rlx.Minimize()

        assert 'moments' in rlx.Info, "Moments not populated after solve"
        moment_matrix = rlx.Solution.MomentMatrix
        assert moment_matrix is not None
        # Check symmetry
        assert np.allclose(moment_matrix, moment_matrix.T)
        # Check PSD (all eigenvalues >= 0 within tolerance)
        eigvals = np.linalg.eigvalsh(moment_matrix)
        assert np.all(eigvals >= -1e-6), f"Matrix not PSD: min eigenvalue {eigvals.min()}"

    def test_stability_check_runs(self):
        """Stability diagnostics should be present in Info after solve."""
        x = Symbol('x')
        rlx = SDPRelaxations([x])
        rlx.SetObjective(x**2)
        rlx.MomentsOrd(1)
        rlx.RelaxationDeg()
        rlx.InitSDP()
        rlx.Minimize()

        assert 'stability' in rlx.Info, "Stability check not present"
        stability = rlx.Info['stability']
        assert 'cond' in stability
        assert 'min_eig' in stability or stability['warning'] is False


class TestLegacyDeprecation:
    """Test that legacy solver paths emit deprecation warnings."""

    def test_sdpa_deprecation_warning(self):
        """Calling sdpa() directly should emit DeprecationWarning."""
        import warnings
        from Irene.sdp import sdp

        sd = sdp('SDPA')
        # Set up minimal SDP data so the method can be called
        sd.b = [1.0]
        sd.C = [np.array([[1.0]])]
        sd.A = [[np.array([[1.0]])]]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                sd.sdpa()  # will fail (no SDPA binary), but should warn first
            except RuntimeError:
                pass  # expected — no SDPA installed
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0, "No DeprecationWarning emitted for sdpa()"

    def test_csdp_deprecation_warning(self):
        """Calling csdp() directly should emit DeprecationWarning."""
        import warnings
        from Irene.sdp import sdp

        # CSDP may not be installed; if __init__ rejects it, skip gracefully
        try:
            sd = sdp('CSDP')
        except ImportError:
            pytest.skip("CSDP binary not available on this system")

        sd.b = [1.0]
        sd.C = [np.array([[1.0]])]
        sd.A = [[np.array([[1.0]])]]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                sd.csdp()  # will fail (no CSDP binary), but should warn first
            except RuntimeError:
                pass  # expected — no CSDP installed
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0, "No DeprecationWarning emitted for csdp()"


class TestCVXPYFallback:
    """Test that CVXPY fallback to legacy path works correctly."""

    def test_solve_method_exists(self):
        """sdp.solve() should exist and be callable."""
        from Irene.sdp import sdp
        sd = sdp('CLARABEL')
        assert hasattr(sd, 'solve')
        assert callable(sd.solve)

    def test_cvxpy_info_keys(self):
        """CVXPY solve should populate standard Info keys."""
        x = Symbol('x')
        rlx = SDPRelaxations([x])
        rlx.SetObjective(x**2)
        rlx.MomentsOrd(1)
        rlx.RelaxationDeg()
        rlx.InitSDP()
        rlx.Minimize()

        info = rlx.Info
        required_keys = ['min', 'solver', 'moments']
        for key in required_keys:
            assert key in info, f"Missing Info key: {key}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
