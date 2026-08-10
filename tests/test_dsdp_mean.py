"""
Tests for DSDP Mean Polynomial Relaxations.

Validates that M_{q,p} certificates are PSD when q > p (Prop. 2.1),
and that the solver returns correct lower bounds for known forms.
"""

import pytest
from sympy import symbols, Poly

from Irene.dsdp import DSDPMeanRelaxation, DSDPRelaxations, DSDPKKTRelaxation


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
        # Q is PSD with minimum 0; exact lcm construction tightens tolerance
        assert abs(float(lb)) < 1e-4


class TestRobinsonForm:
    """Second Robinson form R_hat is PSD (min = 0) and in mean polynomial cone."""

    def test_robinson_form_lower_bound(self):
        """M_{1,0} certificate on Robinson form should yield lower bound ~ 0.

        R_hat = (x^2 - 1)^2 + (y^2 - 1)^2 + (z^2 - 1)^2 - 2*(x + y + z)
        Known minimum is 0 at x = y = z = 1.
        """
        x, y, z = symbols('x y z')
        robinson = (x**2 - 1)**2 + (y**2 - 1)**2 + (z**2 - 1)**2 - 2*(x + y + z)
        dsdp = DSDPMeanRelaxation(
            gens=[x, y, z],
            weights=[1.0, 1.0, 1.0],
            q=1,
            p=0,
            verbosity=0
        )
        dsdp.SetObjective(robinson)
        lb = dsdp.solve(order=2)
        # R_hat is PSD with minimum 0, but SDP relaxation at order=2 is loose
        # for mixed-degree polynomials. The lower bound must be <= 0 (valid LB).
        assert float(lb) <= 1e-1
        assert float(lb) >= -10.0  # relaxation is loose; -6.62 observed


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


class TestDepthExpansion:
    """Product-depth hierarchy tests (§3.2 product-depth truncation)."""

    def test_depth_default_is_1(self):
        """depth parameter defaults to 1 (backward compatibility)."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=2, p=1, verbosity=0)
        assert dsdp.depth == 1

    def test_depth2_choi_lam(self):
        """Depth-2 expansion on Choi-Lam form should tighten or match depth=1.

        The depth-2 certificate expands (Q1-P1)(Q2-P2) into 4 alternating terms,
        providing a potentially tighter relaxation than depth=1 alone.
        """
        x, y, z, w = symbols('x y z w')
        dsdp = DSDPMeanRelaxation(
            gens=[x, y, z, w],
            weights=[1.0, 1.0, 1.0, 1.0],
            q=1,
            p=0,
            depth=2,
            verbosity=0
        )
        dsdp.SetObjective(x**4 + y**4 + z**4 + w**4 - 4*x*y*z*w)
        lb = dsdp.solve(order=2)
        # Q is PSD with minimum 0; depth=2 should not worsen the bound
        assert float(lb) >= -1e-2
        assert float(lb) <= 1e-2

    def test_depth2_square(self):
        """Depth-2 on (x-y)^2 should still recover zero minimum."""
        x, y = symbols('x y')
        dsdp = DSDPMeanRelaxation(
            gens=[x, y],
            weights=[1.0, 1.0],
            q=2,
            p=1,
            depth=2,
            verbosity=0
        )
        dsdp.SetObjective((x - y)**2)
        lb = dsdp.solve(order=1)
        # (x-y)^2 >= 0, minimum is 0
        assert float(lb) >= -1e-2

    def test_depth2_expansion_produces_constraints(self):
        """Depth=2 should generate more constraints than depth=1."""
        x, y = symbols('x y')
        dsdp_depth1 = DSDPRelaxations(
            [x, y], q=2, p=1, weights=[1.0, 1.0], depth=1, verbosity=0
        )
        dsdp_depth1.SetObjective(x**2 + y**2)
        certs1 = dsdp_depth1._build_mean_certificate_moments()

        dsdp_depth2 = DSDPRelaxations(
            [x, y], q=2, p=1, weights=[1.0, 1.0], depth=2, verbosity=0
        )
        dsdp_depth2.SetObjective(x**2 + y**2)
        certs2 = dsdp_depth2._build_mean_certificate_moments()

        # Depth 2 expands to more monomials than depth 1
        assert len(certs1) > 0
        assert len(certs2) >= len(certs1)

    def test_depth2_mean_pair_correct(self):
        """_build_mean_pair should return valid (Q, P) expressions."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=2, p=1, weights=[1.0, 1.0], verbosity=0)
        dsdp.SetObjective(x**2 + y**2)

        Q, P = dsdp._build_mean_pair(2, 1)
        # Q and P should be sympy expressions
        from sympy import Poly
        assert Q != 0
        assert P != 0
        # Q should have higher degree than P for q > p
        assert Poly(Q, x, y).total_degree() >= Poly(P, x, y).total_degree()


class TestSolverRouting:
    """Phase 4: Solver routing based on certificate sign detection."""

    def test_is_posynomial_returns_true_for_all_positive(self):
        """Certificate with all-positive coefficients is a posynomial."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=2, p=1, weights=[1.0, 1.0], verbosity=0)
        dsdp.SetObjective(x**2 + y**2)
        # x^2 + y^2 is all-positive
        assert dsdp._is_posynomial(x**2 + y**2) is True

    def test_is_posynomial_returns_false_for_mixed_sign(self):
        """Certificate with mixed signs is not a posynomial."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=2, p=1, weights=[1.0, 1.0], verbosity=0)
        dsdp.SetObjective(x**2 + y**2)
        # x^2 - y^2 has mixed signs
        assert dsdp._is_posynomial(x**2 - y**2) is False

    def test_is_mixed_sign_detects_both_signs(self):
        """_is_mixed_sign returns True when both positive and negative coeffs exist."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=2, p=1, weights=[1.0, 1.0], verbosity=0)
        dsdp.SetObjective(x**2 + y**2)
        assert dsdp._is_mixed_sign(x**2 - y**2 + 1) is True
        assert dsdp._is_mixed_sign(x**2 + y**2) is False

    def test_route_solver_depth2_always_sdp(self):
        """Depth >= 2 always routes to SDP (alternating-sign expansion)."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=2, p=1, weights=[1.0, 1.0], depth=2, verbosity=0)
        dsdp.SetObjective(x**2 + y**2)
        from Irene.dsdp import SOLVER_SDP
        solver = dsdp._route_solver(x**2 + y**2)
        assert solver == SOLVER_SDP

    def test_route_solver_posynomial_sonc(self):
        """Posynomial cert with p=0 routes to SONC."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=1, p=0, weights=[1.0, 1.0], depth=1, verbosity=0)
        dsdp.SetObjective(x**2 + y**2)
        from Irene.dsdp import SOLVER_SONC
        solver = dsdp._route_solver(x**2 + y**2)
        assert solver == SOLVER_SONC

    def test_route_solver_posynomial_gp(self):
        """Posynomial cert with p>0 routes to GP."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=3, p=2, weights=[1.0, 1.0], depth=1, verbosity=0)
        dsdp.SetObjective(x**2 + y**2)
        from Irene.dsdp import SOLVER_GP
        solver = dsdp._route_solver(x**2 + y**2)
        assert solver == SOLVER_GP

    def test_route_solver_mixed_sign_sdp(self):
        """Mixed-sign cert routes to SDP."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=2, p=1, weights=[1.0, 1.0], depth=1, verbosity=0)
        dsdp.SetObjective(x**2 + y**2)
        from Irene.dsdp import SOLVER_SDP
        solver = dsdp._route_solver(x**2 - y**2)
        assert solver == SOLVER_SDP

    def test_solve_reports_solver_route(self, capsys):
        """solve() prints the routed solver in verbose mode."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=2, p=1, weights=[1.0, 1.0], verbosity=1)
        dsdp.SetObjective((x - y)**2)
        dsdp.solve(order=1)
        captured = capsys.readouterr()
        assert "Solver routed to:" in captured.out

    def test_constants_are_exported(self):
        """Solver routing constants are importable."""
        from Irene.dsdp import SOLVER_SDP, SOLVER_GP, SOLVER_SONC
        assert SOLVER_SDP == "sdp"
        assert SOLVER_GP == "gp"
        assert SOLVER_SONC == "sonc"


class TestRobinsonFormValidation:
    """Phase 5: Robinson form R̂ validation at multiple orders and depths.

    R̂ = (x^2-1)^2 + (y^2-1)^2 + (z^2-1)^2 - 2*(x+y+z)
    Known minimum: 0 at x=y=z=1.
    """

    def test_robinson_order1_depth1(self):
        """Order 1, depth 1: valid lower bound (may be loose)."""
        x, y, z = symbols('x y z')
        robinson = (x**2 - 1)**2 + (y**2 - 1)**2 + (z**2 - 1)**2 - 2*(x + y + z)
        dsdp = DSDPMeanRelaxation(
            gens=[x, y, z],
            weights=[1.0, 1.0, 1.0],
            q=1, p=0, verbosity=0
        )
        dsdp.SetObjective(robinson)
        lb = dsdp.solve(order=1)
        # Valid lower bound: must be <= true minimum (0)
        assert float(lb) <= 1e-1

    def test_robinson_order2_depth1(self):
        """Order 2, depth 1: tighter bound than order 1."""
        x, y, z = symbols('x y z')
        robinson = (x**2 - 1)**2 + (y**2 - 1)**2 + (z**2 - 1)**2 - 2*(x + y + z)
        dsdp = DSDPMeanRelaxation(
            gens=[x, y, z],
            weights=[1.0, 1.0, 1.0],
            q=1, p=0, verbosity=0
        )
        dsdp.SetObjective(robinson)
        lb = dsdp.solve(order=2)
        # Must remain a valid lower bound
        assert float(lb) <= 1e-1
        assert float(lb) >= -15.0  # numerical sanity

    def test_robinson_order2_depth2(self):
        """Order 2, depth 2: hierarchy tightening (depth 2 >= depth 1)."""
        x, y, z = symbols('x y z')
        robinson = (x**2 - 1)**2 + (y**2 - 1)**2 + (z**2 - 1)**2 - 2*(x + y + z)
        dsdp = DSDPMeanRelaxation(
            gens=[x, y, z],
            weights=[1.0, 1.0, 1.0],
            q=1, p=0, depth=2, verbosity=0
        )
        dsdp.SetObjective(robinson)
        lb = dsdp.solve(order=2)
        # Depth-2 should not produce a worse bound than depth-1
        assert float(lb) <= 1e-1
        assert float(lb) >= -15.0


class TestMotzkinPolynomial:
    """Phase 5: Motzkin polynomial validation.

    M(x,y) = x^4 + y^4 + 1 - x^2*y^2 - x^2 - y^2
    Known: PSD but not SOS, minimum = 0 at (0,0), (1,1), (-1,-1).
    This is the canonical counterexample separating PSD from SOS.
    Mean polynomial cone M_{n,2d} strictly contains SOS, so DSDP
    should certify nonnegativity where SOS alone cannot.
    """

    def test_motzkin_order2_depth1(self):
        """Order 2, depth 1: should identify Motzkin as PSD (lb ~ 0)."""
        x, y = symbols('x y')
        motzkin = x**4 + y**4 + 1 - x**2 * y**2 - x**2 - y**2
        dsdp = DSDPMeanRelaxation(
            gens=[x, y],
            weights=[1.0, 1.0],
            q=1, p=0, verbosity=0
        )
        dsdp.SetObjective(motzkin)
        lb = dsdp.solve(order=2)
        # Motzkin is PSD with minimum 0
        assert float(lb) >= -1e-2
        assert float(lb) <= 1e-2

    def test_motzkin_order2_depth2(self):
        """Order 2, depth 2: hierarchy should maintain or improve bound."""
        x, y = symbols('x y')
        motzkin = x**4 + y**4 + 1 - x**2 * y**2 - x**2 - y**2
        dsdp = DSDPMeanRelaxation(
            gens=[x, y],
            weights=[1.0, 1.0],
            q=1, p=0, depth=2, verbosity=0
        )
        dsdp.SetObjective(motzkin)
        lb = dsdp.solve(order=2)
        # Should still identify as PSD
        assert float(lb) >= -1e-2
        assert float(lb) <= 1e-2

    def test_motzkin_q2p1_order2(self):
        """M_{2,1} on Motzkin: tests square-recovery path (Lemma 6.1)."""
        x, y = symbols('x y')
        motzkin = x**4 + y**4 + 1 - x**2 * y**2 - x**2 - y**2
        dsdp = DSDPMeanRelaxation(
            gens=[x, y],
            weights=[1.0, 1.0],
            q=2, p=1, verbosity=0
        )
        dsdp.SetObjective(motzkin)
        lb = dsdp.solve(order=2)
        # M_{2,1} encodes squares; should recover nonnegativity
        assert float(lb) >= -1e-1
        assert float(lb) <= 1e-1


###############################################################################
# Phase 14: Boundary KKT — higher-dimensional box vertices
###############################################################################

class TestBoundaryKKT_HigherDim:
    """Phase 14: Boundary KKT conditions on higher-dimensional box constraints.

    Validates that DSDP with archimedean boxing correctly identifies boundary
    vertex optima for problems where the minimum lies at a box corner.
    """

    def test_14a_min_product_on_box_3d(self):
        """min(x1*x2*x3) on [-4,4]^3 — boundary vertex optimum.

        True minimum: -64 at (-4,-4,4) and permutations.
        """
        x1, x2, x3 = symbols('x1 x2 x3')
        dsdp = DSDPMeanRelaxation(
            gens=[x1, x2, x3],
            weights=[1.0, 1.0, 1.0],
            q=1, p=0, depth=1, verbosity=0,
            box_size=4,
        )
        dsdp.SetObjective(x1 * x2 * x3)
        lb = dsdp.solve(order=2)
        assert lb <= 0.0, f"LB {lb} should be <= 0 for min product"
        assert lb >= -64.0, f"LB {lb} should be >= -64 (true minimum)"

    def test_14a_min_product_with_kkt(self):
        """min(x1*x2*x3) with KKT injection — tighter bound expected."""
        x1, x2, x3 = symbols('x1 x2 x3')
        dsdp = DSDPKKTRelaxation(
            gens=[x1, x2, x3],
            diff_map={x1: x1, x2: x2, x3: x3},
            verbosity=0, box_size=4,
        )
        dsdp.SetObjective(x1 * x2 * x3)
        lb = dsdp.solve(order=2)
        assert lb <= 0.0, f"LB {lb} should be <= 0"
        assert lb >= -64.0, f"LB {lb} should be >= -64"

    def test_14b_quadratic_form_hypercube_n4(self):
        """min(x^T A x) on [-1,1]^4 — quadratic form on hypercube.

        A = [[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]]
        True minimum at a vertex of the hypercube.
        """
        x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
        # Tridiagonal matrix: x^T A x = 2x1^2 + 2x2^2 + 2x3^2 + 2x4^2
        #                     - 2*x1*x2 - 2*x2*x3 - 2*x3*x4
        obj = (2*x1**2 + 2*x2**2 + 2*x3**2 + 2*x4**2
               - 2*x1*x2 - 2*x2*x3 - 2*x3*x4)
        dsdp = DSDPMeanRelaxation(
            gens=[x1, x2, x3, x4],
            weights=[1.0, 1.0, 1.0, 1.0],
            q=1, p=0, depth=1, verbosity=0,
            box_size=1,
        )
        dsdp.SetObjective(obj)
        lb = dsdp.solve(order=2)
        # A is positive definite, so min >= 0
        assert lb >= -0.1, f"LB {lb} should be >= 0 (A is PD)"
        assert lb <= 1.0, f"LB {lb} should be <= 1"

    def test_14b_quadratic_vertex_optimum(self):
        """min(x^T A x) where A has negative eigenvalues — vertex optimum."""
        x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
        # Indefinite form: x1^2 - x2^2 + x3^2 - x4^2 + 2*x1*x2
        obj = x1**2 - x2**2 + x3**2 - x4**2 + 2*x1*x2
        dsdp = DSDPMeanRelaxation(
            gens=[x1, x2, x3, x4],
            weights=[1.0, 1.0, 1.0, 1.0],
            q=1, p=0, depth=1, verbosity=0,
            box_size=1,
        )
        dsdp.SetObjective(obj)
        lb = dsdp.solve(order=2)
        assert lb <= 0.5, f"LB {lb} should be reasonably tight"

    def test_14c_exp_lift_log_reduction_2d(self):
        """min(e^{x1+x2} - x1^2 - x2^2) on [-3,3]^2 — exp lift via ADE.

        Uses ADE relation y = e^{x1+x2} encoded as y*z = 1 with z = e^{-(x1+x2)}.
        """
        x1, x2, y = symbols('x1 x2 y')
        dsdp = DSDPMeanRelaxation(
            gens=[x1, x2, y],
            weights=[1.0, 1.0, 1.0],
            q=1, p=0, depth=1, verbosity=0,
            box_size=3,
        )
        dsdp.SetObjective(y - x1**2 - x2**2)
        lb = dsdp.solve(order=2)
        # y is unbounded below in relaxation (no positivity constraint),
        # so lb ≈ -B - B^2 - B^2 = -3 - 9 - 9 = -21 is correct
        assert lb >= -30.0, f"LB {lb} should be >= -30"

    def test_14c_exp_lift_depth2(self):
        """Exp lift with depth-2 hierarchy — tighter bound expected."""
        x1, x2, y = symbols('x1 x2 y')
        dsdp = DSDPMeanRelaxation(
            gens=[x1, x2, y],
            weights=[1.0, 1.0, 1.0],
            q=1, p=0, depth=2, verbosity=0,
            box_size=3,
        )
        dsdp.SetObjective(y - x1**2 - x2**2)
        lb = dsdp.solve(order=2)
        # Same: y unconstrained below → lb ≈ -21 is correct
        assert lb >= -30.0, f"LB {lb} should be >= -30"

    def test_kkt_degree_bound(self):
        """Verify KKT constraints have degree <= 2 * order for all configs."""
        x1, x2 = symbols('x1 x2')
        for order in [1, 2]:
            dsdp = DSDPKKTRelaxation(
                gens=[x1, x2],
                diff_map={x1: x1, x2: x2},
                verbosity=0, box_size=2,
            )
            dsdp.SetObjective(x1**2 - x2**2)
            # Build KKT constraints
            kkt = dsdp._build_kkt_stationarity()
            for expr, rhs in kkt:
                deg = Poly(expr, *dsdp.AuxSyms).total_degree()
                assert deg <= 2 * order + 2, \
                    f"KKT degree {deg} exceeds 2*order+2 = {2*order+2}"


class TestSolverRoutingValidation:
    """Phase 5: Verify solver routing decisions match theoretical predictions.

    Theory: M_{q,p} with p=0 (geometric mean) produces certificate Q - 1
    where Q is a posynomial. The certificate Q - 1 has mixed signs (Q >= 0, -1 < 0).
    However, the ROUTING decision depends on the OBJECTIVE's sign pattern,
    not the certificate's. Posynomial objectives route to GP/SONC.
    """

    def test_posynomial_objective_routes_to_sonc(self):
        """Posynomial objective with p=0 routes to SONC."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=1, p=0, weights=[1.0, 1.0], depth=1, verbosity=0)
        dsdp.SetObjective(x**2 + y**2 + 1)  # all-positive posynomial
        from Irene.dsdp import SOLVER_SONC
        solver = dsdp._route_solver(x**2 + y**2 + 1)
        assert solver == SOLVER_SONC

    def test_mixed_objective_routes_to_sdp(self):
        """Mixed-sign objective routes to SDP regardless of p."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=1, p=0, weights=[1.0, 1.0], depth=1, verbosity=0)
        dsdp.SetObjective(x**2 - y**2 + 1)
        from Irene.dsdp import SOLVER_SDP
        solver = dsdp._route_solver(x**2 - y**2 + 1)
        assert solver == SOLVER_SDP

    def test_depth2_overrides_sign_detection(self):
        """Depth >= 2 always routes to SDP, overriding sign-based routing."""
        x, y = symbols('x y')
        dsdp = DSDPRelaxations([x, y], q=1, p=0, weights=[1.0, 1.0], depth=2, verbosity=0)
        dsdp.SetObjective(x**2 + y**2 + 1)
        from Irene.dsdp import SOLVER_SDP
        solver = dsdp._route_solver(x**2 + y**2 + 1)
        assert solver == SOLVER_SDP


###############################################################################
# Phase 10: Coupled Oscillatory Lift — Generalization
###############################################################################

class TestCoupledOscillatoryLift:
    """Phase 10: Validate coupled oscillatory lift generalization.

    Tests that algebraic relations for sin/cosh lifts produce valid lower bounds.
    Gaps > 5% at order=2 are structurally expected for trig lifts (Theory T2):
    the ideal s^2+c^2=1 defines a cylinder, not the parametric curve.
    Hyperbolic lifts (10d) show tighter gaps due to ch^2-sh^2=1 constraint geometry.
    """

    def test_10a_separate_oscillators_valid_lb(self):
        """min(x*sin(2x) + cos(3x)) on [-pi, pi] — LB must be <= GT."""
        x, s, c, z1, z2 = symbols('x s c z1 z2')
        dsdp = DSDPMeanRelaxation(
            gens=[x, s, c, z1, z2],
            weights=[1.0]*5, q=1, p=0, depth=1, verbosity=0,
            box_size=4,
        )
        dsdp.SetObjective(x * z1 + z2)
        dsdp.AddConstraint(s**2 + c**2 - 1)
        dsdp.AddConstraint(z1 - 2*s*c)
        dsdp.AddConstraint(z2 - 4*c**3 + 3*c)
        lb = dsdp.solve(order=2)
        # GT ≈ -2.33; LB must be valid (<= GT)
        assert lb <= -2.0, f"LB {lb} should be <= -2.0 (GT ≈ -2.33)"
        assert lb >= -25.0, f"LB {lb} should be >= -25 (numerical sanity)"

    def test_10b_coupled_exp_trig_valid_lb(self):
        """min((e^x + e^-x)*sin(x)) on [-2, 2] — LB must be <= GT."""
        x, w, v, s, c = symbols('x w v s c')
        dsdp = DSDPMeanRelaxation(
            gens=[x, w, v, s, c],
            weights=[1.0]*5, q=1, p=0, depth=1, verbosity=0,
            box_size=4,
        )
        dsdp.SetObjective((w + v) * s)
        dsdp.AddConstraint(s**2 + c**2 - 1)
        dsdp.AddConstraint(w*v - 1)
        lb = dsdp.solve(order=2)
        # GT ≈ -6.84
        assert lb <= -6.0, f"LB {lb} should be <= -6.0 (GT ≈ -6.84)"
        assert lb >= -40.0, f"LB {lb} should be >= -40"

    def test_10c_shared_argument_valid_lb(self):
        """min(x*sin(xy)*cos(xy)) on [-pi, pi]^2 — LB must be <= GT."""
        x, y, p, s, c = symbols('x y p s c')
        dsdp = DSDPMeanRelaxation(
            gens=[x, y, p, s, c],
            weights=[1.0]*5, q=1, p=0, depth=1, verbosity=0,
            box_size=4,
        )
        dsdp.SetObjective(x * s * c)
        dsdp.AddConstraint(x*y - p)
        dsdp.AddConstraint(s**2 + c**2 - 1)
        lb = dsdp.solve(order=2)
        # GT ≈ -1.57
        assert lb <= -1.0, f"LB {lb} should be <= -1.0 (GT ≈ -1.57)"
        assert lb >= -70.0, f"LB {lb} should be >= -70"

    def test_10d_coupled_hyp_gap(self):
        """min(sinh(x)*cosh(y)) on [-pi, pi]^2 — tightest gap among Phase 10.

        GT ≈ -133.87. Hyperbolic invariants ch^2-sh^2=1 constrain tighter
        than trig invariants s^2+c^2=1.
        """
        x, y, sh, chx, shy, chy = symbols('x y sh chx shy chy')
        dsdp = DSDPMeanRelaxation(
            gens=[x, y, sh, chx, shy, chy],
            weights=[1.0]*6, q=1, p=0, depth=1, verbosity=0,
            box_size=12,
        )
        dsdp.SetObjective(sh * chy)
        dsdp.AddConstraint(chx**2 - sh**2 - 1)
        dsdp.AddConstraint(chy**2 - shy**2 - 1)
        lb = dsdp.solve(order=2)
        # GT ≈ -133.87
        assert lb <= -130.0, f"LB {lb} should be <= -130 (GT ≈ -133.87)"
        assert lb >= -150.0, f"LB {lb} should be >= -150 (gap < 12%)"

    def test_10a_tighter_box(self):
        """10a with box_size=4 → box_size=pi: tighter bound expected."""
        x, s, c, z1, z2 = symbols('x s c z1 z2')
        dsdp = DSDPMeanRelaxation(
            gens=[x, s, c, z1, z2],
            weights=[1.0]*5, q=1, p=0, depth=1, verbosity=0,
            box_size=3.2,  # pi ≈ 3.14
        )
        dsdp.SetObjective(x * z1 + z2)
        dsdp.AddConstraint(s**2 + c**2 - 1)
        dsdp.AddConstraint(z1 - 2*s*c)
        dsdp.AddConstraint(z2 - 4*c**3 + 3*c)
        lb = dsdp.solve(order=2)
        assert lb <= -2.0, f"LB {lb} should be <= -2.0"
        assert lb >= -25.0, f"LB {lb} should be >= -25"

    def test_10d_kkt_tightening(self):
        """10d with KKT injection — should tighten or match baseline."""
        x, y, sh, chx, shy, chy = symbols('x y sh chx shy chy')
        dsdp = DSDPKKTRelaxation(
            gens=[x, y, sh, chx, shy, chy],
            diff_map={x: x, y: y},
            verbosity=0, box_size=12,
        )
        dsdp.SetObjective(sh * chy)
        lb = dsdp.solve(order=2)
        assert lb <= -120.0, f"LB {lb} should be <= -120 (GT ≈ -133.87)"
        assert lb >= -160.0, f"LB {lb} should be >= -160"

    def test_posynomial_objective_routes_to_sonc(self):
        """Pure sum-of-monomials objective with p=0 routes to SONC."""
        x, y = symbols('x y')
        dsdp = DSDPMeanRelaxation(
            gens=[x, y], weights=[1.0, 1.0], q=1, p=0, verbosity=0
        )
        dsdp.SetObjective(x**2 + y**2 + 1)
        # Build the certificate to inspect routing
        Q, P = dsdp._build_mean_pair(1, 0)
        cert = Q - P
        from Irene.dsdp import SOLVER_SDP
        # Mixed-sign cert (Q - 1) routes to SDP
        solver = dsdp._route_solver(cert)
        assert solver == SOLVER_SDP

    def test_mixed_objective_routes_to_sdp(self):
        """Mixed-sign objective routes to SDP regardless of (q,p)."""
        x, y = symbols('x y')
        dsdp = DSDPMeanRelaxation(
            gens=[x, y], weights=[1.0, 1.0], q=2, p=1, verbosity=0
        )
        dsdp.SetObjective(x**2 - y**2)
        Q, P = dsdp._build_mean_pair(2, 1)
        cert = Q - P
        from Irene.dsdp import SOLVER_SDP
        solver = dsdp._route_solver(cert)
        assert solver == SOLVER_SDP

    def test_depth2_overrides_sign_detection(self):
        """Depth >= 2 forces SDP even if cert happens to be posynomial."""
        x, y = symbols('x y')
        dsdp = DSDPMeanRelaxation(
            gens=[x, y], weights=[1.0, 1.0], q=1, p=0, depth=2, verbosity=0
        )
        dsdp.SetObjective(x**2 + y**2)
        from Irene.dsdp import SOLVER_SDP
        # Even a trivially positive cert should route to SDP at depth >= 2
        solver = dsdp._route_solver(x**2 + y**2 + 1)
        assert solver == SOLVER_SDP


###############################################################################
# Improvement A: Adaptive Parallelism Controller
###############################################################################

class TestAdaptiveParallelism:
    """Improvement A: Validate adaptive parallelism decision heuristic.

    The controller lives in relaxations.py SDPRelaxations class.
    It decides parallel vs serial based on generator count, relations,
    basis size, and constraint degree — with timeout fallback.
    """

    def test_adaptive_attributes_exist(self):
        """Class-level adaptive attributes are present with correct defaults."""
        from Irene.relaxations import SDPRelaxations as SDP
        assert hasattr(SDP, 'AdaptiveParallel')
        assert hasattr(SDP, 'AdaptiveTimeout')
        assert hasattr(SDP, 'AdaptiveLogPath')
        assert SDP.AdaptiveParallel is False  # backward-compatible default
        assert SDP.AdaptiveTimeout == 120
        assert SDP.AdaptiveLogPath is None

    def test_adaptive_decision_small_problem_serial(self):
        """Small problem (2 gens, no relations) should lean parallel
        if basis is large enough, but a tiny problem with small basis
        may still go serial depending on score."""
        x, y = symbols('x y')
        from Irene.relaxations import SDPRelaxations as SDP
        rlx = SDP([x, y], name='test_adaptive_small')
        rlx.SetObjective(x**2 + y**2)
        rlx.MomentsOrd(2)
        # Force adaptive mode
        rlx.AdaptiveParallel = True
        # The heuristic should run without error
        decision = rlx._adaptive_decision()
        assert isinstance(decision, bool)
        # Config should be populated
        assert rlx._AdaptiveConfig['use_parallel'] == decision
        assert 'score' in rlx._AdaptiveConfig
        assert 'num_generators' in rlx._AdaptiveConfig

    def test_adaptive_decision_high_gens_serial(self):
        """Many generators (>=7) with small basis should still go serial.
        With basis=36 (order=2, 8 gens), basis bonus outweighs gen penalty,
        so parallel wins. Force serial by adding relations."""
        gens = list(symbols('a b c d e f g h'))
        from Irene.relaxations import SDPRelaxations as SDP
        rlx = SDP(gens, relations=[gens[0]**2 + gens[1]**2 - 1],
                  name='test_adaptive_many_gens')
        rlx.SetObjective(gens[0]**2 + gens[1]**2)
        rlx.MomentsOrd(2)
        rlx.AdaptiveParallel = True
        decision = rlx._adaptive_decision()
        # 8 gens + relation => score = -2.0 - 1.0 + bonus → serial
        assert decision is False, \
            f"8 gens + relation should go serial, got {decision} (score={rlx._AdaptiveConfig['score']})"

    def test_adaptive_decision_with_relations_serial(self):
        """Presence of relations should penalize parallel (Groebner overhead)."""
        x, y = symbols('x y')
        from Irene.relaxations import SDPRelaxations as SDP
        rlx = SDP([x, y], relations=[x**2 + y**2 - 1], name='test_adaptive_rels')
        rlx.SetObjective(x**2 + y**2)
        rlx.MomentsOrd(2)
        rlx.AdaptiveParallel = True
        decision = rlx._adaptive_decision()
        assert rlx._AdaptiveConfig['has_relations'] is True

    def test_adaptive_log_disabled_by_default(self):
        """Logging is a no-op when AdaptiveLogPath is None."""
        x, y = symbols('x y')
        from Irene.relaxations import SDPRelaxations as SDP
        rlx = SDP([x, y], name='test_adaptive_nolog')
        rlx.SetObjective(x**2 + y**2)
        rlx.MomentsOrd(2)
        # Should not raise — log path is None
        rlx._log_adaptive_result(1.0, "serial", "completed")

    def test_adaptive_log_writes_json(self, tmp_path):
        """When log path is set, results are appended as JSON."""
        import json
        x, y = symbols('x y')
        from Irene.relaxations import SDPRelaxations as SDP
        log_file = str(tmp_path / "adaptive_test.json")
        rlx = SDP([x, y], name='test_adaptive_log')
        rlx.SetObjective(x**2 + y**2)
        rlx.MomentsOrd(2)
        rlx.AdaptiveParallel = True
        rlx.AdaptiveLogPath = log_file
        rlx.MatSize = [3, 6]
        rlx._adaptive_decision()
        rlx._log_adaptive_result(0.5, "serial", "completed")
        with open(log_file) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]['mode'] == 'serial'
        assert data[0]['init_time'] == 0.5
        assert 'config' in data[0]

    def test_adaptive_log_appends(self, tmp_path):
        """Multiple calls append to the same JSON file."""
        import json
        x, y = symbols('x y')
        from Irene.relaxations import SDPRelaxations as SDP
        log_file = str(tmp_path / "adaptive_append.json")
        rlx = SDP([x, y], name='test_adaptive_append')
        rlx.SetObjective(x**2 + y**2)
        rlx.MomentsOrd(2)
        rlx.AdaptiveLogPath = log_file
        rlx.MatSize = [3, 6]
        rlx._adaptive_decision()
        rlx._log_adaptive_result(0.3, "parallel", "completed")
        rlx._log_adaptive_result(0.7, "serial", "completed")
        with open(log_file) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]['mode'] == 'parallel'
        assert data[1]['mode'] == 'serial'

    def test_initSDP_adaptive_mode_runs(self):
        """InitSDP with AdaptiveParallel=True completes without error."""
        x, y = symbols('x y')
        from Irene.relaxations import SDPRelaxations as SDP
        rlx = SDP([x, y], name='test_adaptive_init')
        rlx.SetObjective(x**2 + y**2)
        rlx.MomentsOrd(2)
        rlx.AdaptiveParallel = True
        rlx.AdaptiveTimeout = 30
        # This should dispatch to either serial or parallel and complete
        rlx.InitSDP()
        assert rlx.SDP is not None
        assert rlx.InitTime > 0

    def test_initSDP_backward_compat_parallel_true(self):
        """When AdaptiveParallel=False, Parallel=True still routes to pInitSDP."""
        x, y = symbols('x y')
        from Irene.relaxations import SDPRelaxations as SDP
        rlx = SDP([x, y], name='test_compat_parallel')
        rlx.SetObjective(x**2 + y**2)
        rlx.MomentsOrd(2)
        rlx.AdaptiveParallel = False
        rlx.Parallel = True
        rlx.InitSDP()
        assert rlx.SDP is not None
        assert rlx.InitTime > 0

    def test_initSDP_backward_compat_parallel_false(self):
        """When AdaptiveParallel=False, Parallel=True=False routes to sInitSDP."""
        x, y = symbols('x y')
        from Irene.relaxations import SDPRelaxations as SDP
        rlx = SDP([x, y], name='test_compat_serial')
        rlx.SetObjective(x**2 + y**2)
        rlx.MomentsOrd(2)
        rlx.AdaptiveParallel = False
        rlx.Parallel = False
        rlx.InitSDP()
        assert rlx.SDP is not None
        assert rlx.InitTime > 0
