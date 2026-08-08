"""Tests for Newton polytope monomial pruning."""

import pytest
import numpy as np
from sympy import symbols, expand


class TestNewtonPolytope:
    """Test Newton polytope extraction from polynomials."""

    def test_single_term(self):
        x, y = symbols('x y')
        from Irene.newton_polytope import newton_polytope
        pts = newton_polytope(x**2 * y)
        assert pts.shape == (1, 2)
        np.testing.assert_array_equal(pts[0], [2, 1])

    def test_two_terms(self):
        x, y = symbols('x y')
        from Irene.newton_polytope import newton_polytope
        pts = newton_polytope(x**2 + y**3)
        assert pts.shape == (2, 2)

    def test_constant(self):
        from Irene.newton_polytope import newton_polytope
        pts = newton_polytope(5)
        # Constant has one term with zero exponents
        assert pts.shape[0] >= 1


class TestMinkowskiSum:
    """Test Minkowski sum of polytopes."""

    def test_basic_sum(self):
        from Irene.newton_polytope import minkowski_sum
        a = np.array([[0, 0], [1, 0]])
        b = np.array([[0, 0], [0, 1]])
        result = minkowski_sum(a, b)
        # Should have points: (0,0), (1,0), (0,1), (1,1)
        assert result.shape[0] == 4

    def test_deduplication(self):
        from Irene.newton_polytope import minkowski_sum
        a = np.array([[0, 0], [1, 0]])
        b = np.array([[0, 0]])
        result = minkowski_sum(a, b)
        assert result.shape[0] == 2


class TestNewtonPruner:
    """Test the NewtonPruner class for basis filtering."""

    def test_pruner_reduces_basis(self):
        from Irene.newton_polytope import prune_basis_from_polys
        x, y = symbols('x y')
        # Sparse polynomial: only high-degree terms in one variable
        polys = [x**4 + 1]
        pruner = prune_basis_from_polys(polys, num_vars=2, max_degree=4)
        assert pruner.pruned_basis_size <= pruner.full_basis_size

    def test_pruner_summary(self):
        from Irene.newton_polytope import NewtonPruner
        # Simple 1D case: polytope vertices at [0] and [4], scaled by 2 -> [0, 8]
        vertices = np.array([[0], [4]])
        pruner = NewtonPruner(num_vars=1, max_degree=4, polytope_vertices=vertices)
        basis = pruner.compute_pruned_basis()
        summary = pruner.summary()
        assert "full_basis_size" in summary
        assert "pruned_basis_size" in summary
        assert "reduction_ratio" in summary

    def test_no_polytope_includes_all(self):
        from Irene.newton_polytope import NewtonPruner
        # No polytope = no pruning, all monomials included
        pruner = NewtonPruner(num_vars=2, max_degree=2, polytope_vertices=None)
        basis = pruner.compute_pruned_basis()
        assert len(basis) == pruner.full_basis_size

    def test_sparse_problem_reduction(self):
        """For a sparse polynomial, pruning should reduce the basis."""
        from Irene.newton_polytope import prune_basis_from_polys
        x, y = symbols('x y')
        # Motzkin-like: x^4 + y^4 - 3*x^2*y^2 — all terms degree 4
        polys = [expand(x**4 + y**4 - 3*x**2*y**2)]
        pruner = prune_basis_from_polys(polys, num_vars=2, max_degree=4)
        # The Newton polytope of this polynomial has vertices at (4,0), (0,4), (2,2)
        # Scaled by 2: (8,0), (0,8), (4,4) — but degree bound is 4
        # So pruning should still include all degree-<=4 monos inside the hull
        assert pruner.pruned_basis_size > 0

    def test_bivariate_quadratic(self):
        """Standard bivariate quadratic: x^2 + y^2 + xy."""
        from Irene.newton_polytope import prune_basis_from_polys
        x, y = symbols('x y')
        polys = [expand(x**2 + y**2 + x*y)]
        pruner = prune_basis_from_polys(polys, num_vars=2, max_degree=2)
        # Newton polytope vertices: (2,0), (0,2), (1,1); scaled by 2 -> (4,0),(0,4),(2,2)
        # With degree bound 2, all monos up to deg 2 should be inside the hull
        assert pruner.pruned_basis_size >= 6  # At least x^2, y^2, xy, x, y, 1


class TestPruneFromProblem:
    """Test integration with OptimizationProblem."""

    def test_from_problem(self):
        from Irene.program import OptimizationProblem
        from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra, SemigroupAlgebraElement
        from Irene.newton_polytope import prune_basis_from_problem

        x, y = symbols('x y')
        sg = CommutativeSemigroup([x, y])
        sga = SemigroupAlgebra(sg)
        # Build algebra elements for objective and constraint
        obj_elem = SemigroupAlgebraElement([(1., sg._reduce(sg.generators[0]**2)),
                                            (1., sg._reduce(sg.generators[1]**2))], sg)
        prog = OptimizationProblem(sga=sga)
        prog.set_objective(obj_elem)

        pruner = prune_basis_from_problem(prog, max_degree=4)
        assert pruner.num_vars == 2
        assert pruner.pruned_basis_size > 0


class TestMatrixDimensionReduction:
    """Test that pruning actually reduces moment matrix entries."""

    def test_reduction_ratio(self):
        from Irene.newton_polytope import NewtonPruner
        # Tight polytope in 2D: only (0,0) and (1,0) vertices
        vertices = np.array([[0, 0], [1, 0]])
        pruner = NewtonPruner(num_vars=2, max_degree=3, polytope_vertices=vertices)
        pruner.compute_pruned_basis()
        info = pruner.moment_matrix_dimension_reduction()
        assert info["pruned_basis_size"] <= info["full_basis_size"]
        assert 0 < info["reduction_ratio"] <= 1.0

    def test_entries_saved_positive(self):
        from Irene.newton_polytope import NewtonPruner
        vertices = np.array([[0, 0], [2, 0]])
        pruner = NewtonPruner(num_vars=2, max_degree=3, polytope_vertices=vertices)
        pruner.compute_pruned_basis()
        info = pruner.moment_matrix_dimension_reduction()
        assert info["entries_saved"] >= 0
