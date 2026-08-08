"""Tests for correlative sparsity detection (P3.4).

Validates:
  - UnionFind correctness (path compression, rank, component counting)
  - CorrelativeSparsity graph construction from term co-occurrence
  - Connected component detection and moment matrix partitioning
  - Reduction factor estimation
  - Integration with OptimizationProblem via detect_sparsity_from_problem
"""

import pytest
from Irene.sparsity import (
    UnionFind,
    CorrelativeSparsity,
    detect_sparsity_from_polys,
)


# ---------------------------------------------------------------------------
# UnionFind unit tests
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_initial_state(self):
        uf = UnionFind(5)
        assert uf.components == 5
        for i in range(5):
            assert uf.find(i) == i

    def test_union_reduces_components(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        assert uf.components == 3
        uf.union(2, 3)
        assert uf.components == 2
        uf.union(0, 2)
        assert uf.components == 1

    def test_path_compression(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        # All should resolve to same root
        root = uf.find(0)
        for i in range(5):
            assert uf.find(i) == root

    def test_get_components(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        # Components: {0,1}, {2,3}, {4}, {5}
        comps = uf.get_components()
        assert len(comps) == 4

    def test_idempotent_union(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        assert not uf.union(0, 1)  # Already same set
        assert uf.components == 2


# ---------------------------------------------------------------------------
# CorrelativeSparsity tests
# ---------------------------------------------------------------------------

class TestCorrelativeSparsity:
    def test_single_variable_no_sparsity(self):
        sp = CorrelativeSparsity(1)
        sp.add_term([0])
        sp.finalize()
        assert not sp.is_sparse
        assert len(sp.components) == 1

    def test_disjoint_variables_are_sparse(self):
        """x and y never co-occur -> two components."""
        sp = CorrelativeSparsity(2)
        sp.add_term([0])  # x alone
        sp.add_term([1])  # y alone
        sp.finalize()
        assert sp.is_sparse
        assert len(sp.components) == 2

    def test_connected_variables_not_sparse(self):
        """x and y co-occur -> one component."""
        sp = CorrelativeSparsity(2)
        sp.add_term([0, 1])  # xy term
        sp.finalize()
        assert not sp.is_sparse
        assert len(sp.components) == 1

    def test_three_variable_partial_sparsity(self):
        """x-y connected, z isolated -> two components."""
        sp = CorrelativeSparsity(3)
        sp.add_term([0, 1])  # xy
        sp.add_term([0])     # x alone
        sp.add_term([2])     # z alone
        sp.finalize()
        assert sp.is_sparse
        assert len(sp.components) == 2

    def test_moment_matrix_partition(self):
        """Partition exponents by component membership."""
        sp = CorrelativeSparsity(3)
        sp.add_term([0])     # x alone -> comp A
        sp.add_term([1])     # y alone -> comp B
        sp.add_term([2])     # z alone -> comp C
        sp.finalize()

        partitions = sp.moment_matrix_partition(deg=1)
        # Each variable is its own component; (0,0,0) goes to first comp
        assert -1 in partitions  # cross-component block exists
        total = sum(len(v) for v in partitions.values())
        assert total > 0

    def test_reduction_factor_dense(self):
        """Dense problem should have reduction factor ~1."""
        sp = CorrelativeSparsity(2)
        sp.add_term([0, 1])  # fully connected
        sp.finalize()
        factor = sp.reduction_factor(deg=2)
        assert abs(factor - 1.0) < 0.01

    def test_reduction_factor_sparse(self):
        """Sparse problem should have reduction factor < 1."""
        sp = CorrelativeSparsity(4)
        sp.add_term([0])     # x alone
        sp.add_term([1])     # y alone
        sp.add_term([2])     # z alone
        sp.add_term([3])     # w alone
        sp.finalize()
        assert sp.is_sparse
        factor = sp.reduction_factor(deg=2)
        assert factor < 1.0

    def test_summary(self):
        sp = CorrelativeSparsity(3)
        sp.add_term([0, 1])
        sp.add_term([2])
        sp.finalize()
        summary = sp.summary()
        assert summary["num_vars"] == 3
        assert summary["is_sparse"] is True
        assert sum(summary["component_sizes"]) == 3

    def test_add_poly_terms(self):
        """Test adding edges from exponent dict."""
        sp = CorrelativeSparsity(3)
        # Two terms: (1,0,0) -> x alone; (0,1,0) -> y alone
        exp_dict = {(1, 0, 0): 1.0, (0, 1, 0): 2.0}
        sp.add_poly_terms(exp_dict)
        sp.finalize()
        assert sp.is_sparse


# ---------------------------------------------------------------------------
# Integration: detect_sparsity_from_polys
# ---------------------------------------------------------------------------

class TestDetectSparsityFromPolys:
    def test_from_symbolic_polys(self):
        from Irene.symbolic_engine import engine
        x, y = engine.symbols('x y')
        # f1 = x^2 + 1 (only x), f2 = y^3 - y (only y) -> sparse
        polys = [x**2 + 1, y**3 - y]
        sp = detect_sparsity_from_polys(polys, num_vars=2)
        assert sp.is_sparse

    def test_connected_polys(self):
        from Irene.symbolic_engine import engine
        x, y = engine.symbols('x y')
        # f1 = xy + 1 (x and y co-occur) -> not sparse
        polys = [x * y + 1]
        sp = detect_sparsity_from_polys(polys, num_vars=2)
        assert not sp.is_sparse
