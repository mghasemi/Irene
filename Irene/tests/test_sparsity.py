"""Tests for correlative sparsity detection (P3.4).

Validates:
  - UnionFind correctness (path compression, rank, component counting)
  - CorrelativeSparsity graph construction from term co-occurrence
  - Connected component detection and moment matrix partitioning
  - Reduction factor estimation
  - Integration with OptimizationProblem via detect_sparsity_from_problem
"""

import pytest
import sympy as _sp
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


# ---------------------------------------------------------------------------
# P5.8: Sparsity-block SDP decomposition tests
# ---------------------------------------------------------------------------

class TestSparsityBlockSDP:
    """Test that sparsity_block_sdp routes through _sInitSDP_sparse and solves correctly."""

    def test_block_diagonal_decomposition(self):
        """Block-diagonal objective should decompose into independent clique SDPs.

        min x1^2 + x2^2  (no cross terms, no constraints)
        Variables {x1} and {x2} are in separate cliques.
        Expected lower bound: 0 (achieved at origin).
        """
        from Irene.relaxations import SDPRelaxations, RelaxationConfig

        x1, x2 = _sp.symbols('x1 x2')
        config = RelaxationConfig(
            reduction_method="none",
            sparsity_block_sdp=True,
            verbose_reduction=False,
        )
        rlx = SDPRelaxations([x1, x2], config=config)
        rlx.SetObjective(x1**2 + x2**2)
        rlx.MmntOrd = 2

        # Run the sparse path directly
        rlx._sInitSDP_sparse()

        # Should find lower bound close to 0 (feasible at origin)
        assert rlx.f_min is not None
        assert rlx.f_min <= 1e-3  # within tolerance of true minimum 0

    def test_separable_with_constraints(self):
        """Separable objective with per-clique constraints.

        min x1^2 + x2^2
        s.t. (x1 - 1)^2 >= 0, (x2 - 1)^2 >= 0
        Both constraints are clique-local; decomposition should still work.
        """
        from Irene.relaxations import SDPRelaxations, RelaxationConfig

        x1, x2 = _sp.symbols('x1 x2')
        config = RelaxationConfig(
            reduction_method="none",
            sparsity_block_sdp=True,
            verbose_reduction=False,
        )
        rlx = SDPRelaxations([x1, x2], config=config)
        rlx.SetObjective(x1**2 + x2**2)
        rlx.AddConstraint((x1 - 1)**2 >= 0)
        rlx.AddConstraint((x2 - 1)**2 >= 0)
        rlx.MmntOrd = 2

        rlx._sInitSDP_sparse()

        assert rlx.f_min is not None
        # Lower bound should be <= true minimum (0 at origin, constraints satisfied)
        assert rlx.f_min + 1e-6 <= 0.1

    def test_dense_problem_fallback(self):
        """Dense problem (Motzkin-like) should fall back to monolithic SDP."""
        from Irene.relaxations import SDPRelaxations, RelaxationConfig

        x, y = _sp.symbols('x y')
        config = RelaxationConfig(
            reduction_method="none",
            sparsity_block_sdp=True,
            verbose_reduction=False,
        )
        rlx = SDPRelaxations([x, y], config=config)
        # Motzkin polynomial -- dense in both variables
        rlx.SetObjective(x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1)
        rlx.MmntOrd = 2

        # Should fall back gracefully (no decomposition for dense problem)
        rlx._sInitSDP_sparse()

        assert rlx.f_min is not None
        # Motzkin is non-negative, so lower bound should be >= -tolerance
        assert rlx.f_min >= -1e-3

    def test_init_sdp_dispatches_sparse_path(self):
        """InitSDP() should route through _sInitSDP_sparse when config enables it."""
        from Irene.relaxations import SDPRelaxations, RelaxationConfig

        x1, x2 = _sp.symbols('x1 x2')
        config = RelaxationConfig(
            reduction_method="none",
            sparsity_block_sdp=True,
            verbose_reduction=False,
        )
        rlx = SDPRelaxations([x1, x2], config=config)
        rlx.SetObjective(x1**2 + x2**2)
        rlx.MmntOrd = 2
        rlx.Parallel = False

        # Patch _sInitSDP_sparse to verify it was called
        original_sparse = rlx._sInitSDP_sparse
        called = [False]

        def spy_sparse():
            called[0] = True
            return original_sparse()

        rlx._sInitSDP_sparse = spy_sparse

        rlx.InitSDP()
        assert called[0], "InitSDP should have dispatched to _sInitSDP_sparse"

    def test_reduction_method_sparsity_triggers_decomposition(self):
        """reduction_method='sparsity' should also trigger the sparse path."""
        from Irene.relaxations import SDPRelaxations, RelaxationConfig

        x1, x2 = _sp.symbols('x1 x2')
        config = RelaxationConfig(
            reduction_method="sparsity",
            sparsity_block_sdp=False,  # explicit False -- but reduction_method should still trigger
            verbose_reduction=False,
        )
        rlx = SDPRelaxations([x1, x2], config=config)
        rlx.SetObjective(x1**2 + x2**2)
        rlx.MmntOrd = 2
        rlx.Parallel = False

        original_sparse = rlx._sInitSDP_sparse
        called = [False]

        def spy_sparse():
            called[0] = True
            return original_sparse()

        rlx._sInitSDP_sparse = spy_sparse
        rlx.InitSDP()
        assert called[0], "reduction_method='sparsity' should dispatch to sparse path"

    def test_sparsity_detection_does_not_change_sdp_dispatch(self):
        """Detection alone must not switch to the experimental block SDP."""
        from Irene.relaxations import SDPRelaxations, RelaxationConfig

        x1, x2 = _sp.symbols('x1 x2')
        config = RelaxationConfig(
            reduction_method="none",
            sparsity_detection=True,
            verbose_reduction=False,
        )
        rlx = SDPRelaxations([x1, x2], config=config)
        rlx.SetObjective(x1**2 + x2**2)
        rlx.MmntOrd = 2
        rlx.Parallel = False

        original_sparse = rlx._sInitSDP_sparse
        called = [False]

        def spy_sparse():
            called[0] = True
            return original_sparse()

        rlx._sInitSDP_sparse = spy_sparse
        rlx.InitSDP()
        assert not called[0], "detection alone must preserve monolithic SDP dispatch"

    def test_fallback_when_sparsity_module_unavailable(self):
        """Graceful degradation when detect_sparsity is None."""
        import Irene.relaxations as rlx_mod
        from Irene.relaxations import SDPRelaxations, RelaxationConfig

        original = rlx_mod.detect_sparsity
        rlx_mod.detect_sparsity = None
        try:
            x1, x2 = _sp.symbols('x1 x2')
            config = RelaxationConfig(
                reduction_method="none",
                sparsity_block_sdp=True,
                verbose_reduction=False,
            )
            rlx = SDPRelaxations([x1, x2], config=config)
            rlx.SetObjective(x1**2 + x2**2)
            rlx.MmntOrd = 2

            # Should fall back to sInitSDP without raising
            rlx._sInitSDP_sparse()
            assert rlx.f_min is not None
        finally:
            rlx_mod.detect_sparsity = original
