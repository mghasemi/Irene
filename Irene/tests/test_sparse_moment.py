"""Tests for the sparse moment method semigroup-ring setup (S51.0).

Covers: toric ideal computation, rho_A bound, A-graduation deg_A, and input
validation in ``Irene.sparse_moment``.
"""

import numpy as np
import pytest
from sympy import symbols, expand

from Irene.sparse_moment import (
    SparseMomentSDP, _moment_matrix_from_vector, solve_algorithm1,
    moment_indices, SparseBorderBasis, recover_all_sparse_roots,
    solve_sparse_real_roots)


class TestToricIdeal:
    """Toric ideal T_A = ker(psi_A) via elimination."""

    def test_r_x2_x3(self):
        # Paper Example 3.18: R[A] = R[x^2, x^3], T_A < z1^3 - z2^2>, rho_A = 3
        from Irene.sparse_moment import toric_ideal
        gens = toric_ideal([(2, 0), (3, 0)])
        assert len(gens) == 1
        from sympy import Poly
        p = Poly(expand(gens[0]), *tuple(symbols('z1:3')))
        # total degree of the generator must be exactly rho_A = 3
        assert max(sum(k) for k in p.as_dict().keys()) == 3
        # psi_A(g) must vanish: substitute z_i -> x^{alpha_i}
        x = symbols('x')
        assert expand(gens[0].subs({p.gens[0]: x**2, p.gens[1]: x**3})) == 0

    def test_full_polynomial_ring(self):
        # A = standard basis: monomials algebraically independent -> T_A < 0>
        from Irene.sparse_moment import toric_ideal
        assert toric_ideal([(1, 0), (0, 1)]) == []

    def test_veronese_square(self):
        # R[x^2, xy, y^2]: z1*z3 - z2^2, degree 2
        from Irene.sparse_moment import toric_ideal
        x, y = symbols('x y')
        gens = toric_ideal([(2, 0), (1, 1), (0, 2)])
        assert len(gens) == 1
        z = tuple(symbols('z1:4'))
        sub = dict(zip(z, (x**2, x * y, y**2)))
        assert expand(gens[0].subs(sub)) == 0

    def test_sparse_three_var(self):
        # R[x^2, xy, y^3]: expect single generator of degree 6
        from Irene.sparse_moment import toric_ideal
        x, y = symbols('x y')
        gens = toric_ideal([(2, 0), (1, 1), (0, 3)])
        assert len(gens) == 1
        z = tuple(symbols('z1:4'))
        sub = dict(zip(z, (x**2, x * y, y**3)))
        assert expand(gens[0].subs(sub)) == 0

    def test_generators_vanish_under_psi_A(self):
        # general sanity: every generator annihilates under the semigroup map
        from Irene.sparse_moment import toric_ideal
        A = [(1, 2), (3, 0), (0, 4)]
        x, y = symbols('x y')
        z = tuple(symbols('z1:%d' % (len(A) + 1)))
        gens = toric_ideal(A)
        sub = dict(zip(z, [x**a[0] * y**a[1] for a in A]))
        for g in gens:
            assert expand(g.subs(sub)) == 0


class TestToricSetup:

    def test_attributes(self):
        from Irene.sparse_moment import ToricSetup
        ts = ToricSetup.from_exponents([(2, 0), (3, 0)])
        assert ts.A == ((2, 0), (3, 0))
        assert ts.r == 2
        assert ts.ambient_dim == 2           # exponents live in N^2
        assert ts.rho_A == 3
        assert not ts.is_full_ring

    def test_full_ring_flag(self):
        from Irene.sparse_moment import ToricSetup
        ts = ToricSetup.from_exponents([(1, 0), (0, 1)])
        assert ts.G_A == ()
        assert ts.rho_A == 0
        assert ts.is_full_ring

    def test_dedup_and_order(self):
        from Irene.sparse_moment import ToricSetup
        ts = ToricSetup.from_exponents([(2, 0), (3, 0), (2, 0)])
        assert ts.A == ((2, 0), (3, 0))

    def test_veronese(self):
        from Irene.sparse_moment import ToricSetup
        ts = ToricSetup.from_exponents([(2, 0), (1, 1), (0, 2)])
        assert ts.rho_A == 2


class TestDegA:
    """The A-graduation deg_A(f) = min{s : f in R[A]^s}."""

    def test_x7_in_r_x2x3(self):
        from Irene.sparse_moment import deg_A
        x = symbols('x')
        assert deg_A(x**7, [(2, 0), (3, 0)]) == 3      # 7 = 2+2+3

    def test_x10_in_r_x2x3(self):
        from Irene.sparse_moment import deg_A
        x = symbols('x')
        assert deg_A(x**10, [(2, 0), (3, 0)]) == 4     # 10 = 3+3+2+2

    def test_x5(self):
        from Irene.sparse_moment import deg_A
        x = symbols('x')
        assert deg_A(x**5, [(2, 0), (3, 0)]) == 2      # 5 = 2+3

    def test_constant_and_zero(self):
        from Irene.sparse_moment import deg_A
        x = symbols('x')
        assert deg_A(5, [(1, 0)]) == 0
        assert deg_A(0, [(1, 0), (0, 1)]) == 0

    def test_full_ring_is_total_degree(self):
        from Irene.sparse_moment import deg_A
        x, y = symbols('x y')
        A = [(1, 0), (0, 1)]
        assert deg_A(x**2 * y**3 + x*y, A) == 5

    def test_subadditivity(self):
        from Irene.sparse_moment import deg_A
        x, y = symbols('x y')
        # R[x^2, xy]: monomials are x^(2l+k) y^k for l,k >= 0
        A = [(2, 0), (1, 1)]
        f = x**3 * y                    # (2,0)+(1,1): deg_A = 2
        g = x*y + x**2                  # both one summand: deg_A = 1
        assert deg_A(f, A) == 2
        assert deg_A(g, A) == 1
        fg = expand(f * g)              # x^4 y^2 + x^3 y -> deg_A = 3
        assert deg_A(fg, A) <= deg_A(f, A) + deg_A(g, A)
        assert deg_A(fg, A) == 3
        assert deg_A(f + g, A) <= max(deg_A(f, A), deg_A(g, A))

    def test_accepts_setup_object(self):
        from Irene.sparse_moment import ToricSetup, deg_A
        x = symbols('x')
        ts = ToricSetup.from_exponents([(2, 0), (3, 0)])
        assert deg_A(x**7, ts) == 3

    def test_outside_semigroup_raises(self):
        from Irene.sparse_moment import deg_A
        x, y = symbols('x y')
        with pytest.raises(ValueError):
            deg_A(y, [(2, 0), (3, 0)])                # y not in R[x^2,x^3]

    def test_too_many_symbols_raises(self):
        from Irene.sparse_moment import deg_A
        x, y, z = symbols('x y z')
        with pytest.raises(ValueError):
            deg_A(x * y * z, [(1, 0), (0, 1)])       # 3 symbols vs N^2 exponents

    def test_negative_exponent_raises(self):
        from Irene.sparse_moment import toric_ideal
        with pytest.raises(ValueError):
            toric_ideal([(1, -1)])

    def test_zero_vector_raises(self):
        from Irene.sparse_moment import ToricSetup
        with pytest.raises(ValueError):
            ToricSetup.from_exponents([(0, 0), (1, 0)])

    def test_ragged_dimensions_raise(self):
        from Irene.sparse_moment import toric_ideal
        with pytest.raises(ValueError):
            toric_ideal([(1, 0), (0, 1, 0)])


class TestMomentIndices:
    """Basis enumeration of R[A]^t."""

    def test_full_ring_counts(self):
        from Irene.sparse_moment import moment_indices
        x = symbols('x')
        monoms, index, levels = moment_indices([(1,)], 4)
        assert len(monoms) == 5                   # binom(1+4, 4) = 5
        assert [index[(i,)] for i in range(5)] == list(range(5))

    def test_sparse_r_x2x3(self):
        from Irene.sparse_moment import moment_indices
        monoms, index, levels = moment_indices([(2, 0), (3, 0)], 6)
        # R[x^2,x^3]_t=6: x^(k,0) for k in {0} U [2..18] -> 18 monomials (N^2 exponents)
        assert set(monoms) == {(0, 0)} | {(i, 0) for i in range(2, 19)}
        assert len(monoms) == 18
        # A-degree of x^(k,0): k=0 -> 0; k>=2 -> min number of 2s and 3s summing to k
        assert levels[(0, 0)] == 0 and levels[(4, 0)] == 2 and levels[(5, 0)] == 2

    def test_negative_t_raises(self):
        from Irene.sparse_moment import moment_indices
        with pytest.raises(ValueError):
            moment_indices([(1,)], -1)


class TestProlongations:
    """H_t construction (paper eq. 9)."""

    def test_single_gen_1d(self):
        from Irene.sparse_moment import moment_indices, prolongations
        x = symbols('x')
        monoms, index, levels = moment_indices([(1,)], 3)
        H = prolongations([x**2 - 1], [(1,)], 3, index)
        # deg_A(f)=2, so beta with level <= 1: {0, x} -> rows x^beta*(x^2-1),
        # plus the t-level boundary gives {x^2-1, x^3-x, x^4-x^2}? No -- t=3 keeps
        # only products of A-degree <= 3; beta levels {0,1} -> rows: x^2-1, x^3-x.
        # (level map for A=[(1,)]: level(k)=k, so deg<=1 gives beta in {0,(1,)}.)
        assert len(H) == 2

    def test_deduplication(self):
        from Irene.sparse_moment import moment_indices, prolongations
        x = symbols('x')
        monoms, index, levels = moment_indices([(1,)], 4)
        # deg_A(f)=2; beta with level <= 2: {0, x, x^2} -> distinct products
        # {x^2-1, x^3-x, x^4-x^2}; duplicate f copies collapse.
        H = prolongations([x**2 - 1, x**2 - 1], [(1,)], 4, index)
        assert len(H) == 3                        # both copies collapse to 3 distinct rows

    def test_degree_exceeds_t_raises(self):
        from Irene.sparse_moment import moment_indices, prolongations
        x = symbols('x')
        monoms, index, levels = moment_indices([(1,)], 3)
        with pytest.raises(ValueError):
            prolongations([x**4], [(1,)], 3, index)

    def test_sparse_2d(self):
        from Irene.sparse_moment import moment_indices, prolongations
        x, y = symbols('x y')
        A = [(2, 0), (1, 1)]                      # R[x^2, xy]
        f = x**3 * y - 1                          # (3,1) = (2,0)+(1,1): deg_A = 2
        monoms, index, levels = moment_indices(A, 4)
        H = prolongations([f], A, 4, index)
        # beta with deg_A(beta) <= 4-2=2: {(0,0),(1,1),(2,0),(2,2),(3,1),(4,0)} -> 6 rows
        assert len(H) == 6


class TestSparseMomentSDP:
    """K_t spectrahedron as SDP (paper eq. 10)."""

    def test_1d_three_real_roots(self):
        from Irene.sparse_moment import SparseMomentSDP
        x = symbols('x')
        f = x**3 - x                              # roots in {-1, 0, 1}
        sdp = SparseMomentSDP([f], [(1,)], t=4)
        assert sdp.n_monomials_t == 5             # R[x]_4 has 5 monomials
        res = sdp.solve()
        assert res.status == 'Optimal'
        lam = res.lambda_vec
        assert lam is not None
        zero = sdp.index[(0,)]
        assert abs(lam[zero] - 1.0) < 1e-6
        # moment consistency: Lambda(x^2) within convex hull of root squares [0, 1]
        v2 = lam[sdp.index[(2,)]]
        assert -1e-9 <= v2 <= 1 + 1e-9
        # prolongation residual vanishes
        idx = sdp.index
        hres = (lam[idx[(3,)]] - lam[idx[(1,)]])   # Lambda(x^3-x)
        assert abs(hres) < 1e-8

    def test_no_real_roots_infeasible(self):
        from Irene.sparse_moment import SparseMomentSDP
        x = symbols('x')
        sdp = SparseMomentSDP([x**2 + 1], [(1,)], t=2)
        res = sdp.solve()
        assert res.status == 'Infeasible'         # Stengle: 1 is SOS mod I

    def test_paper_example_3_16(self):
        from Irene.sparse_moment import SparseMomentSDP
        x, y = symbols('x y')
        A = [(2, 1), (1, 2)]
        a1, a2 = A
        f1 = x**(2*a1[0]+a2[0]) * y**(2*a1[1]+a2[1])              # x^(2 alpha1 + alpha2)
        f2 = f1 - x**a2[0]*y**a2[1] + x**a1[0]*y**a1[1] + 1      # x^(2alpha1+alpha2)-x^alpha2+x^alpha1+1
        f3 = x**(2*a1[0])*y**(2*a1[1]) - x**a1[0]*y**a1[1]
        f4 = x**(2*a2[0])*y**(2*a2[1]) - x**a2[0]*y**a2[1] \
             + x**(2*a1[0]+4*a2[0])*y**(2*a1[1]+4*a2[1])
        sdp = SparseMomentSDP([f1, f2, f3, f4], A, t=6)
        res = sdp.solve()
        assert res.status == 'Optimal'            # K_6 non-empty: system solvable in R^2
        lam = res.lambda_vec
        assert lam is not None
        zero = sdp.index[(0, 0)]
        assert abs(lam[zero] - 1.0) < 1e-6

    def test_custom_objective_length_mismatch(self):
        from Irene.sparse_moment import SparseMomentSDP
        x = symbols('x')
        with pytest.raises(ValueError):
            SparseMomentSDP([x**2 - 1], [(1,)], t=2, objective=[0.5])   # wrong length

    def test_t_below_degree_raises(self):
        from Irene.sparse_moment import SparseMomentSDP
        x = symbols('x')
        with pytest.raises(ValueError):
            SparseMomentSDP([x**3], [(1,)], t=2)  # deg_A(x^3)=3 > t=2

    def test_str(self):
        from Irene.sparse_moment import SparseMomentSDP
        x = symbols('x')
        sdp = SparseMomentSDP([x**2 - 1], [(1,)], t=4)
        assert 'SparseMomentSDP' in str(sdp)



# ---------------------------------------------------------------------------
# S51.2: Theorem 3.27 rank conditions and the Algorithm-1 termination loop
# ---------------------------------------------------------------------------

class TestNumericalRank:
    """_numerical_rank mirrors SDPRelaxations.NumericalRank (absolute threshold)."""

    def test_diagonal(self):
        from Irene.sparse_moment import _numerical_rank
        M = np.diag([1.0, 2.0, 1e-9])
        assert _numerical_rank(M, 1e-6) == 2

    def test_empty(self):
        from Irene.sparse_moment import _numerical_rank
        assert _numerical_rank(np.zeros((0, 0)), 1e-6) == 0

    def test_nonsymmetric_input_symmetrized(self):
        from Irene.sparse_moment import _numerical_rank
        M = np.array([[1.0, 3.7], [2.9, 1.5]])
        assert _numerical_rank(M, 1e-6) == 2

    def test_matches_numpy_matrix_rank(self):
        from Irene.sparse_moment import _numerical_rank
        rng = np.random.default_rng(0)
        A = rng.standard_normal((7, 5))
        M = A @ A.T                                  # PSD, rank 5 with probability 1
        assert _numerical_rank(M, 1e-6) == 5


class TestMomentMatrixFromVector:
    """_moment_matrix_from_vector assembles M_s(Lambda) from the moment vector."""

    def test_univariate_psd_symmetric(self):
        import sympy as sp
        x = sp.Symbol('x')
        sdp = SparseMomentSDP([x**3 - x], [(1,)], 6)
        lam = np.asarray(sdp.solve().lambda_vec, float)
        M2 = _moment_matrix_from_vector(lam, sdp.monoms_t, sdp.levels, sdp.index, 2)
        assert M2.shape == (3, 3)
        assert np.allclose(M2, M2.T, atol=1e-10)
        ev = np.linalg.eigvalsh(0.5 * (M2 + M2.T))
        assert np.min(ev) > -1e-8                      # PSD
        assert abs(M2[0, 0] - 1.0) < 1e-9              # Lambda(x^0) = 1

    def test_matches_SparseMomentResult_Ms(self):
        import sympy as sp
        x = sp.Symbol('x')
        sdp = SparseMomentSDP([x**3 - x], [(1,)], 6)
        res = sdp.solve()
        lam = np.asarray(res.lambda_vec, float)
        M3 = _moment_matrix_from_vector(lam, sdp.monoms_t, sdp.levels, sdp.index, 3)
        assert np.allclose(M3, np.asarray(res.M_s), atol=1e-9)

    def test_level_cutoff(self):
        import sympy as sp
        x = sp.Symbol('x')
        sdp = SparseMomentSDP([x**3 - x], [(1,)], 6)
        lam = np.asarray(sdp.solve().lambda_vec, float)
        M0 = _moment_matrix_from_vector(lam, sdp.monoms_t, sdp.levels, sdp.index, 0)
        assert M0.shape == (1, 1) and abs(M0[0, 0] - 1.0) < 1e-9


class TestColumnBasisPositions:
    """Greedy monomial column basis of a moment matrix."""

    def test_full_rank(self):
        from Irene.sparse_moment import _column_basis_positions
        assert _column_basis_positions(np.eye(3), 1e-6) == [0, 1, 2]

    def test_rank_one(self):
        from Irene.sparse_moment import _column_basis_positions
        v = np.array([1.0, 2.0, 3.0])
        assert _column_basis_positions(np.outer(v, v), 1e-6) == [0]

    def test_empty(self):
        from Irene.sparse_moment import _column_basis_positions
        assert _column_basis_positions(np.zeros((0, 0)), 1e-6) == []


class TestTheorem327:
    """test_theorem_327 rank-condition convergence test."""

    @staticmethod
    def _x3x_t6():
        import sympy as sp
        x = sp.Symbol('x')
        sdp = SparseMomentSDP([x**3 - x], [(1,)], 6)
        res = sdp.solve()
        assert res.status == 'Optimal' and res.lambda_vec is not None
        return x, sdp, res

    def test_flat_convergence(self):
        from Irene.sparse_moment import test_theorem_327
        x, sdp, res = self._x3x_t6()
        lam = np.asarray(res.lambda_vec, float)
        rt = test_theorem_327(lam, [x**3 - x], [(1,)], 6,
                              sdp.index, sdp.levels, sdp.monoms_t)
        assert (rt.converged, rt.condition, rt.s_star, rt.n_real) == \
            (True, 'flat', 3, 3)

    def test_rank_profile(self):
        from Irene.sparse_moment import test_theorem_327
        x, sdp, res = self._x3x_t6()
        lam = np.asarray(res.lambda_vec, float)
        rt = test_theorem_327(lam, [x**3 - x], [(1,)], 6,
                              sdp.index, sdp.levels, sdp.monoms_t)
        # doctest-verified profile: flat extension at s* = 3
        assert dict(rt.ranks) == {0: 1, 1: 2, 2: 3, 3: 3}

    def test_D_d_rho_fields(self):
        from Irene.sparse_moment import test_theorem_327
        x, sdp, res = self._x3x_t6()
        lam = np.asarray(res.lambda_vec, float)
        rt = test_theorem_327(lam, [x**3 - x], [(1,)], 6,
                              sdp.index, sdp.levels, sdp.monoms_t)
        assert (rt.D, rt.d, rt.rho_A) == (3, 2, 0)

    def test_admissible_range_empty_not_converged(self):
        # Hand-crafted interior vector for <x^2 - 1> at t=3: M_1 = [[1,.5],[.5,1]]
        # has rank 2 != rank(M_0)=1; case (i) needs s >= max{D, rho_A} = 2 > floor(t/2).
        import sympy as sp
        x = sp.Symbol('x')
        from Irene.sparse_moment import SparseMomentSDP, test_theorem_327
        sdp = SparseMomentSDP([x**2 - 1], [(1,)], 3)
        lam = np.array([1.0, 0.5, 1.0, 0.5])           # in K_3 (H_3: m3=m1, m2=1)
        rt = test_theorem_327(lam, [x**2 - 1], [(1,)], 3,
                              sdp.index, sdp.levels, sdp.monoms_t)
        assert rt.converged is False
        assert dict(rt.ranks) == {0: 1, 1: 2}

    def test_zero_ideal_handcrafted(self):
        # Delta-at-zero moment vector (boundary of K_4; fine -- this unit-tests the
        # rank logic, not the solver). D=0 -> d=1; plateau at s*=1.
        import sympy as sp
        from Irene.sparse_moment import SparseMomentSDP, test_theorem_327
        x = sp.Symbol('x')
        sdp = SparseMomentSDP([sp.sympify(0.0)], [(1,)], 4)
        lam = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        rt = test_theorem_327(lam, [sp.sympify(0.0)], [(1,)], 4,
                              sdp.index, sdp.levels, sdp.monoms_t)
        assert (rt.converged, rt.condition, rt.s_star, rt.n_real) == \
            (True, 'flat', 1, 1)
        assert (rt.D, rt.d) == (0, 1)


class TestSolveAlgorithm1:
    """solve_algorithm1: the K_t SDP + rank-stabilization termination loop."""

    def test_three_real_roots(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        assert r.converged and not r.no_real_roots
        # smallest convergent truncation is t=6 (t0 = max{4, 2D} = 6)
        assert r.t == 6
        rt = r.rank_test
        assert (rt.condition, rt.s_star, rt.n_real) == ('flat', 3, 3)

    def test_kernel_generator_is_the_polynomial(self):
        # kernel of M_{s*} spans J: single generator proportional to x^3 - x,
        # compared up to a nonzero scalar (sign/normalization-robust)
        import sympy as sp
        from sympy import Poly
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        assert len(r.kernel_polys) == 1
        p = Poly(sp.expand(r.kernel_polys[0]), x).as_expr()
        q = sp.simplify(p / Poly(p, x).LC())
        assert sp.simplify(q - (x**3 - x)) == 0

    def test_kernel_polys_clean_strings(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**2 - 1], A=[(1,)])
        assert sorted(str(p) for p in r.kernel_polys) == ['x**2 - 1']

    def test_two_real_roots_and_quotient_basis(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**2 - 1], A=[(1,)])
        assert r.converged
        rt = r.rank_test
        assert (rt.condition, rt.s_star, rt.n_real) == ('flat', 2, 2)
        # Corollary 3.19: |B| = n_real for generic Lambda; 'connected-to-one'
        assert len(r.quotient_basis) == 2
        assert r.quotient_basis[0] == (0,)

    def test_quotient_border_basis_x3x(self):
        # B = {1, x, x^2}: the connected-to-one border basis of R[x]/<x^3-x>
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        assert sorted(r.quotient_basis) == [(0,), (1,), (2,)]

    def test_no_real_roots_stengle(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**2 + 1], A=[(1,)])
        assert not r.converged and r.no_real_roots
        assert r.infeasible_at_t == r.t
        assert 'no real toric root' in str(r)

    def test_moment_matrix_flat_psd(self):
        # M_{s*}(Lambda) is PSD; null(M_s*) has dim = |R[A]^{s*}| - rank(M_s*)
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        assert r.rank_test is not None and r.rank_test.s_star is not None \
            and r.rank_test.n_real is not None
        sdp = SparseMomentSDP([x**3 - x], [(1,)], r.t)
        dim_s = sum(1 for e in sdp.monoms_t if sdp.levels[e] <= r.rank_test.s_star)
        Ms = 0.5 * (r.moment_matrix + r.moment_matrix.T)
        ev = np.linalg.eigvalsh(Ms)
        assert np.min(ev) > -1e-8
        nullity = int(np.sum(ev < 1e-6))
        assert nullity == dim_s - r.rank_test.n_real      # rank(M_s*) = n_real

    def test_lambda_normalized(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        sdp = SparseMomentSDP([x**3 - x], [(1,)], r.t)
        assert abs(r.lambda_vec[sdp.index[(0,)]] - 1.0) < 1e-6

    def test_A_inference_single_variable(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**2 - 1])               # A=None -> [(1,)]
        assert r.converged and (r.rank_test.s_star, r.rank_test.n_real) == (2, 2)

    def test_A_inference_multivariable_raises(self):
        x, y = symbols('x y')
        with pytest.raises(ValueError, match='pass it explicitly'):
            solve_algorithm1([x * y + 1])              # 2 free symbols

    def test_t0_walk_until_convergence(self):
        # starting below the smallest convergent truncation: t walks up (step 4)
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)], t0=4)
        assert r.converged and r.t == 6

    def test_max_iterations_guard(self):
        import sympy as sp
        x = sp.Symbol('x')
        with pytest.raises(RuntimeError, match='did not converge'):
            solve_algorithm1([x**3 - x], A=[(1,)], t0=4, max_iterations=2)

    def test_sparse_semigroup_radical(self):
        # R[x^2, x^3]: f = (x^2-1)(x^2-4), real roots +-1, +-2 -> n_real = 4;
        # verified: converges at t=7, s*=3 (flat), |B| = 4.
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**4 - 5 * x**2 + 4], A=[(2, 0), (3, 0)])
        assert not r.no_real_roots and r.converged
        rt = r.rank_test
        assert (rt.condition, rt.s_star, rt.n_real) == ('flat', 3, 4)
        assert len(r.quotient_basis) == 4

    def test_result_str_not_converged(self):
        from Irene.sparse_moment import Algorithm1Result
        out = Algorithm1Result(iterations=2, t=5)
        assert 'not converged' in str(out)
        out2 = Algorithm1Result(no_real_roots=True, infeasible_at_t=4, t=4)
        assert 'no real toric root' in str(out2)


# ---------------------------------------------------------------------------
# S51.3: Sparse border basis (connected-to-one, normal forms, multiplication tables)
# S51.4: Sparse Stickelberger eigenvalue recovery + BWE
# ---------------------------------------------------------------------------

class TestSparseBorderBasis:
    """S51.3: SparseBorderBasis construction and verification."""

    def test_univariate_x3x(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        monoms, index, levels = moment_indices([(1,)], r.t)
        bb = SparseBorderBasis(r.lambda_vec, list(r.quotient_basis), [(1,)],
                               r.t, int(r.rank_test.s_star), monoms, index, levels)
        assert bb.r_ == 3
        assert tuple(bb.basis) == ((0,), (1,), (2,))
        assert bb.connected_to_one is True

    def test_univariate_tables(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        monoms, index, levels = moment_indices([(1,)], r.t)
        bb = SparseBorderBasis(r.lambda_vec, list(r.quotient_basis), [(1,)],
                               r.t, int(r.rank_test.s_star), monoms, index, levels)
        T1 = bb.tables[0]
        # x * 1 = x -> col 0: [0,1,0]^T
        assert np.allclose(T1[:, 0], [0, 1, 0])
        # x * x = x^2 -> col 1: [0,0,1]^T
        assert np.allclose(T1[:, 1], [0, 0, 1])
        # x * x^2 = x^3 = x (mod x^3-x) -> col 2: [0,1,0]^T
        assert np.allclose(T1[:, 2], [0, 1, 0])

    def test_sparse_r_x2x3(self):
        import sympy as sp
        x = sp.Symbol('x')
        fB = (x**2 - 1)*(x**2 - 4)
        r = solve_algorithm1([sp.expand(fB)], A=[(2, 0), (3, 0)])
        monoms, index, levels = moment_indices([(2, 0), (3, 0)], r.t)
        bb = SparseBorderBasis(r.lambda_vec, list(r.quotient_basis),
                               [(2, 0), (3, 0)], r.t, int(r.rank_test.s_star),
                               monoms, index, levels)
        assert bb.r_ == 4
        # Canonical connected-to-one border basis of R[x^2,x^3]/J for roots
        # {+-1, +-2}: {1, x^2, x^3, x^5}.  (x^4 = (x^2)^2 = 5x^2 - 4 on the
        # roots, so x^4 is a combination of {1, x^2} and is NOT an independent
        # column; x^5 = x^2 x^3 is independent.)  Remark 3.28 selects this
        # greedily in increasing deg_A order; Def A.2 reachability holds.
        assert tuple(bb.basis) == ((0, 0), (2, 0), (3, 0), (5, 0))
        assert bb.connected_to_one is True

    def test_sparse_toric_relation(self):
        """Def A.3: z1^3 - z2^2 = 0 on the quotient for R[x^2,x^3]."""
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([sp.expand((x**2-1)*(x**2-4))], A=[(2, 0), (3, 0)])
        monoms, index, levels = moment_indices([(2, 0), (3, 0)], r.t)
        bb = SparseBorderBasis(r.lambda_vec, list(r.quotient_basis),
                               [(2, 0), (3, 0)], r.t, int(r.rank_test.s_star),
                               monoms, index, levels)
        # G_A generator for R[x^2,x^3] is z1^3 - z2^2 (degree rho_A = 3).
        # check_relations evaluates g(T_1, T_2) and returns ||g||_2.
        residual = bb.check_relations()
        assert residual < 1e-6

    def test_normal_form_basis_elements(self):
        """Normal form of a basis element is the standard unit vector."""
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        monoms, index, levels = moment_indices([(1,)], r.t)
        bb = SparseBorderBasis(r.lambda_vec, list(r.quotient_basis), [(1,)],
                               r.t, int(r.rank_test.s_star), monoms, index, levels)
        for k, b in enumerate(bb.basis):
            d = bb.normal_form(b)
            assert abs(d[index[b]] - 1.0) < 1e-12

    def test_normal_form_nonbasis(self):
        """x^3 has normal form x (mod x^3-x), so nf(x^3) should give [0,1,0]."""
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        monoms, index, levels = moment_indices([(1,)], r.t)
        bb = SparseBorderBasis(r.lambda_vec, list(r.quotient_basis), [(1,)],
                               r.t, int(r.rank_test.s_star), monoms, index, levels)
        # x^3 is not in B={0,1,2}, so its normal form must be computed via lstsq
        d = bb.normal_form((3,))
        assert abs(d[index[(1,)]] - 1.0) < 1e-6     # coefficient of x should be ~1

    def test_basis_must_start_with_one(self):
        monoms, index, levels = moment_indices([(1,)], 4)
        with pytest.raises(ValueError, match='constant'):
            SparseBorderBasis(np.ones(5), [(1,), (0,)], [(1,)], 4, 2,
                              monoms, index, levels)

    def test_repr(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        monoms, index, levels = moment_indices([(1,)], r.t)
        bb = SparseBorderBasis(r.lambda_vec, list(r.quotient_basis), [(1,)],
                               r.t, int(r.rank_test.s_star), monoms, index, levels)
        s = repr(bb)
        assert 'SparseBorderBasis' in s


class TestRecoverAllSparseRoots:
    """S51.4: Stickelberger eigenvalue recovery."""

    def test_univariate_x3x(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        out = recover_all_sparse_roots(r, [(1,)], [x**3 - x])
        assert len(out.roots) == 3
        zvals = sorted(v['z1'] for v in out.roots)
        np.testing.assert_allclose(zvals, [-1.0, 0.0, 1.0], atol=1e-6)

    def test_univariate_bwe(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        out = recover_all_sparse_roots(r, [(1,)], [x**3 - x])
        # BWE should be very small for exact roots (f vanishes)
        assert all(b < 1e-6 for b in out.bwe)

    def test_univariate_x2m1(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**2 - 1], A=[(1,)])
        out = recover_all_sparse_roots(r, [(1,)], [x**2 - 1])
        assert len(out.roots) == 2
        zvals = sorted(v['z1'] for v in out.roots)
        np.testing.assert_allclose(zvals, [-1.0, 1.0], atol=1e-6)

    def test_sparse_r_x2x3_roots(self):
        """Case B: R[x^2,x^3], f=(x^2-1)(x^2-4), roots at x = -2,-1,1,2."""
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([sp.expand((x**2-1)*(x**2-4))], A=[(2, 0), (3, 0)])
        out = recover_all_sparse_roots(r, [(2, 0), (3, 0)],
                                       [sp.expand((x**2-1)*(x**2-4))])
        assert len(out.roots) == 4
        # z1 = x^2 values: {4, 1, 1, 4}; z2 = x^3 values: {-8,-1,1,8}
        z1vals = sorted(v['z1'] for v in out.roots)
        np.testing.assert_allclose(z1vals, [1.0, 1.0, 4.0, 4.0], atol=1e-6)

    def test_kernel_polys_carried(self):
        import sympy as sp
        x = sp.Symbol('x')
        r = solve_algorithm1([x**3 - x], A=[(1,)])
        out = recover_all_sparse_roots(r, [(1,)])
        assert len(out.kernel_polys) >= 1

    def test_not_converged_raises(self):
        from Irene.sparse_moment import Algorithm1Result
        bad = Algorithm1Result(iterations=2, t=5)   # not converged
        with pytest.raises(ValueError, match='CONVERGED'):
            recover_all_sparse_roots(bad, [(1,)])

    def test_result_str(self):
        from Irene.sparse_moment import SparseRootsResult
        out = SparseRootsResult()
        assert 'no roots' in str(out)


class TestSolveSparseRealRoots:
    """Top-level pipeline: Algorithm 1 + border basis + Stickelberger."""

    def test_univariate_x3x(self):
        import sympy as sp
        x = sp.Symbol('x')
        out = solve_sparse_real_roots([x**3 - x], A=[(1,)])
        assert len(out.roots) == 3
        zvals = sorted(v['z1'] for v in out.roots)
        np.testing.assert_allclose(zvals, [-1.0, 0.0, 1.0], atol=1e-6)

    def test_no_real_roots_empty(self):
        import sympy as sp
        x = sp.Symbol('x')
        out = solve_sparse_real_roots([x**2 + 1], A=[(1,)])
        assert len(out.roots) == 0   # no real roots -> empty result

    def test_A_inference_raises_multivar(self):
        import sympy as sp
        x, y = sp.symbols('x y')
        with pytest.raises(ValueError, match='pass it explicitly'):
            solve_sparse_real_roots([x*y + 1])   # 2 variables, A=None

    def test_sparse_r_x2x3_pipeline(self):
        """Full pipeline on R[x^2,x^3] recovers all 4 roots."""
        import sympy as sp
        x = sp.Symbol('x')
        out = solve_sparse_real_roots([sp.expand((x**2-1)*(x**2-4))],
                                      A=[(2, 0), (3, 0)])
        assert len(out.roots) == 4
