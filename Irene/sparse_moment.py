"""Sparse moment method for real roots of sparse polynomial systems.

Implements the algorithm of Baumbach & Bender, "The Moment Method for
Computing Real Points of Sparse Polynomial Systems" (arXiv:2609.33313):
polynomial systems solved on the semigroup ring R[A] = R[x^{alpha_1}, ...,
x^{alpha_r}] by exploiting the toric ideal T_A to keep moment matrices small,
with a termination guarantee via sparse flat-extension rank conditions
(Theorem 3.27 of the paper).

This module currently provides S51.0 -- the semigroup-ring setup:

* ``ToricSetup``      -- exponent set A, ambient dimension, toric generators
* ``toric_ideal(A)``  -- a Groebner basis G_A of T_A = ker(psi_A) in R[z]
* ``rho_A``           -- max total degree of the generators (0 if T_A < 0>)
* ``deg_A(f, A)``     -- the A-graduation: min{s : f in R[A]^s}

LATER SUBTASKS (Vikunja #51): S51.2 Theorem 3.27 rank conditions and termination loop,
S51.3 sparse border basis, S51.4 sparse Stickelberger eigenvalue recovery, S51.5
Algorithm 2 (randomized single root).

S51.1 provides: ``semigroup_level_map``, ``moment_indices`` (R[A]^t enumeration),
``prolongations`` (H_t, eq. (9)) and ``SparseMomentSDP`` -- the K_t spectrahedron
of eq. (10) as a standard-form SDP on CvxpySDPSolver.

NOTATION (paper sections referenced)
------------------------------------
A = {alpha_1, ..., alpha_r} subset N^m              finite set of exponents
R[A]^s = < x^{sum_{i<=s} alpha_i} >                 s-truncation (Def 2.4)
psi_A : R[z] -> R[x], z_i |-> x^{alpha_i}           the semigroup map  (1)
T_A := ker(psi_A), G_A a generating set,
rho_A := max{deg(g) : g in G_A}, rho_A = 0 if T_A < 0>   (3)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Toric ideal and A-graduation
# ---------------------------------------------------------------------------

def _validate_exponents(A: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    """Validate the exponent set and return it as a list of int tuples.

    Rules (paper Section 2): each alpha_i in N^m with m >= 1; no zero vector
    (it would force psi_A(0) = 1 and make R[A] degenerate); duplicates removed,
    order preserved.
    """
    A_int: List[Tuple[int, ...]] = []
    for a in A:
        if len(a) == 0:
            raise ValueError("exponent vector must be non-empty")
        t = tuple(int(e) for e in a)
        if any(e < 0 for e in t):
            raise ValueError(
                "semigroup-ring exponents must lie in N^m (got %s); "
                "Laurent monomials are out of scope" % (t,))
        if all(e == 0 for e in t):
            raise ValueError("exponent vector 0 is not allowed in A")
        A_int.append(t)
    m = len(A_int[0])
    for t in A_int:
        if len(t) != m:
            raise ValueError("all exponent vectors must have the same length "
                             "(ambient dimension)")
    # dedupe, preserving first-seen order
    seen = set()
    out = []
    for t in A_int:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def toric_ideal(A: Sequence[Sequence[int]]) -> List[sp.Expr]:
    """Compute a Groebner basis G_A of the toric ideal T_A = ker(psi_A).

    Uses standard elimination (Sturmfels, "Grobner Bases and Convex Polytopes",
    Theorem 2.3): introduce u_1..u_m for x_1..x_m and z_1..z_r for the r
    monomials x^{alpha_i}, then eliminate u from

        < z_i - prod_j u_j^{alpha_{i,j}} : i = 1..r >   (lex, u first)

    The ideal G_A in R[z] alone is exactly T_A.

    Parameters
    ----------
    A : sequence of exponent vectors in N^m

    Returns
    -------
    list of SymPy expressions in z_1..z_r generating T_A; ``[]`` iff the
    monomials are algebraically independent (T_A = <0>).
    """
    A_int = _validate_exponents(A)
    r, m = len(A_int), len(A_int[0])
    z = sp.symbols('z1:%d' % (r + 1))
    u = sp.symbols('u1:%d' % (m + 1))
    polys: List[sp.Expr] = []
    for i, alpha in enumerate(A_int):
        mono = sp.Integer(1)
        for j, e in enumerate(alpha):
            if e:
                mono *= u[j] ** e
        polys.append(z[i] - mono)
    # lex with the eliminated variables listed FIRST: SymPy's lexicographic
    # order eliminates gens[0], so G ∩ R[z] is read off as the part of G in z.
    G = sp.groebner(polys, *tuple(u), *tuple(z), order='lex', domain=sp.QQ)
    return [g.as_expr() for g in G.polys
            if not (set(g.free_symbols) & set(u))]


def _total_degree(expr: sp.Expr, z: Sequence[sp.Symbol]) -> int:
    p = sp.Poly(sp.expand(expr), *z)
    return max(sum(k) for k in p.as_dict().keys())


@dataclass(frozen=True)
class ToricSetup:
    """Semigroup-ring data for the sparse moment method.

    Attributes
    ----------
    A : tuple of exponent vectors (validated, deduplicated)
    r : number of generators = len(A); ambient dimension m = len(A[0])
    z : the generator symbols z_1..z_r of R[z] ~= R[A] (isomorphic)
    G_A : Groebner basis of the toric ideal T_A in R[z] (empty if none)
    rho_A : max total degree over G_A, or 0 when T_A = <0>  (paper eq. 3)
    """
    A: Tuple[Tuple[int, ...], ...]
    r: int
    z: Tuple[sp.Symbol, ...]
    G_A: Tuple[sp.Expr, ...]
    rho_A: int

    @property
    def ambient_dim(self) -> int:
        return len(self.A[0])

    @property
    def is_full_ring(self) -> bool:
        """True iff A is a monomial basis of the full polynomial ring
        (T_A < 0>); then rho_A = 0 and no toric reduction applies."""
        return self.rho_A == 0

    @classmethod
    def from_exponents(cls, A: Sequence[Sequence[int]]) -> "ToricSetup":
        """Build the setup for an exponent set A subset of N^m.

        Example
        -------
        >>> import sympy as sp
        >>> ts = ToricSetup.from_exponents([(2, 0), (3, 0)])   # R[x^2, x^3]
        >>> ts.rho_A
        3
        >>> [sp.simplify(g) for g in ts.G_A]
        [z1**3 - z2**2]
        """
        A_int = tuple(_validate_exponents(A))
        r = len(A_int)
        z = sp.symbols('z1:%d' % (r + 1))
        G_A = tuple(toric_ideal(A_int))
        if not G_A:
            rho_A = 0
        else:
            rho_A = max(_total_degree(g, z) for g in G_A)
        return cls(A=A_int, r=r, z=tuple(z), G_A=G_A, rho_A=rho_A)


def deg_A(f: sp.Expr, A: Sequence[Sequence[int]]) -> int:
    """A-graduation of f in R[A] (paper Section 2):

        deg_A(f) = min{ s in N : f in R[A]^s },

    where R[A]^s is spanned by the monomials x^{alpha_{i_1} + ... + alpha_{i_s}}
    with at most s summands. Subadditive: deg_A(fg) <= deg_A(f)+deg_A(g),
    deg_A(f+g) <= max{deg_A(f), deg_A(g)}.

    Parameters
    ----------
    f : SymPy polynomial (or 0); its free symbols are taken as the ambient
        variables x_1..x_m in sorted-name order, so len(free_symbols) must
        equal the ambient dimension of A. All monomials must lie in R[A].
    A : exponent set; may also be a ``ToricSetup``.

    Raises
    ------
    ValueError if f contains monomials outside R[A] or its number of free
    symbols does not match len(A[0]).

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> deg_A(x**7, [(2, 0), (3, 0)])   # x^7 in R[x^2, x^3]
    3
    """
    if isinstance(A, ToricSetup):
        A_int: Tuple[Tuple[int, ...], ...] = A.A
    else:
        A_int = tuple(_validate_exponents(A))

    f = sp.expand(sp.sympify(f))
    m = len(A_int[0])
    free = sorted(f.free_symbols, key=lambda s: str(s))
    if f.is_zero or not free:
        return 0                            # constants live in R[A]^0 < 1 >
    if len(free) > m:
        raise ValueError(
            "polynomial has %d free symbols but A is a set of %d-dimensional "
            "exponents" % (len(free), m))
    p = sp.Poly(f, *free)
    # Full exponent vectors in N^m: coordinates not appearing in f carry 0.
    monoms = set()
    for monom in p.monoms():
        vec = [0] * m
        for j, e in enumerate(monom):
            if e:
                vec[j] = e
        monoms.add(tuple(vec))
    # Every summand a in A has |a| >= 1, so any x^beta = a_{i_1}+...+a_{i_s}
    # needs s <= |beta|. Hence it suffices to build levels up to the maximal
    # total degree of f. level[s] = exponents reachable with exactly s summands;
    # stop only when the whole semigroup below the bound is exhausted, then read
    # off each monomial's minimal summand count (they may differ).
    max_tot = max(sum(k) for k in monoms)
    level: set = {tuple([0] * m)}              # exponents reachable by <= s summands
    while True:
        new = set()
        for b in level:
            for a in A_int:
                nb = tuple(e + ai for e, ai in zip(b, a))
                if sum(nb) <= max_tot:
                    new.add(nb)
        added = new - level
        if not added:
            break                              # semigroup exhausted below bound
        level |= added
    missing = sorted(monoms - level)
    if missing:
        raise ValueError("f contains monomials outside the semigroup ring "
                         "R[A]: %s" % (missing,))
    return max(_min_summands(t, A_int) for t in monoms)


def _as_exponents(A) -> Tuple[Tuple[int, ...], ...]:
    """Normalize an exponent set or ``ToricSetup`` to a tuple of int tuples."""
    if isinstance(A, ToricSetup):
        return A.A
    return tuple(_validate_exponents(A))


def semigroup_level_map(A: Sequence[Sequence[int]], max_level: int) -> dict:
    r"""Minimal summand count for every exponent reachable within ``max_level``.

    Returns the map  :math:`\beta \mapsto d_A(\beta)` for all
    :math:`\beta \in \bigcup_{s \le max\_level} A^s`, where
    :math:`d_A(x^\beta) = min\{ s : x^\beta \in R[A]^s \}` is the minimal
    number of summands of :math:`A` adding up to :math:`\beta`.  BFS over
    the exponent lattice with steps in ``A``; exponents whose total degree
    already exceeds :math:`max\_level \cdot max|a_i|` are pruned (all
    relevant paths stay below that bound).

    Parameters
    ----------
    A : sequence of exponent vectors in N^m, or a ``ToricSetup``.
    max_level : int >= 0; the largest A-degree to enumerate.
    """
    if max_level < 0:
        raise ValueError("max_level must be non-negative")
    A_int = _as_exponents(A)
    zero = tuple([0] * len(A_int[0]))
    dist = {zero: 0}
    frontier = [zero]
    for lvl in range(1, max_level + 1):
        nxt = []
        for p in frontier:
            pt = sum(p)
            for a in A_int:
                q = tuple(e + ai for e, ai in zip(p, a))
                if pt + sum(a) > lvl * _max_a_total(A_int):
                    continue                  # beyond any level-<=lvl exponent
                if q not in dist:
                    dist[q] = lvl
                    nxt.append(q)
        frontier = nxt
    return dist


def _max_a_total(A_int: Tuple[Tuple[int, ...], ...]) -> int:
    return max(sum(a) for a in A_int)


def moment_indices(A: Sequence[Sequence[int]], t: int):
    r"""Enumerate the truncated ring :math:`R[A]^t`.

    Returns ``(monoms, index, levels)`` where ``monoms`` is the sorted list of
    exponent tuples in R[A]^t (by A-degree, then lexicographic), ``index``
    maps each exponent to its position in that list, and ``levels`` gives
    ``deg_A(x^beta)`` for every enumerated beta.  The length of ``monoms`` is
    the number of SDP variables in K_t; the sub-list with level <= floor(t/2)
    indexes the rows/columns of the moment matrix M_{floor(t/2)}.

    Example
    -------
    >>> from Irene.sparse_moment import moment_indices
    >>> monoms, index, levels = moment_indices([(1,)], 4)   # R[x], t = 4
    >>> len(monoms), index[(0,)]
    (5, 0)
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    A_int = _as_exponents(A)
    levels = semigroup_level_map(A_int, t)
    monoms = sorted(levels, key=lambda e: (levels[e], e))
    index = {e: i for i, e in enumerate(monoms)}
    return monoms, index, levels


def _min_summands(target: Tuple[int, ...], A_int: Tuple[Tuple[int, ...], ...]) -> int:
    """Minimal s with target = a_{i_1}+...+a_{i_s}, a_i in A (BFS over sums)."""
    if all(e == 0 for e in target):
        return 0
    zero = tuple([0] * len(target))
    seen = {target}
    frontier = [target]
    steps = 1                                   # one subtraction = one summand
    while frontier:
        next_frontier = []
        for t in frontier:
            for a in A_int:
                if any(e < ai for e, ai in zip(t, a)):
                    continue                  # would go negative
                d = tuple(e - ai for e, ai in zip(t, a))
                if d == zero:
                    return steps
                if d not in seen:
                    seen.add(d)
                    next_frontier.append(d)
        frontier = next_frontier
        steps += 1
    raise ValueError("target is not a sum of elements of A")


# ---------------------------------------------------------------------------
# S51.1: sparse truncated moment matrix and K_t spectrahedron (paper eqs. 9-10)
# ---------------------------------------------------------------------------

def _poly_to_dense(f, m: int):
    """Dense coefficient dict {exponent-tuple-in-N^m: float} for f in R[x_1..x_m]."""
    p = sp.Poly(sp.expand(sp.sympify(f)), *sp.symbols('x1:%d' % (m + 1))) if m else None
    if p is None:
        raise ValueError("ambient dimension must be positive")
    # Poly.monoms() are in the declared generators x_1..x_m -- exactly N^m.
    return {tuple(e): float(c) for e, c in zip(p.monoms(), p.coeffs())}


def _poly_to_sparse(f, A: Sequence[Sequence[int]]):
    """Coefficients of f as a dict over R[A], with the same variable convention
    as ``deg_A`` (free symbols sorted by str). Returns ``(dict {beta: c}, m)``."""
    fs = sp.expand(sp.sympify(f))
    m = len(A[0])
    if not fs.free_symbols:
        return {tuple([0] * m): float(fs)}, m
    free = sorted(fs.free_symbols, key=lambda s: str(s))
    if len(free) > m:
        raise ValueError(
            "polynomial has %d free symbols but A is a set of %d-dimensional "
            "exponents" % (len(free), m))
    p = sp.Poly(fs, *free)
    out: Dict[Tuple[int, ...], float] = {}
    for monom, c in zip(p.monoms(), p.coeffs()):
        vec = [0] * m
        for j, e in enumerate(monom):
            if e:
                vec[j] = e
        out[tuple(vec)] = out.get(tuple(vec), 0.0) + float(c)
    return out, m


def prolongations(f_system: Sequence[sp.Basic], A: Sequence[Sequence[int]], t: int,
                  index: Dict[Tuple[int, ...], int]):
    r"""The set :math:`H_t` of eq. (9): all products ``x^beta f_i`` with
    ``deg_A(x^beta) <= t - deg_A(f_i)``, expressed in the moment basis.

    Returns one dict per distinct polynomial of H_t, each mapping monomial
    *positions* (in ``index``) to float coefficients -- i.e. the row vector of
    the linear constraint :math:`\Lambda(h) = 0` over R[A]_t.

    Parameters
    ----------
    f_system : polynomials f_i in R[A] (sympy expressions).
    A : exponent set, or a ``ToricSetup``.
    t : truncation degree; must be >= deg_A(f_i) for every i.
    index : monomial -> position map from ``moment_indices(A, t)``.

    Raises
    ------
    ValueError if some f_i is not in R[A] or has A-degree > t.

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> monoms, index, levels = moment_indices([(1,)], 3)
    >>> H = prolongations([x**2 - 1], [(1,)], 3, index)   # {x^2-1, x^3-x}
    >>> sorted(H[0].items()), sorted(H[1].items())        # doctest: +SKIP
    ([(0, -1.0), (2, 1.0)], [(1, -1.0), (3, 1.0)])
    """
    A_int = _as_exponents(A)
    levels = semigroup_level_map(A_int, max(t, 0))
    # H_t is the SET of individual products x^beta f_i (eq. 9): each product
    # yields its own linear constraint row; duplicates removed afterwards.
    rows: List[Dict[Tuple[int, ...], float]] = []
    for f in f_system:
        coeffs, m = _poly_to_sparse(f, A_int)
        fs = sp.sympify(f)
        df = 0 if (not coeffs or all(c == 0 for c in coeffs.values())) else deg_A(fs, A_int)
        if df > t:
            raise ValueError(
                "deg_A(f_i) = %d exceeds the truncation degree t = %d" % (df, t))
        for beta, db in levels.items():
            if db > t - df:
                continue
            h_poly: Dict[Tuple[int, ...], float] = {}
            for gamma, c in coeffs.items():
                h = tuple(e + g for e, g in zip(beta, gamma))
                h_poly[h] = h_poly.get(h, 0.0) + c
            rows.append(h_poly)
    # Deduplicate identical products and rekey by position; drop zero rows.
    seen: set = set()
    out: List[Dict[int, float]] = []
    for coeff_dict in rows:
        pos_map = {index[h]: c for h, c in coeff_dict.items()}
        if not any(c != 0 for c in pos_map.values()):
            continue                    # zero polynomial: no constraint
        key = tuple(sorted(pos_map.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(pos_map)
    return out


@dataclass
class SparseMomentResult:
    """Structured output of ``SparseMomentSDP.solve``.

    Attributes
    ----------
    status : 'Optimal' | 'Infeasible' | ... (solver verdict).
        Infeasible means K_t = empty, hence by Lemma 3.29(i) the system has no
        real toric solution (1 is an SOS modulo I -- a Stengle certificate).
    lambda_vec : moment vector Lambda(x^alpha), alpha in R[A]_t (position-ordered).
    M_s : the sparse truncated moment matrix M_{floor(t/2)}(Lambda).
    rank_Ms, rank_Ms_minus_1 : ranks of M_s and its top-left principal
        submatrix on R[A]^{s-1} -- the flat-extension pair (Theorem 3.27).
    n_monomials_t : |R[A]_t|, number of SDP variables.
    n_monomials_s : |R[A]_{floor(t/2)}|, moment-matrix dimension (the sparsity
        benchmark vs the dense monomial count binom(n+s-1, s)).
    h_count : number of prolongation constraints in H_t after deduplication.
    objective_value : min c^T Lambda over K_t (constant 1 for Algorithm 1).
    wall_time : seconds spent inside CvxpySDPSolver.solve().
    solver_info : the legacy ``Info`` dict from CvxpySDPSolver.
    """
    status: str
    lambda_vec: Optional[np.ndarray] = None
    M_s: Optional[np.ndarray] = None
    rank_Ms: Optional[int] = None
    rank_Ms_minus_1: Optional[int] = None
    n_monomials_t: int = 0
    n_monomials_s: int = 0
    h_count: int = 0
    objective_value: Optional[float] = None
    wall_time: float = 0.0
    solver_info: dict = field(default_factory=dict)


class SparseMomentSDP:
    r"""The K_t spectrahedron of eq. (10), built on :class:`CvxpySDPSolver`.

    .. math::
        K_t = \{ \Lambda \in R[A]^*_t : \Lambda(1)=1,\; M_{\lfloor t/2 \rfloor}(\Lambda) \succeq 0,
               \; \Lambda(f)=0 \;\forall f \in H_t \}.

    One SDP variable per monomial of R[A]_t (enumerated by A-degree, not total
    degree -- this is exactly where the sparsity pays off); one PSD block for
    M_{floor(t/2)}; affine equalities for normalization and prolongations.

    Parameters
    ----------
    f_system : generators f_i of I in R[A].
    A : exponent set, or a ``ToricSetup``.
    t : truncation degree >= max_i deg_A(f_i).
    objective : 'constant' (Algorithm 1: min Lambda(1) -- feasibility probe)
        or an array-like c over R[A]_t (one entry per monomial position, as in
        ``moment_indices``); a random Gaussian choice realizes Algorithm 2.
    solver : SDP backend name passed to CvxpySDPSolver.

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> sdp = SparseMomentSDP([x*(x-1)*(x+1)], [(1,)], 4)   # I < x^3-x >
    >>> res = sdp.solve()
    >>> bool(abs(res.lambda_vec[sdp.index[(0,)]] - 1.0) < 1e-6)   # Lambda(1) = 1
    True
    """

    def __init__(self, f_system: Sequence[sp.Basic], A: Sequence[Sequence[int]], t: int,
                 objective: Union[str, Sequence[float]] = 'constant', solver=None):
        if isinstance(A, ToricSetup):
            A_int = A.A
        else:
            A_int = tuple(_validate_exponents(A))
        self.A = A_int
        self.t = int(t)
        if self.t < 0:
            raise ValueError("t must be non-negative")
        self.f_system = [sp.sympify(f) for f in f_system]

        # Basis of R[A]_t and the moment-matrix index set R[A]_{floor(t/2)}
        monoms, index, levels = moment_indices(A_int, self.t)
        s = self.t // 2
        self.s = s
        self.monoms_t: List[Tuple[int, ...]] = list(monoms)
        self.index: Dict[Tuple[int, ...], int] = dict(index)
        self.levels = levels
        self.n_monomials_t = len(self.monoms_t)

        # Prolongations H_t (eq. 9): one linear constraint per distinct h in R[A]_t
        h_list = prolongations(self.f_system, A_int, self.t, index)
        self.h_count = len(h_list)

        # Objective: constant 1 -> probe feasibility with min Lambda(1);
        # otherwise a coefficient vector over the moment positions.
        if isinstance(objective, str) and objective == 'constant':
            b = np.zeros(self.n_monomials_t)
            zero = tuple([0] * len(A_int[0]))
            b[index[zero]] = 1.0
            self.objective_kind = 'constant'
        else:
            b = np.asarray(objective, dtype=np.float64).ravel()
            if len(b) != self.n_monomials_t:
                raise ValueError(
                    "objective length %d does not match |R[A]_t| = %d"
                    % (len(b), self.n_monomials_t))
            self.objective_kind = 'custom'
        self.b = b

        # Build the standard-form SDP (mirrors SDPRelaxations.InitSDP but on the
        # sparse index set; C[j] blocks are all zero because M_s is linear in Lambda)
        from .cvxpy_solver import CvxpySDPSolver
        self.solver_obj = CvxpySDPSolver(solver=solver) if solver else CvxpySDPSolver()

        sidx = [i for i, e in enumerate(self.monoms_t) if levels[e] <= s]  # R[A]^s rows/cols
        n_s = len(sidx)
        self.s_positions = sidx

        # Gram construction: M_s[p,q] = Lambda(m_p * m_q), so the variable
        # x[var_pos] (coefficient of Lambda(x^beta)) contributes 1 to entry
        # (p, q) exactly when beta == monoms_t[p] + monoms_t[q].
        A_blocks_per_var = [np.zeros((n_s, n_s)) for _ in range(self.n_monomials_t)]
        for var_pos, beta in enumerate(self.monoms_t):
            M = np.zeros((n_s, n_s))
            for a_i, p1 in enumerate(sidx):
                e1 = self.monoms_t[p1]
                for b_j, p2 in enumerate(sidx):
                    e2 = self.monoms_t[p2]
                    if tuple(x + y for x, y in zip(e1, e2)) == beta:
                        M[a_i, b_j] = 1.0
            A_blocks_per_var[var_pos] = M

        for M in A_blocks_per_var:
            self.solver_obj.AddConstraintBlock([M])
        self.solver_obj.AddConstantBlock([np.zeros((n_s, n_s))])
        self.solver_obj.SetObjective(self.b)

        # Affine equalities: Lambda(1) = 1 and Lambda(h) = 0 for h in H_t
        zero = tuple([0] * len(A_int[0]))
        norm = np.zeros(self.n_monomials_t)
        norm[index[zero]] = 1.0
        self.solver_obj.AddEquality(norm, 1.0)
        for pos_map in h_list:
            a = np.zeros(self.n_monomials_t)
            for pos, c in pos_map.items():
                a[pos] += c
            if not (a != 0).any():
                continue                     # zero polynomial: no constraint
            self.solver_obj.AddEquality(a, 0.0)

    def solve(self) -> SparseMomentResult:
        """Solve min b^T Lambda over K_t and return structured moment data."""
        res = self.solver_obj.solve()
        out = SparseMomentResult(
            status=res.status,
            n_monomials_t=self.n_monomials_t,
            n_monomials_s=len(self.s_positions),
            h_count=self.h_count,
            wall_time=res.wall_time,
            solver_info=dict(self.solver_obj.Info),
        )
        if res.status != 'Optimal' or res.x is None:
            return out

        lam = np.asarray(res.x, dtype=np.float64)
        zero = tuple([0] * len(self.A[0]))
        out.lambda_vec = lam
        out.objective_value = float(res.primal_obj) if res.primal_obj is not None else None

        # Assemble M_s and its principal (s-1)-submatrix from the moment vector
        sidx = self.s_positions
        Ms = np.zeros((len(sidx), len(sidx)))
        for a_i, p1 in enumerate(sidx):
            e1 = self.monoms_t[p1]
            for b_j, p2 in enumerate(sidx):
                e2 = self.monoms_t[p2]
                gam = tuple(x + y for x, y in zip(e1, e2))
                Ms[a_i, b_j] = lam[self.index[gam]]
        # symmetrize (numerical) and compute ranks
        Ms = 0.5 * (Ms + Ms.T)
        out.M_s = Ms

        def _rank(M, tol=1e-9):
            return int(np.linalg.matrix_rank(M, tol=tol))
        out.rank_Ms = _rank(Ms)
        # principal submatrix on R[A]^{s-1}: rows/cols with level <= s-1
        prev_idx = [i for i in sidx if self.levels[self.monoms_t[i]] <= self.s - 1]
        if prev_idx:
            Mp = Ms[np.ix_(prev_idx, prev_idx)]
            out.rank_Ms_minus_1 = _rank(0.5 * (Mp + Mp.T))
        else:
            out.rank_Ms_minus_1 = 0          # s = 0: M_{-1} is empty by convention
        return out

    def __str__(self):
        return ("SparseMomentSDP(t=%d, |R[A]_t|=%d, |M_s|=%dx%d, |H_t|=%d)"
                % (self.t, self.n_monomials_t, len(self.s_positions),
                   len(self.s_positions), self.h_count))


# ---------------------------------------------------------------------------
# S51.2: Theorem 3.27 rank conditions and the Algorithm-1 termination loop
# ---------------------------------------------------------------------------

def _numerical_rank(M, tol):
    r"""Numerical rank of a symmetric matrix by eigenvalues (absolute threshold).

    Mirrors ``SDPRelaxations.NumericalRank`` (relaxations.py:1950): count the
    eigenvalues with :math:`|\lambda_i| \ge tol`.  For PSD moment matrices this
    equals the SVD-rank convention; the absolute threshold is what the paper's
    practical tolerances (:math:`10^{-3}..10^{-7}`) are calibrated for.

    Example
    -------
    >>> import numpy as np
    >>> _numerical_rank(np.diag([1.0, 2.0, 1e-9]), 1e-6)
    2
    """
    if M.size == 0:
        return 0
    M = 0.5 * (M + M.T)
    ev = np.linalg.eigvalsh(M)
    return int(np.sum(np.abs(ev) >= tol))


def _moment_matrix_from_vector(lam_vec, monoms, levels, index, s_level):
    r"""Assemble :math:`M_{s\_level}(\Lambda)` from a moment vector over R[A]_t.

    Rows/columns are indexed by the sub-list ``monoms[i]`` with A-degree <=
    ``s_level``, in that same order; entry (p, q) is ``Lambda(x^{e_p + e_q})``.

    Parameters
    ----------
    lam_vec : array of moments Lambda(x^alpha), position-ordered over R[A]_t.
    monoms, levels, index : the basis data from ``moment_indices(A, t)``.
    s_level : A-degree cutoff s; all products e_p + e_q must satisfy
        deg_A(e_p + e_q) <= 2*s_level <= t (guaranteed when s_level <= t//2).
    """
    idx = [i for i, e in enumerate(monoms) if levels[e] <= s_level]
    n = len(idx)
    M = np.zeros((n, n))
    for a_i in range(n):
        e1 = monoms[idx[a_i]]
        for b_j in range(a_i, n):
            e2 = monoms[idx[b_j]]
            gam = tuple(x + y for x, y in zip(e1, e2))
            v = float(lam_vec[index[gam]])
            M[a_i, b_j] = v
            M[b_j, a_i] = v
    return 0.5 * (M + M.T)


def _column_basis_positions(M, tol):
    r"""Positions indexing a monomial column basis of ``M`` (greedy in order).

    Scans columns left to right and keeps those that increase the numerical
    rank; for moment matrices whose rows are ordered by nondecreasing A-degree
    this yields a basis "connected-to-one" (Remark 3.28: select greedily in
    increasing deg_A order to get a sparse border basis).

    Example
    -------
    >>> import numpy as np
    >>> _column_basis_positions(np.array([[1., 0.], [0., 0.]]), 1e-6)
    [0]
    """
    if M.size == 0:
        return []
    chosen = []
    for j in range(M.shape[1]):
        cand = sorted(chosen + [j])
        sub = M[:, cand]
        # independence via the (symmetric) Gram matrix -- works on rectangular stacks
        if _numerical_rank(sub.T @ sub, tol) > len(chosen):
            chosen.append(j)
    return chosen


def _border_basis_positions(M, sub_monoms, A_int, tol, r):
    r"""Positions indexing a *connected-to-one* monomial border basis of ``M``.

    Wrapper around :func:`_column_basis_positions` that enforces Definition A.2
    (reachability from the constant monomial 1 by successive multiplication by
    generators of ``A``).  The greedy-in-increasing-degree selection of Remark
    3.28 already yields a connected-to-one basis whenever :math:`\Lambda` is a
    *generic* element of :math:`K_t` (relative interior).  When the optimizer
    sits on a face boundary and the greedy Gram test becomes tolerance-sensitive
    (a first-order solver artifact), we fall back to a bounded enumeration of
    the full-rank column subsets of size ``r`` and return the first (in
    increasing-degree / lexicographic order) that is connected-to-one -- exactly
    the border basis that the paper's Algorithm 1 step 5 must recover.

    Parameters
    ----------
    M : the moment matrix :math:`M_{s-1}(\Lambda)`.
    sub_monoms : the exponent tuples indexing the columns of ``M``, already in
        the same (A-degree, lex) order used by :func:`moment_indices`.
    A_int : tuple of generator exponent vectors.
    tol : absolute eigenvalue threshold for numerical rank.
    r : target basis size (= ``n_real`` when converged and Lambda generic).

    Returns a list of column positions into ``M``.
    """
    n = M.shape[1]
    if n == 0 or r <= 0:
        return list(range(min(r, n)))

    def connected(exps):
        m = len(A_int[0])
        one = (0,) * m
        if one not in exps:
            return False
        Bs = set(exps)
        reach = {one}
        stack = [one]
        while stack:
            e = stack.pop()
            for a in A_int:
                f_ = tuple(x + y for x, y in zip(e, a))
                if f_ in Bs and f_ not in reach:
                    reach.add(f_)
                    stack.append(f_)
        return len(reach) == len(exps)

    def full_rank(positions):
        # SVD rank on the rectangular column submatrix directly.  The Gram test
        # sub.T@sub SQUARES singular values (eig^-2 of a 2.4e-3 column -> 5.8e-6),
        # which an absolute threshold would mis-drop; counting the submatrix's own
        # singular values measures true column independence at the rank-test tol.
        sv = np.linalg.svd(M[:, np.array(sorted(positions))], compute_uv=False)
        return int(np.sum(sv >= tol)) >= r

    greedy = _column_basis_positions(M, tol)

    # Prefer the lex-first (lowest deg_A / smallest positions) connected-to-one
    # full-rank size-r column subset, which is the canonical border basis that
    # Algorithm 1 step 5 must recover (Remark 3.28).  The greedy Gram test can
    # drop a genuinely-independent low-degree column when the optimizer sits on a
    # face boundary (tolerance-sensitive), so enumerate for small sizes.
    from itertools import combinations
    if r <= 6 and n <= 12 or n <= 20 and r <= 5:
        for pos in combinations(range(n), r):
            if full_rank(pos) and connected([sub_monoms[i] for i in pos]):
                return list(pos)
    if len(greedy) == r and connected([sub_monoms[i] for i in greedy]):
        return greedy
    return greedy


def _snap_coeff(c, tol):
    """Snap a float coefficient: |c| < tol -> 0; near-integer -> int (clean str)."""
    if abs(c) < tol:
        return 0.0
    ci = round(c)
    if abs(c - ci) < 1e-8 * max(1.0, abs(ci)):
        return int(ci)
    return c


def _nullspace_polys(Ms, monoms_s, names, tol):
    r"""Basis of ``ker(Ms)`` as sympy polynomials in the variables ``names``.

    Each null vector (eigenvector with |lambda| < tol of the symmetric matrix
    ``Ms``, whose rows/columns are indexed by the monomials ``monoms_s``) is
    turned into :math:`p = \sum_i v_i x^{\beta_i}`; before snapping, it is
    normalized so that its largest-magnitude coefficient has absolute value 1.

    Returns an empty list for a full-rank matrix (J = <0> case handled by the
    caller via the rank test, not here).
    """
    if Ms.size == 0:
        return []
    ev, evec = np.linalg.eigh(Ms)                    # ascending: nullspace FIRST
    n_null = int(np.sum(ev < tol))
    polys: List[sp.Expr] = []
    for k in range(n_null):
        v = evec[:, k]
        cm = float(np.max(np.abs(v))) if v.size else 0.0
        if cm > tol:
            v = v / cm                              # largest |coeff| == 1 -> snapping works
        terms = []
        for i, beta in enumerate(monoms_s):
            c = _snap_coeff(float(v[i]), tol)
            if c == 0.0:
                continue
            mon = sp.sympify(1)
            for j, e_j in enumerate(beta):
                if e_j:
                    mon *= names[j] ** e_j
            terms.append(sp.Mul(c, mon))
        p = sp.expand(sum(terms)) if terms else sp.sympify(0.0)
        polys.append(p)
    return polys


@dataclass
class RankTestResult:
    """Outcome of the Theorem 3.27 convergence test at one truncation t.

    Attributes
    ----------
    converged : True iff condition (i) or (ii) holds for some admissible s.
    condition : 'flat' (case i), 'interpolation' (case ii), or None.  The two
        cases carry different proof obligations in Thm 3.27, so the distinction
        is kept even though both imply a flat extension at level s.
    s_star : smallest admissible s at which convergence was detected.
    D : max_i deg_A(f_i) (0 when I < 0>); d = max(1, ceil(D/2)); rho_A from T_A.
    ranks : {k: rank(M_k(Lambda))} for all 0 <= k <= floor(t/2).
    n_real : rank(M_{s_star}(Lambda)) -- the number of real toric roots when
        converged and Lambda generic (Lemma 3.14 / Thm 3.27).
    t, tol : truncation degree and eigenvalue threshold used.

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> sdp = SparseMomentSDP([x**3 - x], [(1,)], 6)
    >>> res = sdp.solve()
    >>> rt = test_theorem_327(res.lambda_vec, [x**3 - x], [(1,)], 6,
    ...                       sdp.index, sdp.levels, sdp.monoms_t)
    >>> (rt.converged, rt.condition, rt.s_star, rt.n_real)
    (True, 'flat', 3, 3)
    """
    converged: bool = False
    condition: Optional[str] = None
    s_star: Optional[int] = None
    D: int = 0
    d: int = 1
    rho_A: int = 0
    ranks: Dict[int, int] = field(default_factory=dict)
    n_real: Optional[int] = None
    t: int = 0
    tol: float = 1e-6


def test_theorem_327(lam_vec, f_system: Sequence[sp.Basic], A: Sequence[Sequence[int]],
                     t: int, index: Dict[Tuple[int, ...], int], levels: dict,
                     monoms: List[Tuple[int, ...]], tol: float = 1e-6) -> RankTestResult:
    r"""Theorem 3.27 convergence test for an optimizer :math:`\Lambda \in K_t`.

    With :math:`D = \max_i \deg_A(f_i)` and :math:`d = \lceil D/2\rceil`, check
    for admissible s (case i: :math:`\max\{D, \rho_A\} \le s \le \lfloor t/2\rfloor`;
    case ii: :math:`\max\{d, \rho_A\} \le s \le \lfloor t/2\rfloor`):

    (i)   rank(M_s(Lambda)) = rank(M_{s-1}(Lambda)), or
    (ii)  rank(M_s(Lambda)) = rank(M_{s-d}(Lambda)).

    Either condition implies :math:`\sqrt R I \subseteq J := \langle \ker
    M_s(\Lambda)
angle`; for generic Lambda, equality (paper Thm 3.27).

    The full rank profile is always returned so a non-convergent truncation can
    still report which ranks were computed.  Note: for small t the admissible
    range may be empty -- then ``converged`` is False and the caller must
    increment t (Algorithm 1 step 4).

    Parameters
    ----------
    lam_vec : moment vector Lambda(x^alpha) over R[A]_t (position-ordered).
    f_system : generators of I (sympy expressions in R[A]).
    A : exponent set, or a ToricSetup.
    t : the truncation degree K_t was solved at; must be >= 2*max admissible s.
    index, levels, monoms : the basis data from ``moment_indices(A, t)``.
    tol : absolute eigenvalue threshold for numerical ranks.

    Returns a :class:`RankTestResult`.
    """
    A_int = _as_exponents(A)
    fs = [sp.sympify(f) for f in f_system]
    D = 0
    for f in fs:
        if not sp.sympify(f).is_zero:
            D = max(D, deg_A(f, A_int))
    d = int(np.ceil(D / 2.0)) if D > 0 else 1       # I < 0> guard (degenerate)
    rho_A = ToricSetup.from_exponents(A_int).rho_A

    out = RankTestResult(t=t, tol=tol, D=D, d=d, rho_A=rho_A)
    s_max = t // 2
    for k in range(0, s_max + 1):
        M_k = _moment_matrix_from_vector(lam_vec, monoms, levels, index, k)
        out.ranks[k] = _numerical_rank(M_k, tol)

    s_min_i = max(D, rho_A)
    s_min_ii = max(d, rho_A)
    for s in range(0, s_max + 1):
        r_s = out.ranks[s]
        if s >= s_min_i and out.ranks.get(s - 1, None) is not None                 and out.ranks[s - 1] == r_s:
            out.converged = True
            out.condition = 'flat'
            out.s_star = s
            out.n_real = r_s
            break
        if s >= s_min_ii and out.ranks.get(s - d, None) is not None                 and out.ranks[s - d] == r_s:
            out.converged = True
            out.condition = 'interpolation'
            out.s_star = s
            out.n_real = r_s
            break
    return out


@dataclass
class Algorithm1Result:
    """Outcome of the Algorithm-1 termination loop (paper, end of Section 3.4).

    Attributes
    ----------
    converged : False when K_t was infeasible at some t (Lemma 3.29(i): then
        V_{R[A]}(I) is empty -- a Stengle certificate), or when
        ``max_iterations`` was exceeded without rank stabilization.
    no_real_roots : True iff the loop stopped on an infeasible K_t.
    t : truncation degree at which the loop stopped.
    iterations : number of K_t SDPs solved (t increments by 1 each iteration).
    rank_test : final RankTestResult (None when not converged).
    kernel_polys : sympy expressions p = sum v_i x^{beta_i} spanning
        ker(M_{s_star}(Lambda)) -- a generating set for J = <ker M_s(Lambda)>.
        When converged and Lambda generic, J is exactly the real radical.
        Variable names follow the ``deg_A`` convention (sorted free symbols of
        the input system).  Coefficients are snapped to 0/integers when close.
    quotient_basis : exponent tuples indexing a monomial column basis B of
        M_{s_star - 1}(Lambda); by Corollary 3.19 their residue classes form a
        basis of R[A]/J (Remark 3.28: the greedy-in-order choice gives a
        connected-to-one border basis).  len(quotient_basis) = n_real when
        converged and Lambda generic.
    lambda_vec, moment_matrix : the optimizer and its flat matrix M_{s_star}
        (diagnostics; None until convergence).
    infeasible_at_t : t at which K_t was reported empty (None otherwise).

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> r = solve_algorithm1([x**3 - x], A=[(1,)])
    >>> (r.converged, r.t, sorted(str(p) for p in r.kernel_polys))
    (True, 6, ['x**3 - x'])

    Notes
    -----
    The constant-objective SDP returns a generic element of K_t with
    interior-point solvers (Lemma 3.23 / eq. (11): all relint points share the
    same kernel space N_t); first-order solvers may land on the boundary, in
    which case pass ``solver='CLARABEL'`` explicitly.
    """
    converged: bool = False
    no_real_roots: bool = False
    t: int = 0
    iterations: int = 0
    rank_test: Optional[RankTestResult] = None
    kernel_polys: List[sp.Expr] = field(default_factory=list)
    quotient_basis: List[Tuple[int, ...]] = field(default_factory=list)
    lambda_vec: Optional[np.ndarray] = None
    moment_matrix: Optional[np.ndarray] = None
    infeasible_at_t: Optional[int] = None

    def __str__(self):
        if self.no_real_roots:
            return "Algorithm1Result: K_%d empty -> no real toric root" \
                % (self.infeasible_at_t or 0)
        if not self.converged:
            return ("Algorithm1Result: not converged after %d iterations "
                    "(t up to %d)" % (self.iterations, self.t))
        rt = self.rank_test
        assert rt is not None and rt.s_star is not None and rt.n_real is not None
        return ("Algorithm1Result: t=%d, s*=%d (%s), n_real=%d, |B|=%d"
                % (self.t, rt.s_star, rt.condition, rt.n_real, len(self.quotient_basis)))


def solve_algorithm1(f_system: Sequence[sp.Basic], A: Optional[Sequence[Sequence[int]]] = None,
                     t0: Optional[int] = None, tol: float = 1e-6, max_iterations: int = 8,
                     solver=None) -> Algorithm1Result:
    r"""Algorithm 1 (all real roots): the K_t SDP + Theorem 3.27 termination loop.

    Solves ``min Lambda(1)`` over :math:`K_t` (eq. (10)) at truncation t, tests
    the rank conditions of Theorem 3.27, and on failure increments
    :math:`t \leftarrow t + 1` and re-solves (paper step 4).  Stopping cases:

    * K_t infeasible -> by Lemma 3.29(i) there is no real toric solution;
    * rank stabilization at some admissible s -> J = <ker M_s(Lambda)> is the
      real radical for generic Lambda (Thm 3.27);
    * ``max_iterations`` exceeded -> RuntimeError with the last rank profile.

    Parameters
    ----------
    f_system : generators of I, each in R[A] (sympy expressions).
    A : exponent set; optional only for single-variable systems, where it is
        inferred as [(1,)] from the free symbols.
    t0 : starting truncation degree; defaults to ``2*max{D, rho_A}`` per
        Algorithm 1 step 1 (with a floor of max(4, D) so K_t contains H_t).
    tol : eigenvalue threshold for numerical ranks (paper practice: 1e-3..1e-7).
    max_iterations : guard against Lemma 3.29's "sufficiently large t".
    solver : SDP backend name passed to ``CvxpySDPSolver``; interior-point
        methods are preferred so the optimizer lies in relint(K_t) (Lemma 3.23).

    Returns an :class:`Algorithm1Result`.

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> r = solve_algorithm1([x**3 - x], A=[(1,)])
    >>> (r.converged, r.t, sorted(str(p) for p in r.kernel_polys))
    (True, 6, ['x**3 - x'])

    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> r = solve_algorithm1([x**2 + 1], A=[(1,)])
    >>> (r.converged, r.no_real_roots)
    (False, True)
    """
    if A is None:
        free = sorted(set().union(*[set(sp.sympify(f).free_symbols) for f in f_system]),
                      key=str)
        if len(free) != 1:
            raise ValueError("A cannot be inferred from %d variables; pass it "
                             "explicitly" % len(free))
        A_int = ((1,),)
    else:
        A_int = _as_exponents(A)

    fs = [sp.sympify(f) for f in f_system]
    D = 0
    for f in fs:
        if not sp.sympify(f).is_zero:
            D = max(D, deg_A(f, A_int))
    rho_A = ToricSetup.from_exponents(A_int).rho_A
    if t0 is None:
        t0 = max(4, 2 * max(D, rho_A), D)

    names = sorted(set().union(*[set(sp.sympify(f).free_symbols) for f in fs]),
                   key=str) or [sp.Symbol('x')]
    # pad to ambient dimension with x1..xn-style fallbacks (shouldn't happen:
    # free symbols of R[A]-polynomials span all ambient coordinates used)
    while len(names) < len(A_int[0]):
        names.append(sp.Symbol('v%d' % (len(names) + 1)))

    out = Algorithm1Result()
    last_rt: Optional[RankTestResult] = None
    t = int(t0)
    for it in range(1, max_iterations + 1):
        sdp = SparseMomentSDP(fs, A_int, t, objective='constant', solver=solver)
        res = sdp.solve()
        out.iterations = it
        out.t = t
        if res.status != 'Optimal' or res.lambda_vec is None:
            # K_t empty: by Lemma 3.29(i), V_{R[A]}(I) = empty (Stengle).
            out.no_real_roots, out.infeasible_at_t = True, t
            return out

        lam = np.asarray(res.lambda_vec, dtype=np.float64)
        rt = test_theorem_327(lam, fs, A_int, t, sdp.index, sdp.levels,
                              sdp.monoms_t, tol)
        last_rt = rt
        if not rt.converged:
            t += 1                      # paper step 4: t <- t + 1 and re-solve
            continue

        out.converged = True
        out.rank_test = rt
        out.lambda_vec = lam
        s_star = rt.s_star
        assert s_star is not None

        Ms = _moment_matrix_from_vector(lam, sdp.monoms_t, sdp.levels,
                                        sdp.index, s_star)
        out.moment_matrix = Ms

        # --- kernel polynomials: J = <ker M_s(Lambda)> (paper step 5)
        monoms_s = [sdp.monoms_t[i] for i in range(len(sdp.monoms_t))
                    if sdp.levels[sdp.monoms_t[i]] <= s_star]
        out.kernel_polys = _nullspace_polys(Ms, monoms_s, names, tol)

        # --- quotient basis: column basis of M_{s-1} (Corollary 3.19 / Remark 3.28)
        if s_star >= 1:
            Mp = _moment_matrix_from_vector(lam, sdp.monoms_t, sdp.levels,
                                            sdp.index, s_star - 1)
            sub = [sdp.monoms_t[i] for i in range(len(sdp.monoms_t))
                   if sdp.levels[sdp.monoms_t[i]] <= s_star - 1]
            positions = _border_basis_positions(Mp, sub, A_int, tol, rt.n_real)
            out.quotient_basis = [sub[i] for i in positions]
        else:
            zero = tuple([0] * len(A_int[0]))
            out.quotient_basis = [zero]      # M_0 rank 1 -> basis {1} of R[A]/J
        return out

    ranks = last_rt.ranks if last_rt is not None else {}
    raise RuntimeError(
        "Algorithm 1 did not converge after %d iterations (t up to %d). "
        "Last rank profile: %s" % (max_iterations, t, dict(sorted(ranks.items()))))


# ---------------------------------------------------------------------------
# S51.3: Sparse border basis of R[A]/J (connected-to-one, paper Def A.1-A.3)
# ---------------------------------------------------------------------------


class SparseBorderBasis:
    r"""Sparse border basis B of the quotient algebra :math:`R[A]/J`, J =
    :math:`\langle \ker M_{s^{*}}(\Lambda)\rangle`, built from the moment data of a
    converged Algorithm-1 run (Theorem 3.27 + Corollary 3.19 / Remark 3.28).

    B is taken to be the quotient basis of the :class:`Algorithm1Result` --
    the greedy-in-deg_A-order column basis of :math:`M_{s^{*}-1}(\Lambda)` --
    which Remark 3.28 guarantees connected-to-one (Def A.2); this class
    *verifies* that property by BFS rather than assuming it.

    For every exponent e with :math:`\deg_A(e) \le s^{*}` the normal form
    :math:`[x^e] = \sum_j d_j b_j` (i.e. :math:`x^e - \sum_j d_j b_j \in J`) is
    recovered from the moment equalities of flat extension: under the rank
    condition, Lambda admits a representing measure with n_real atoms on
    V_{R[A]}(J), hence for all u with :math:`\deg_A(u) + \deg_A(e) \le t`

        Lambda(x^{u+e}) = sum_j d_j Lambda(x^{u+b_j}),

    an exactly consistent linear system solved in least squares (residuals are
    exposed for diagnostics).  The multiplication table of each semigroup
    generator x^{a_i} is then the matrix whose k-th column is [x^{a_i} b_k] in B;
    these tables are self-adjoint on V, commute, and respect every toric
    relation (Def A.3) -- verified numerically by :meth:`check_relations`
    (Appendix A.1 normal-form criterion).

    Parameters
    ----------
    lam_vec : moment vector Lambda(x^alpha) over R[A]^t (position-ordered), t >= 2*s_star.
    basis : exponent tuples of B, with basis[0] == (0,...,0); normally
        ``Algorithm1Result.quotient_basis``.
    A : exponent set (sequence or ToricSetup).
    t : truncation degree; levels must cover R[A]^{s_star}.
    s_star : level at which flat extension was detected.
    monoms, index, levels : basis data from ``moment_indices(A, t)``.
    tol : least-squares / residual threshold (default 1e-6).

    Attributes
    ----------
    basis, r_ : the border basis and its cardinality (= n_real on convergence).
    tables : list of r_ x r_ arrays; column k = [x^{a_i} b_k] in B.
    normal_forms[e] : coefficient vector d over R[A]^{s_star}, x^e - sum d_j b_j in J.
    nf_residuals[e] : lstsq residual of the moment system for e (0 on exact flatness).
    connected_to_one : Def A.2 flag, verified by BFS from 1 inside B.

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> r = solve_algorithm1([x**3 - x], A=[(1,)])
    >>> monoms, index, levels = moment_indices([(1,)], r.t)
    >>> bb = SparseBorderBasis(r.lambda_vec, r.quotient_basis, [(1,)], r.t,
    ...                        r.rank_test.s_star, monoms, index, levels)
    >>> (bb.r_, tuple(bb.basis), bb.connected_to_one)
    (3, ((0,), (1,), (2,)), True)

    Notes
    -----
    Semigroup-aware twin of :class:`Irene.border_basis.BorderBasis` (dense ring):
    here the extra constraint is Def A.3 -- every g in G_A must annihilate V
    under M(g), which ``check_relations`` tests.
    """

    def __init__(self, lam_vec, basis: Sequence[Sequence[int]], A, t: int,
                 s_star: int, monoms: List[Tuple[int, ...]],
                 index: Dict[Tuple[int, ...], int], levels: dict,
                 tol: float = 1e-6):
        if isinstance(A, ToricSetup):
            A_int = tuple(A.A)
        else:
            A_int = _as_exponents(A)
        self.lam_vec = np.asarray(lam_vec, dtype=np.float64)
        self.A = A_int
        self.t = int(t)
        self.s_star = int(s_star)
        self.monoms_t = list(monoms)
        self.index = dict(index)
        self.levels = levels
        self.tol = float(tol)

        m = len(A_int[0])
        zero = (0,) * m
        basis = [tuple(int(v) for v in e) for e in basis]
        if not basis or basis[0] != zero:
            raise ValueError("border basis must start with the constant 1")
        self.basis = list(basis)
        self.r_ = len(self.basis)
        posB = {e: k for k, e in enumerate(self.basis)}

        # --- normal forms of all enumerated monomials up to level s_star -----
        self._Ainf = [e for e in self.monoms_t if self.levels[e] <= self.s_star]
        n_inf = len(self._Ainf)
        # moment system rows: u with deg_A(u) <= t - s_star (enough so that both
        # u+e and u+b_j stay in R[A]^t for every stored e, b_j).  Theorem 3.17's
        # flat extension pins moments exactly up to level 2*s_star; rows u whose
        # level exceeds 2*s_star - level(e) would need moments beyond that flat
        # range, which a first-order solver end may leave as boundary-only noise.
        # So each exponent e uses only rows with level(u) <= 2*s_star - level(e).
        _u_max = max(0, self.t - self.s_star)
        us_all = [e for e in self.monoms_t if 0 <= self.levels[e] <= _u_max]
        _2s = 2 * self.s_star
        # U_all[k, j] = Lambda(x^{u_k + _Ainf[j]}) -- full normal-form coordinate.
        U_all = np.zeros((len(us_all), n_inf))
        for k, u in enumerate(us_all):
            for j, b in enumerate(self._Ainf):
                ub = tuple(x_ + y_ for x_, y_ in zip(u, b))
                if ub in self.index:
                    U_all[k, j] = float(self.lam_vec[self.index[ub]])
        Bcols = [self.index[b] for b in self.basis]     # positions of basis in _Ainf
        # store per-row basis-coordinate extraction: normal form vector over _Ainf
        # restricted to B columns is exactly what the tables need; keep full vectors.
        nf: Dict[Tuple[int, ...], np.ndarray] = {}
        res: Dict[Tuple[int, ...], float] = {}
        for e in self._Ainf:
            if e in posB:
                d = np.zeros(n_inf)
                d[self.index[e]] = 1.0
                nf[e] = d
                res[e] = 0.0
            else:
                # rows valid for THIS e (flat-extension pinning range)
                rows = [k for k, u in enumerate(us_all) if self.levels[e] <= _2s - self.levels[u]]
                UB = U_all[np.array(rows, dtype=int) if rows else np.empty(0, int)][:, Bcols]
                rhs = np.array([float(self.lam_vec[self.index[tuple(
                    x_ + y_ for x_, y_ in zip(u, e))]]) for u in (us_all[k] for k in rows)])
                d, _, _, _ = np.linalg.lstsq(UB, rhs, rcond=None)
                # lift to the full R[A]^{s_star} coordinate vector (zeros outside B
                # are NOT asserted: only the B-coordinates carry meaning under J)
                dfull = np.zeros(n_inf)
                for j, b in enumerate(self.basis):
                    dfull[self.index[b]] += d[j]
                nf[e] = dfull
                res[e] = float(np.linalg.norm(UB @ d - rhs) if UB.shape[0] else 0.0)
        self.normal_forms: Dict[Tuple[int, ...], np.ndarray] = nf
        self.nf_residuals: Dict[Tuple[int, ...], float] = res

        # --- multiplication tables (column k = [x^{a_i} b_k]) -----------------
        self.tables: List[np.ndarray] = []
        for a in A_int:
            T = np.zeros((self.r_, self.r_))
            for k, b in enumerate(self.basis):
                e = tuple(x_ + y_ for x_, y_ in zip(b, a))
                if e not in nf or e not in self.index:
                    raise ValueError(
                        "product x^(%s) * basis element left R[A]^t; increase t" % (e,))
                d = nf[e]
                for j, bj in enumerate(self.basis):
                    T[j, k] += float(d[self.index[bj]])
            self.tables.append(T)

        # --- connected-to-one verification (Def A.2), BFS from 1 --------------
        reach = {zero}
        stack = [zero]
        while stack:
            e = stack.pop()
            for a in A_int:
                f_ = tuple(x_ + y_ for x_, y_ in zip(e, a))
                if f_ in posB and f_ not in reach:
                    reach.add(f_)
                    stack.append(f_)
        self.connected_to_one: bool = (len(reach) == self.r_)

    # -- helpers --------------------------------------------------------------

    def normal_form(self, e):
        """Coefficient vector d over R[A]^{s_star}: x^e - sum_j d[b_j] b_j in J."""
        if e not in self.normal_forms:
            raise ValueError(
                "exponent %s has A-degree > s_star; no normal form stored" % (e,))
        return self.normal_forms[e]

    def check_relations(self) -> float:
        r"""Max over g in G_A of :math:`\|g(M_1,\ldots,M_r)\|_2` -- the Def A.3 /
        Appendix A.1 compatibility residual (0 when every toric relation holds on V).

        Example
        -------
        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> r = solve_algorithm1([x**4 - 5*x**2 + 4], A=[(2, 0), (3, 0)])
        >>> monoms, index, levels = moment_indices([(2, 0), (3, 0)], r.t)
        >>> bb = SparseBorderBasis(r.lambda_vec, r.quotient_basis, [(2, 0), (3, 0)],
        ...                        r.t, r.rank_test.s_star, monoms, index, levels)
        >>> bool(bb.check_relations() < 1e-6)      # z1^3 = z2^2 holds on V
        True
        """
        ts = ToricSetup.from_exponents(self.A)
        if not ts.G_A:
            return 0.0
        zs = list(ts.z)
        worst = 0.0
        for g in ts.G_A:
            d_ = sp.Poly(sp.expand(g), *zs).as_dict()
            acc = np.zeros((self.r_, self.r_))
            for beta, c in d_.items():
                Mprod = np.eye(self.r_)
                for jj in range(len(beta)):
                    if not beta[jj]:
                        continue
                    P = np.eye(self.r_)
                    for _ in range(int(beta[jj])):     # true matrix powers (not elementwise)
                        P = P @ self.tables[jj]
                    Mprod = Mprod @ P
                acc += float(c) * Mprod
            worst = max(worst, float(np.linalg.norm(acc, ord=2)))
        return worst

    def __repr__(self):
        return ("SparseBorderBasis(|A|=%d, |B|=%d, s*=%d%s)" %
                (len(self.A), self.r_, self.s_star,
                 "" if self.connected_to_one else ", NOT connected-to-one"))


# ---------------------------------------------------------------------------
# S51.4: Sparse Stickelberger eigenvalue recovery (paper Appendix A.2) + BWE
# ---------------------------------------------------------------------------


@dataclass
class SparseRootsResult:
    """Output of the sparse Stickelberger eigenvalue method (Appendix A.2).

    Attributes
    ----------
    roots : one dict per real toric root, {'z1': v1, ..., 'z_r': vr} with
        z_i = x^{a_i} the semigroup generators (values m^{a_i}).
    bwe : backward error per root, eq. (12): (1/m) sum_i |Lambda_m(f_i)| /
        (sum_alpha |c_{alpha,f} Lambda_m(x^alpha)| + 1).  The paper reports
        log10(BWE) in [-7, -3] for its random systems.
    bwe_components : per-root list of the m summands above (diagnostics).
    eigenvalues : the r_ distinct real values Lambda_m(f) of the successful
        Stickelberger functional f = c_1 x^{a_1} + ... + c_r x^{a_r}.
    coefficients : integer coefficient vector c of that functional.
    trials : number of random functionals tried (genericity, Theorem A.4).
    kernel_polys : generators of J = <ker M_{s_star}(Lambda)> (Algorithm 1 step 5).
    connected_to_one : Def A.2 flag from the border basis used.

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> r = solve_algorithm1([x**3 - x], A=[(1,)])
    >>> out = recover_all_sparse_roots(r, [(1,)], [x**3 - x])
    >>> sorted(round(v['z1'], 6) for v in out.roots)
    [-1.0, 0.0, 1.0]
    """
    roots: List[Dict[str, float]] = field(default_factory=list)
    bwe: List[float] = field(default_factory=list)
    bwe_components: List[List[float]] = field(default_factory=list)
    eigenvalues: List[float] = field(default_factory=list)
    coefficients: Optional[Tuple[int, ...]] = None
    trials: int = 0
    kernel_polys: List[sp.Expr] = field(default_factory=list)
    connected_to_one: bool = False

    def __str__(self):
        if not self.roots:
            return "SparseRootsResult: no roots recovered"
        worst = max(self.bwe) if self.bwe else float('nan')
        return ("SparseRootsResult: %d real toric root(s), worst BWE %.3e, %d trial(s)"
                % (len(self.roots), worst, self.trials))


def _semigroup_decomposition(beta: Tuple[int, ...], A_int: Tuple[Tuple[int, ...], ...]) -> Optional[List[int]]:
    """One representation beta = sum_i n_i a_i in N^r (any works; the value on V
    is independent of the choice since G_A vanishes there).  None if unreachable."""
    zero = tuple([0] * len(beta))
    if beta == zero:                       # the constant 1 is always in <A>
        return [0] * len(A_int)
    reps = {zero: [0] * len(A_int)}
    frontier = [zero]
    while frontier:
        nxt = []
        for e in frontier:
            for i, a in enumerate(A_int):
                f_ = tuple(x_ + y_ for x_, y_ in zip(e, a))
                if f_ == beta:
                    r0 = reps[e][:]
                    r0[i] += 1
                    return r0
                # BFS only inside the box [0..beta] (monotone paths suffice)
                if any(fc > bc for fc, bc in zip(f_, beta)):
                    continue
                if f_ not in reps:
                    r0 = reps[e][:]
                    r0[i] += 1
                    reps[f_] = r0
                    nxt.append(f_)
        frontier = nxt
    return None


def _poly_support_A(f) -> Dict[Tuple[int, ...], float]:
    """Monomial support of an R[A]-polynomial as {exponent_tuple: coefficient}.

    Exponents are read off the free symbols sorted by name (the deg_A /
    solve_algorithm1 convention); raises ValueError on negative exponents.
    """
    fs = sp.sympify(f)
    names = sorted(fs.free_symbols, key=str) or [sp.Symbol('x')]
    out: Dict[Tuple[int, ...], float] = {}
    p = sp.Poly(sp.expand(fs), *names)
    for exp_key, c in p.as_dict().items():
        if any(int(e) < 0 for e in exp_key):
            raise ValueError("negative exponents in %s" % f)
        out[tuple(int(e) for e in exp_key)] = float(c)
    return out


def recover_all_sparse_roots(result: "Algorithm1Result", A,
                             f_system: Optional[Sequence] = None, *,
                             seed: int = 2026, max_trials: int = 50,
                             tol: float = 1e-8) -> SparseRootsResult:
    r"""Recover all points of :math:`V_{R[A]}(I)` from a converged Algorithm-1 run.

    Implements the sparse Stickelberger eigenvalue method (paper Appendix A.2,
    Theorem A.4): pick random integer coefficients c = (c_1,...,c_r), form the
    multiplication matrix of :math:`f = \sum_i c_i x^{a_i}` on the border basis B,
    take its left eigenvectors w normalized by their 1-coordinate (eq. 14:
    :math:`w \propto (\Lambda_m(x^{b_1}),\ldots,\Lambda_m(x^{b_r}))`), and read off each
    generator value -- directly from the W-row when :math:`x^{a_i} \in B`, else via
    the normal form of x^{a_i} (eq. 15).  A trial is generic iff it yields exactly
    r_ distinct real eigenvalues; random integer coefficients are distinct w.p. 1,
    so a single retry suffices with overwhelming probability.

    Parameters
    ----------
    result : CONVERGED ``Algorithm1Result`` (lambda_vec and quotient_basis set).
    A : the exponent set used for the run ((sequence of tuples) or ToricSetup);
        must match what produced ``result``.
    f_system : generators of I, used only for the BWE diagnostic (eq. 12);
        defaults to [] (BWE = 0).

    Returns a :class:`SparseRootsResult`.
    """
    from scipy import linalg as sla   # local: heavy dependency only at call time

    if not result.converged or result.lambda_vec is None or result.rank_test is None:
        raise ValueError("result must be a CONVERGED Algorithm1Result")
    rt = result.rank_test
    s_star, r_ = int(rt.s_star), int(rt.n_real)
    lam = np.asarray(result.lambda_vec, dtype=np.float64)

    if isinstance(A, ToricSetup):
        A_int = tuple(A.A)
    else:
        A_int = _as_exponents(A)
    t = int(result.t)
    monoms, index, levels = moment_indices(A_int, t)

    bb = SparseBorderBasis(lam, list(result.quotient_basis), A_int, t, s_star,
                           monoms, index, levels)
    out = SparseRootsResult(kernel_polys=list(result.kernel_polys),
                            connected_to_one=bb.connected_to_one)
    fs = [sp.sympify(f) for f in (f_system or [])]

    zero = (0,) * len(A_int[0])
    posB = {e: k for k, e in enumerate(bb.basis)}
    i_one = posB[zero]

    def values_from_w(w):
        c0 = w[i_one]
        if abs(c0) < 1e-8:
            return None
        vals_B = w / c0                       # row m of W, normalized by Lambda_m(1)=1... (W[m,zero]=r_>0; eq. 14 normalizes to 1)
        snap = lambda v: 0.0 if abs(v) < tol else float(v)   # -0.0 / 1e-16 -> clean 0
        gens = []
        for a in A_int:
            if a in posB:
                gens.append(snap(vals_B[posB[a]]))
            else:
                d = bb.normal_form(a)         # level(a)=1 <= s_star always
                v = 0.0
                for j, b in enumerate(bb.basis):
                    cj = float(d[index[b]])
                    if abs(cj) > tol:
                        v += cj * vals_B[j]
                gens.append(snap(v))
        return tuple(gens)

    rng = np.random.default_rng(seed)
    roots_this = None
    cv_used: Optional[Tuple[int, ...]] = None
    for trial in range(1, max_trials + 1):
        cv = [int(v) for v in rng.integers(-9, 10, size=len(A_int))]
        if all(c == 0 for c in cv):
            continue
        Mf = np.zeros((r_, r_))
        for j, Tj in enumerate(bb.tables):
            Mf += cv[j] * Tj
        w_eig, V = sla.eig(Mf.T)              # columns of V: left eigenvectors of Mf
        cand = []
        ok = True
        for lam_e, vec in zip(w_eig, V.T):    # rows of V^T transpose: row k is the left evec
            if abs(lam_e.imag) > tol * max(1.0, abs(lam_e)):
                continue
            got = values_from_w(vec.astype(float))
            if got is None:
                ok = False
                break
            cand.append((float(np.real(lam_e)), got))
        out.trials = trial
        if not ok or len(cand) != r_:
            continue
        evs = [a for a, _ in cand]
        if len(set(round(v, 6) for v in evs)) != r_:   # distinct-eigenvalue genericity (Thm A.4)
            continue
        roots_this = cand
        cv_used = tuple(cv)
        break

    if roots_this is None or cv_used is None:
        raise RuntimeError(
            "no generic Stickelberger functional found after %d trials (r_=%d); "
            "increase max_trials" % (max_trials, r_))

    # --- per-root dicts + BWE (eq. 12), evaluating moments via generator values
    def make_lam_m(gens):
        cache: Dict[Tuple[int, ...], float] = {}
        def lam_m(beta):
            if beta in cache:
                return cache[beta]
            rep = _semigroup_decomposition(beta, A_int)
            if rep is None:
                raise ValueError("exponent %s not in semigroup <A>" % (beta,))
            v = 1.0
            for i, n in enumerate(rep):
                if n:
                    v *= float(gens[i]) ** n
            cache[beta] = v
            return v
        return lam_m

    supp = [_poly_support_A(f) for f in fs]
    for a, gens in roots_this:
        out.roots.append({'z%d' % (i + 1): g for i, g in enumerate(gens)})
        lmv = make_lam_m(gens)
        comps = []
        if supp:
            for s_ in supp:
                num = 0.0
                den = 1.0
                for beta, c in s_.items():
                    v = lmv(beta)
                    num += c * v
                    den += abs(c * v)
                comps.append(abs(num) / den)
            out.bwe.append(sum(comps) / len(supp))
        else:
            out.bwe.append(0.0)
        out.bwe_components.append(comps)

    order = sorted(range(len(roots_this)), key=lambda i: roots_this[i][0])
    out.roots = [out.roots[i] for i in order]
    out.bwe = [out.bwe[i] for i in order]
    out.bwe_components = [out.bwe_components[i] for i in order]
    out.eigenvalues = [roots_this[i][0] for i in order]
    out.coefficients = cv_used
    return out


def solve_sparse_real_roots(f_system: Sequence[sp.Basic],
                            A: Optional[Sequence[Sequence[int]]] = None,
                            t0: Optional[int] = None, tol: float = 1e-6,
                            max_iterations: int = 8, seed: int = 2026,
                            solver=None) -> SparseRootsResult:
    """Top-level pipeline: Algorithm 1 (S51.2) + sparse border basis (S51.3) +
    Stickelberger recovery (S51.4).

    Returns a :class:`SparseRootsResult` with one dict per real toric root of the
    system in the semigroup generators z_i = x^{a_i}, backward-error diagnostics
    (eq. 12), and the J-generators from ker M_{s_star}.  When K_t is infeasible
    (Lemma 3.29(i): no real toric solution) an empty result is returned with
    ``no_real_roots`` recorded on it via ``roots == []`` and a non-empty kernel
    list left untouched.

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> out = solve_sparse_real_roots([x*(x-1)*(x+1)], A=[(1,)])
    >>> sorted(round(v['z1'], 6) for v in out.roots)
    [-1.0, 0.0, 1.0]
    """
    if A is None:
        free = sorted(set().union(*[set(sp.sympify(f).free_symbols) for f in f_system]),
                      key=str)
        if len(free) != 1:
            raise ValueError("A cannot be inferred from %d variables; pass it "
                             "explicitly" % len(free))
        A_int = ((1,),)
    else:
        A_int = _as_exponents(A)

    result = solve_algorithm1(f_system, A=list(A_int), t0=t0, tol=tol,
                              max_iterations=max_iterations, solver=solver)
    if not result.converged or result.rank_test is None or result.lambda_vec is None:
        return SparseRootsResult(kernel_polys=list(result.kernel_polys))
    return recover_all_sparse_roots(result, A_int, f_system=f_system, seed=seed)
