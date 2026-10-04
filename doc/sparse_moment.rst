========================================
Sparse Moment Method (SparseMomentSDP)
========================================

The ``sparse_moment.py`` module implements the **sparse moment method** of
Baumbach & Bender [BB2026]_ for computing the *real* points of sparse polynomial
systems directly on an affine toric variety, without first solving the (larger)
complex system.  It is Irene's toric extension of Lasserre–Laurent–Rostalski
moment-SDP real-radical computation to the semigroup ring :math:`R[A]`.

Theory
======

For a finite exponent set :math:`A = \{\alpha_1, \dots, \alpha_r\} \subset
\mathbb{N}^m`, the semigroup ring :math:`R[A] = \mathbb{R}[x^{\alpha_1}, \dots,
x^{\alpha_r}] \subset \mathbb{R}[x_1, \dots, x_m]` is the coordinate ring of an
affine toric variety.  A linear functional :math:`\Lambda\in R[A]^*` is
*positive* when its moment matrix

.. math::

   M_s(\Lambda) := \big(\Lambda(x^{\alpha}x^{\beta})\big)_{\alpha,\beta \in A_s}

is positive semidefinite, where :math:`A_s = \{\sum_{i\le s}\alpha_i\}` is the
:math:`s`-truncation.  For a system :math:`I = \langle f_1, \dots, f_\mu\rangle
\subset R[A]` with finite real toric variety, the truncated spectrahedron

.. math::

   K_t = \{\Lambda \in R[A]^*_t : \Lambda(1)=1,\;
   M_{\lfloor t/2\rfloor}(\Lambda)\succeq 0,\; \Lambda(h)=0\;\forall h\in H_t\}

(where :math:`H_t` is the set of prolongations :math:`x^{\beta}f_i`, eq. (9) of
the paper) is a polytope whose vertices are the evaluations at the real toric
points (paper Theorem 3.15).  A *generic* element :math:`\Lambda\in
\operatorname{relint}(K_t)` has a kernel ideal that eventually equals the real
radical :math:`\sqrt[\scriptstyle R]{I}` (Lemma 3.14, Theorem 3.26), and the flat
extension theorem (paper Theorem 3.17) plus its rank conditions (Theorem 3.27)
give a terminating algorithm:

1. Solve :math:`\min_{\Lambda\in K_t} \Lambda(1)` — a feasibility SDP
   (eq. (11)); a generic optimizer is a boundary-free point giving the real
   radical.
2. Test the flatness rank conditions :math:`\operatorname{rank} M_s =
   \operatorname{rank} M_{s-1}` for admissible :math:`s`; if none holds,
   increment the truncation :math:`t` and re-solve.
3. On convergence, read the real toric points off the sparse Stickelberger
   eigenvalue problem on a connected-to-one border basis of
   :math:`R[A]/\sqrt[\scriptstyle R]{I}`.

Two details distinguish the sparse from the dense moment method and are enforced
by this module:

- **Toric relation bound :math:`\rho_A`** (paper eq. (3)): the flat extension
  theorem requires :math:`s \ge \rho_A`, where :math:`\rho_A = \max\deg(g)` over
  generators :math:`G_A` of the toric ideal :math:`T_A = \ker\psi_A`.  Without
  this, one can have :math:`\operatorname{rank}M_1 = \operatorname{rank}M_0 = 1`
  yet no rank-one flat extension (paper Example 3.18 in :math:`R[x^2,x^3]`,
  where :math:`1^3 \ne 2^2`).
- **Connected-to-one border basis** (paper Appendix A, Remark 3.28): the
  Stickelberger functional values are recovered by choosing a monomial column
  basis of :math:`M_{s-1}` greedily in increasing :math:`\deg_A` order, which is
  a border basis connected to one.

Reference

.. [BB2026] T. Baumbach, M. Bender, *The Moment Method for Computing Real Points
   of Sparse Polynomial Systems*, arXiv:2609.33313 [cs.SC], 2026.

Pipeline
========

1. ``ToricSetup.from_exponents`` computes the toric ideal :math:`T_A`, its
   generators :math:`G_A`, and the bound :math:`\rho_A` (paper equation (3)).
2. ``moment_indices(A, t)`` enumerates :math:`R[A]^t` (smaller than the dense
   monomial set) and :math:`SparseMomentSDP` builds :math:`K_t` (eq. (10)) as an SDP.
3. ``solve_algorithm1`` runs the Algorithm-1 termination loop (Theorem 3.27 rank
   conditions; :math:`K_t = \emptyset` certifies "no real toric solution" via
   Lemma 3.29).
4. ``SparseBorderBasis`` builds the connected-to-one border basis and its
   multiplication tables; ``recover_all_sparse_roots`` solves the sparse
   Stickelberger eigenvalue problem and returns the real toric points.

Quick Example
=============

.. code-block:: python

   import sympy as sp
   from Irene.sparse_moment import solve_sparse_real_roots

   x = sp.Symbol("x")

   # Real roots of x^3 - x on the dense line R[x].
   out = solve_sparse_real_roots([x**3 - x], A=[(1,)])
   print(sorted(v["z1"] for v in out.roots))     # [-1.0, 0.0, 1.0]

   # Sparse form: roots of (x^2 - 1)(x^2 - 4) = x^4 - 5x^2 + 4 in
   # R[A] = R[x^2, x^3] (A = {(2,0),(3,0)}).  Only the two generators needed.
   out = solve_sparse_real_roots([x**4 - 5*x**2 + 4], A=[(2, 0), (3, 0)])
   print(sorted(v["z1"] for v in out.roots))     # [-2.0, -1.0, 1.0, 2.0]

The first is the dense/univariate case; the second exercises the toric relations
:math:`(x^2)^3 = (x^3)^2` (= :math:`x^6`) so the moment matrices are sized by
exponents of :math:`R[A]` rather than the full ring.

Public API
==========

- ``ToricSetup.from_exponents(A)`` — semigroup ring :math:`R[A]`, toric
  generators :math:`G_A`, and :math:`\rho_A` (eq. (3)).
- ``toric_ideal(A)`` / ``deg_A(f, A)`` — toric ideal :math:`T_A` (via
  Sturmfels' elimination) and the A-graduation.
- ``moment_indices(A, t)`` / ``prolongations(f_system, A, t, index)`` — basis of
  :math:`R[A]^t` and the prolongation set :math:`H_t` (eq. (9)).
- ``SparseMomentSDP(f_system, A, t, objective, ...)`` — the spectrahedron
  :math:`K_t` (eq. (10)) as a CVXPY-based SDP; ``objective='constant'`` realizes
  Algorithm 1, a random Gaussian vector realizes Algorithm 2 (single real root).
- ``test_theorem_327(...)`` — the rank conditions of Theorem 3.27.
- ``solve_algorithm1(f_system, A, t0, tol, solver)`` — the full termination loop;
  returns ``Algorithm1Result`` (real radical generators, quotient basis, rank
  profile).
- ``SparseBorderBasis(...)`` — connected-to-one border basis of
  :math:`R[A]/\sqrt[\scriptstyle R]{I}` with multiplication tables and
  ``check_relations`` residual.
- ``recover_all_sparse_roots(result, A, f_system, seed, tol)`` — sparse
  Stickelberger recovery of all real toric points with per-root backward error
  (paper eq. (12)).
- ``solve_sparse_real_roots(f_system, A, ...)`` — top-level pipeline
  (Algorithm 1 + border basis + Stickelberger).

Numerical Notes
===============

- First-order SDP solvers may return a boundary point of :math:`K_t`, making
  both the rank test and the greedy border-basis selection tolerance-sensitive.
  The module's border-basis selector enforces connected-to-one (Def A.2) and an
  explicit SVD rank test on the rectangular column matrix (rather than a squared
  Gram test), and normal forms are pinned to the flat-extension range
  :math:`2s^* - \deg_A(e)` — the moments genuinely fixed by Theorem 3.17.
- When a real root lies on the toric variety boundary (a generator value 0, e.g.
  paper Example 3.16's point :math:`(0,1)`), the recovered values still match the
  asserted ideal; the real radical :math:`\; I = \langle x^{\alpha_1+\alpha_2},
  x^{\alpha_1+\alpha_2}-x^{\alpha_1}-x^{\alpha_2}+1 \rangle` yields the two
  toric points :math:`\{(0,1),(1,0)\}`.

Integration Notes
=================

- The module is an independent solver on the *support structure* of the system;
  it complements (rather than routes through) ``SDPRelaxations``.  Where the
  dense Lasserre hierarchy keeps every monomial up to degree :math:`t`, the
  sparse moment method keeps only the monomials in :math:`R[A]^t`, which is
  often far smaller.
- `scripts/verify_article_examples.py` reproduces the paper's concrete claims
  (Examples 3.8, 3.16, 3.18, 4.8, 4.13) against this implementation — run it
  with the project venv to confirm the numerical statements in this page.
