========================================
Newton Polytope Pruning
========================================

The ``newton_polytope.py`` module implements basis pruning via Newton polytope
geometry: it computes the convex hull of the exponent vectors of a polynomial
system, filters candidate monomials by membership in the scaled Newton body, and
always retains every exponent passed as ``required_support``.  The relaxation
pipeline currently passes the full degree-bounded support -- hull membership
alone does not certify that an exponent is removable from a Gram/moment basis --
so until a certified sparsity set exists, the reduction is a designed no-op (see
Theory).

Theory
======

Newton Polytopes of Polynomial Systems
--------------------------------------

The **Newton polytope** of a polynomial :math:`f = \sum_\alpha c_\alpha x^\alpha` is the
convex hull of its exponent vectors:

.. math::

   \text{New}(f) = \text{conv}\{\alpha \in \mathbb{N}^n : c_\alpha \neq 0\}.

For a system of polynomials :math:`F = \{f_0, f_1, \dots, f_m\}` (objective plus
constraints), the relevant geometry is captured by the **Minkowski sum**:

.. math::

   \text{New}(F) = \sum_{i=0}^m \text{New}(f_i).

At relaxation order :math:`t`, the moment matrix uses a monomial basis indexed by
exponents in :math:`\Lambda_t = \{\alpha : |\alpha| \leq t\}`. However, many of
these exponents may lie outside the scaled Newton body :math:`t \cdot \text{New}(F)`,
meaning they cannot contribute to valid certificates of nonnegativity for the
given problem structure.

Scaled Newton Bodies and Basis Pruning
--------------------------------------

The key theorem (see Parrilo 2000, Lasserre 2003) states that the moment matrix
can be restricted to exponents in:

.. math::

   \Lambda_t^{\text{pruned}} = \Lambda_t \cap t \cdot \text{New}(F) \cap \mathbb{N}^n.

This intersection removes monomials whose exponents lie outside the scaled Newton
body while preserving all monomials needed for valid SDP relaxations.  In
practice, however, hull membership alone is *not* a certificate that an exponent
may be removed from a Gram/moment basis (the order-3 Motzkin instance is a known
counterexample: hull-only over-pruning left the relaxed SDP with an unbounded
recession ray).  Until such a certificate exists, the relaxation pipeline retains
the full degree-bounded support via ``required_support``, which makes the prune a
designed no-op on dense instances; see "Integration with Relaxation Pipeline".

Dimension Reduction in Practice
-------------------------------

For a bivariate degree-6 problem like Motzkin (:math:`x^4 y^2 + x^2 y^4 - 3x^2 y^2 + 1`),
the full basis at order 3 has :math:`\binom{8}{2} = 28` elements.  The hull-only Newton
prune (no required support) reduces this to **10** monomials -- a factor of 2.8x -- because
18 exponents of total degree at most 6 lie outside :math:`2\,\text{New}(f)`, where
:math:`\text{New}(f) = \operatorname{conv}\{(0,0), (2,2), (2,4), (4,2)\}`.

In higher dimensions, the hull-only reduction factor grows exponentially with :math:`n`
for sparse polynomial systems, making Newton pruning one of the most impactful
optimizations there; the pipeline activates it as soon as a certified sparsity set
replaces today's full-support retention.

Minkowski Sum Computation
-------------------------

The implementation computes Minkowski sums via convex hull operations on the union
of translated exponent sets:

.. math::

   P \oplus Q = \text{conv}\{p + q : p \in P, q \in Q\}.

For efficiency, the sum is computed incrementally using the ``scipy.spatial.ConvexHull``
routine on the combined vertex set. The final scaled body is obtained by multiplying
all vertices by the relaxation order :math:`t`.

API Reference
=============

NewtonPruner Class
------------------

.. code-block:: python

   from Irene.newton_polytope import NewtonPruner, prune_basis_from_polys

   # Prune the monomial basis of a polynomial system at degree bound 6
   pruner = prune_basis_from_polys([objective, g1, g2], num_vars=2, max_degree=6)

``prune_basis_from_polys`` (and ``prune_basis_from_problem``) return a configured
:class:`NewtonPruner` with the pruned basis already computed.  Key attributes:

- **full_basis_size** (int): Number of monomials in the unpruned degree-bounded basis :math:`\Lambda_t`
- **pruned_basis_size** (int): Number of monomials after Newton polytope filtering, plus every exponent in ``required_support``
- **reduction_ratio** (float): Fraction of the basis retained (1.0 = nothing pruned; smaller means more pruning).  The reduction factor is its reciprocal, e.g. a ratio of :math:`10/28 \approx 0.36` corresponds to a factor of 2.8x.

Extra detail (``matrix_entry_reduction``, ``entries_saved``) is available via
``pruner.summary()`` or ``pruner.moment_matrix_dimension_reduction()``, and the exponent
tuples themselves via ``pruner.compute_pruned_basis()``.

Example Usage
-------------

The classic bivariate degree-6 test case is Motzkin's polynomial.  Built through
Irene's documented semigroup-algebra construction path, it prunes from the full
28-monomial basis to the 10 monomials inside the scaled Newton body:

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.newton_polytope import prune_basis_from_polys

   sg = CommutativeSemigroup(['x', 'y'])
   sga = SemigroupAlgebra(sg)
   x, y = sga['x'], sga['y']

   # Motzkin polynomial (total degree 6)
   motzkin = x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2

   pruner = prune_basis_from_polys([motzkin], num_vars=2, max_degree=6)
   print(pruner.full_basis_size)    # 28
   print(pruner.pruned_basis_size)  # 10
   print(pruner.reduction_ratio)    # ~0.357 (retained fraction; factor = 28/10 = 2.8x)

Plain SymPy input works identically: ``prune_basis_from_polys([f], num_vars=2,
max_degree=6)`` with :math:`f = x^4 y^2 + x^2 y^4 - 3x^2 y^2 + 1` returns the same
28 to 10 counts.  The exponent tuples are obtained from
``pruner.compute_pruned_basis()``; ``prune_basis_from_problem(prog, max_degree=6)``
accepts an :class:`~Irene.program.OptimizationProblem` directly and gives the same result.

Integration with Relaxation Pipeline
====================================

Newton polytope pruning is activated via the relaxation configuration:

.. code-block:: python

   from Irene.relaxations import RelaxationConfig
   from Irene.relaxation_api import RelaxationEngine

   config = RelaxationConfig(
       reduction_method="newton_polytope",
       monomial_pruning=True,
       verbose_reduction=True,
   )
   engine = RelaxationEngine(prog, order=3, config=config)
   result = engine.solve("sos")

When ``verbose_reduction=True``, the pipeline prints one measured line per monomial
basis it builds -- e.g., on Motzkin at order 3::

   [SDPRlx] Newton pruning: 28 -> 28 (100.00% retained)

The standalone hull-only call in "Example Usage" above reduces the same basis from 28 to
10, but the relaxation pipeline passes the full degree-bounded support as
``required_support`` (certified-safe retention): until a certified SOS-support reduction
exists, hull membership alone does not justify dropping an exponent from the Gram/moment
basis, so on dense instances like Motzkin the pruned basis is identical to the unpruned
one by design -- and the relaxed SDP (bound ``-526.08114071``) matches the baseline at
every order.

Practical Notes
===============

1. Newton pruning is **most effective** for problems where the objective and constraints have sparse support relative to their degree
2. The pruning step adds negligible overhead (~milliseconds) compared to SDP solve times (seconds to minutes)
3. For fully dense polynomials, the reduction factor approaches 1x (no pruning benefit), but correctness is preserved
4. Newton pruning and correlative sparsity are **complementary** — they can be applied together for multiplicative reduction effects

References
==========

- Parrilo, P. A. (2000). "Structured semidefinite programs and semialgebraic geometry methods in robustness and optimization." *Caltech PhD Thesis*.
- Lasserre, J.-B. (2003). "Moment inequalities, and a global optimization approach to polynomial optimization with sparse polynomials." *SIAM Journal on Optimization*, 17(4), 822–843.
- Iliman, S. & de Wolff, T. (2016). "Lower Bounds for Polynomials with Simplex Newton Polytopes Based on Geometric Programming." *SIAM Journal on Optimization*, 26(2), 1128–1146. DOI: 10.1137/140962425.
