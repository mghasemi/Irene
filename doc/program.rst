=================================
Problem Representation
=================================

The class ``OptimizationProblem`` in ``program.py`` is the bridge between
algebraic expressions and concrete relaxation models.

Core Responsibilities
=================================

1. Store objective and constraints as semigroup-algebra expressions.
2. Track degrees used by relaxation constructors.
3. Provide support analysis utilities used by geometric and SONC modules.

Key Methods
=================================

``set_objective``
   Registers the objective and updates degree metadata.

``add_constraints``
   Adds constrained expressions and updates per-constraint degree metadata.

``delta``
   Extracts terms relevant for nonnegativity and circuit-based conditions.

``mono2ord_tuple`` and ``tuple2mono``
   Convert symbolic monomials to numeric exponent tuples and back.

``newton``, ``convex_combination``, and related helpers
   Support Newton-polytope and barycentric calculations required by SONC/GP paths.

From Equations to Methods
=================================

This chapter connects internal utilities in ``OptimizationProblem`` to the
equation-level objects used in :doc:`geometric` and :doc:`sonc`.

.. csv-table::
    :header: "Mathematical role", "OptimizationProblem method", "Where it appears later"

    "Build active support and delta terms", "``delta``", "Section 4 GP construction and Section 3 constrained SONC families"
    "Move between symbolic monomials and exponent tuples", "``mono2ord_tuple`` and ``tuple2mono``", "Newton-polytope geometry and constraint assembly in GP and SONC"
    "Compute support geometry", "``newton``", "Support-point extraction for SONC and transformed GP structures"
    "Compute barycentric coefficients", "``convex_combination`` and ``linear_combination``", "SONC weights :math:`\lambda^{(\beta)}` and related support relations"

In practice, these methods are the bridge from symbolic semigroup-algebra input
to the numeric data structures used by the geometric and SONC solvers.

Minimal Construction Pattern
=================================

::

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem

   S = CommutativeSemigroup(['x', 'y'])
   SA = SemigroupAlgebra(S)
   x = SA['x']
   y = SA['y']

   P = OptimizationProblem(SA)
   P.set_objective(1 + x**2 + y**2)
   P.add_constraints([1 - x**2, 1 - y**2])
