=================================
Group-Ring Foundations
=================================

This chapter introduces the algebraic layer behind the optimization modules.

Commutative Semigroup and Semigroup Algebra
===========================================

The module ``grouprings.py`` provides:

1. ``CommutativeSemigroup``: generator-based semigroup representation.
2. ``SemigroupAlgebra``: algebra construction over that semigroup.
3. ``AtomicSGElement`` and ``SemigroupAlgebraElement``: monomial and polynomial-like elements.

This representation is used by higher layers to extract supports, exponents,
and structural information required by geometric and SONC relaxations.

Differential Operators
=================================

Beyond static algebraic representation, ``SemigroupAlgebra`` supports derivations.
Given a base map, derivatives are propagated through expressions using product-rule behavior.

For factors :math:`u` and :math:`v`, the implementation follows:

.. math::

   D(uv) = D(u)v + uD(v).

This enables a workflow where optimization is not only over algebraic objects,
but also over structures enriched with differential operators.

In code, the derivative path is organized as:

1. ``SemigroupAlgebra.add_derivative`` registers a derivation map.
2. ``SemigroupAlgebra.derivative`` selects a registered derivation.
3. ``SemigroupAlgebra.diff`` applies recursive product-rule expansion.

This method-level design makes differentiation explicit and extensible for
problem formulations where algebraic structure and operator behavior are coupled.

Why This Matters for POP
=================================

The shift from polynomial-only notation to group-ring structures improves the
connection between symbolic representation and geometric computation:

1. Exponent vectors are directly accessible for convex hull and delta-set routines.
2. Algebraic reductions remain explicit and programmatic.
3. The same representation supports SDP, GP, and SONC method families.
