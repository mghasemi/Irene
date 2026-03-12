=================================
Group-Ring Foundations
=================================

This chapter introduces the algebraic layer behind the optimization modules.

Algebraic Setting
=================================

Let :math:`S` be a finitely generated commutative semigroup and let
:math:`\mathbb{R}[S]` denote its semigroup algebra. In Irene, elements of
:math:`\mathbb{R}[S]` are represented as finite sums

.. math::

   f = \sum_{\alpha \in \mathrm{supp}(f)} c_\alpha \alpha,

where :math:`\alpha` is a semigroup element and :math:`c_\alpha \in \mathbb{R}`.
This perspective generalizes classical polynomial notation and keeps support,
degree, and structural operations explicit for optimization routines.

When relations are present, the algebra is effectively handled modulo those
relations through reduction in the semigroup representation. This is useful for
modeling quotient structures that arise naturally in symbolic formulations.

Commutative Semigroup and Semigroup Algebra
===========================================

The module ``grouprings.py`` provides:

1. ``CommutativeSemigroup``: generator-based semigroup representation.
2. ``SemigroupAlgebra``: algebra construction over that semigroup.
3. ``AtomicSGElement`` and ``SemigroupAlgebraElement``: monomial and polynomial-like elements.

This representation is used by higher layers to extract supports, exponents,
and structural information required by geometric and SONC relaxations.

Support and Geometry
=================================

The support of :math:`f` is central in both geometric and SONC constructions.
By converting semigroup monomials to exponent tuples, Irene can compute Newton
polytope information and barycentric relations directly from algebraic input.

This is the key bridge from symbolic algebra to convex-geometric objects used
in lower-bound certificates.

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

From a theoretical viewpoint, derivations are linear maps
:math:`D: \mathbb{R}[S] \to \mathbb{R}[S]` that satisfy Leibniz rules. Irene's
``add_derivative`` and ``diff`` pipeline implements this behavior directly on
semigroup-algebra elements.

Why This Matters for POP
=================================

The shift from polynomial-only notation to group-ring structures improves the
connection between symbolic representation and geometric computation:

1. Exponent vectors are directly accessible for convex hull and delta-set routines.
2. Algebraic reductions remain explicit and programmatic.
3. The same representation supports SDP, GP, and SONC method families.
