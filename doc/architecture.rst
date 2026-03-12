=================================
Architecture and Method Guide
=================================

The current codebase supports multiple constrained polynomial optimization pathways.
Historically, the documentation emphasized SDP hierarchies. The present structure
extends this to include geometric and SONC methods on the same algebraic backbone.

From Polynomial Expressions to Group-Rings
==========================================

Instead of treating input only as classical polynomial expressions, Irene builds
optimization models through commutative semigroups and their semigroup algebras.
This representation supports:

1. Symbolic manipulation of monomial support and exponents.
2. Geometric operations on Newton polytopes.
3. Differential operators through user-provided derivation maps.

The shift in perspective is important: instead of viewing optimization only as
operations in :math:`\mathbb{R}[x_1,\dots,x_n]`, Irene treats expressions as elements
of a semigroup algebra that can be converted to geometric data (supports,
exponent tuples, convex combinations) used directly in GP and SONC routines.

Module Layers
=================================

1. Algebra layer: ``grouprings.py``.
2. Problem layer: ``program.py``.
3. Relaxation layer: ``relaxations.py``, ``geometric.py``, ``sonc.py``.
4. Solver layer: ``sdp.py`` and external solver backends.

Choosing a Relaxation Family
=================================

Use SDP relaxations when you need a moment/SOS hierarchy and semidefinite certificates.
Use geometric relaxations when the transformed formulation is naturally handled by GP.
Use SONC relaxations for sparse circuit-structured formulations in constrained settings.

.. list-table:: Method Selection Snapshot
	 :header-rows: 1

	 * - Method
		 - Primary object
		 - Typical strength
		 - Main module
	 * - SDP hierarchy
		 - Moment and localizing matrices
		 - Certified lower bounds via SOS/moment conditions
		 - ``relaxations.py`` and ``sdp.py``
	 * - Geometric relaxation
		 - Posynomial/signomial GP model
		 - Scalable lower bounds for suitable transformed instances
		 - ``geometric.py``
	 * - SONC relaxation
		 - Circuit polynomial constraints
		 - Sparse constrained formulations with barycentric structure
		 - ``sonc.py``

Typical Workflow
=================================

1. Define algebra generators and build expressions in a semigroup algebra.
2. Form an optimization problem with objective and constraints.
3. Select relaxation family (SDP, GP, or SONC).
4. Solve and interpret a certified lower bound.
