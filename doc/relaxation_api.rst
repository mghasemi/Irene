========================================
Unified Relaxation API
========================================

The ``relaxation_api.py`` module provides a unified interface for running SOS, SONC,
and SOSONC relaxations through a single engine class. It wraps the individual
relaxation modules behind a consistent API and offers convenience functions for
comparing multiple relaxation methods side by side.

.. contents::
   :local:
   :depth: 2

RelaxationEngine
================

The ``RelaxationEngine`` class is the primary entry point for running relaxations
with configurable reduction pipelines:

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem
   from Irene.relaxation_api import RelaxationEngine
   from Irene.relaxations import RelaxationConfig

   # Build problem
   sg = CommutativeSemigroup(['x', 'y'])
   sga = SemigroupAlgebra(sg)
   x, y = sga['x'], sga['y']

   prog = OptimizationProblem(sga)
   prog.set_objective(x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2)

   # Configure and run
   config = RelaxationConfig(
       reduction_method="newton_polytope",
       monomial_pruning=True,
       sparsity_detection=True,
   )

   engine = RelaxationEngine(prog, order=2, solver="clarabel", config=config)
   result = engine.solve("sos")  # or "sonc" or "sosonc"
   print(f"SOS bound: {result.value:.8f}")

Constructor Parameters
----------------------

- **prog** (OptimizationProblem): The problem to relax
- **order** (int): Relaxation order :math:`t`
- **solver** (str, optional): Solver backend — ``"clarabel"`` (default), ``"cvxopt"``, ``"dsdp"``
- **verbosity** (int, optional): Output level 0–2 (default: 1)
- **config** (RelaxationConfig, optional): Reduction pipeline configuration

The ``solve()`` Method
----------------------

.. code-block:: python

   result = engine.solve(method="sos")

Accepts ``method`` as one of:

- **``"sos"``**: Sum-of-squares relaxation via moment matrix PSD constraints
- **``"sonc"``**: SONC relaxation via geometric programming
- **``"sosonc"``**: Combined SOS+SONC two-step optimization

Returns a result object with attributes:

- **value** (float): Lower bound on the global minimum
- **status** (str): Solver status (``"optimal"``, ``"infeasible"``, etc.)
- **order** (int): Relaxation order used
- **basis_size** (int): Number of moment variables after pruning

RelaxationConfig
================

The ``RelaxationConfig`` dataclass controls the reduction pipeline:

.. code-block:: python

   from Irene.relaxations import RelaxationConfig

   config = RelaxationConfig(
       reduction_method="newton_polytope",  # "none" | "border_basis" | "newton_polytope" | "sparsity"
       monomial_pruning=False,              # Enable/disable pruning
       sparsity_detection=False,            # Auto-detect correlative sparsity
       quotient_basis="groebner",           # "groebner" | "border" -- reduction engine
       verbose_reduction=False,             # Print reduction diagnostics
   )

Fields:

- **reduction_method** (str): Basis reduction strategy. ``"none"`` uses the full monomial basis; ``"border_basis"`` applies border basis reduction; ``"newton_polytope"`` prunes via Newton polytope geometry; ``"sparsity"`` decomposes into independent SDP blocks.
- **monomial_pruning** (bool): Whether to apply monomial-level pruning within the chosen method
- **sparsity_detection** (bool): Whether to auto-detect and exploit correlative sparsity
- **quotient_basis** (str): Quotient-ring reduction engine used by ``ReduceExp`` and ``ReducedMonomialBase``. ``"groebner"`` (default) uses the classical SymPy Groebner-basis reduction — the behavior of original Irene; ``"border"`` uses IreneRewrite's ``BorderBasis`` quotient-algebra reduction (numerically computed multiplication tables). The environment variable ``IRENE_QUOTIENT_BASIS=groebner|border`` sets the default when no explicit config is passed.
- **verbose_reduction** (bool): Print detailed reduction diagnostics during construction

Two-Stage Hybrid Monoid-Graph Reduction Theorem
================================================

The real power of IreneRewrite's reduction pipeline lies in the **synergistic
combination** of algebraic quotienting and structural graph decomposition.  When
all three reduction flags are enabled, the engine applies a two-stage reduction
that composes monoid-theoretic elimination with chordal-graph decomposition.

.. admonition:: Theorem (Hybrid Monoid-Graph Reduction)
   :class: note

   Let :math:`\mathcal{F} = \{f_0, f_1, \dots, f_m\} \subset \mathbb{R}[S]` be
   a polynomial/differential system with ideal :math:`\mathcal{I} =
   \langle\mathcal{F}\rangle`, and let :math:`t` be the relaxation order.

   **Stage 1 — Inner Algebraic Reduction (Monoid Quotienting).**
   The quotient basis at degree :math:`2t` is

   .. math::

      B_{2t} = \operatorname{supp}\!\big(\mathbb{R}[S] / \mathcal{I}_{\le 2t}\big),

   computed via the selected ``quotient_basis`` engine (Gröbner or Border).
   The Newton polytope pruner further restricts this to

   .. math::

      B_{2t}^{\text{pruned}} = B_{2t} \cap (2t \cdot \operatorname{New}(\mathcal{F})).

   **Stage 2 — Outer Structural Reduction (Chordal Graph Decomposition).**
   Construct the correlative sparsity graph :math:`G = (V, E)` on the vertex set
   :math:`V = B_t^{\text{pruned}} \cap \Theta_{\le t}Y`.  After chordal
   completion, extract maximal cliques :math:`\{C_1, \dots, C_p\}` satisfying
   the running intersection property.  Each clique defines a local sub-basis

   .. math::

      B_{t,k} = \{\alpha \in B_t^{\text{pruned}} : \operatorname{supp}(\alpha) \subseteq C_k\},

   and the dense PSD constraint :math:`M_{B_t}(y) \succeq 0` is replaced by
   :math:`p` coupled, smaller PSD blocks:

   .. math::

      M_{B_{t,k}}(y) \succeq 0, \qquad k = 1, \dots, p.

   The reduction factors are multiplicative: if Newton pruning yields a factor
   :math:`r_N` and chordal decomposition yields :math:`r_C`, the total moment
   matrix dimension is reduced by :math:`\approx r_N \cdot r_C`.

The pipeline is illustrated below::

   Polynomial / Differential System  F
                    |
                    v
   [Inner Step] Monoid Quotienting (Border / Groebner)
       B_{2t} = supp( R[S] / I_{<=2t} )
                    |
                    v
   [Newton Pruning]  B_{2t} cap (2t * New(F))
                    |
                    v
   [Outer Step] Correlative Sparsity Graph G
       Vertices V = Pruned Basis
                    |
                    v
   Chordal Completion & Clique Extraction  C_1, ..., C_p
                    |
                    v
   Coupled Local Moment Blocks:  M_{B_{t,k}}(y) >= 0

Configuration
-------------

All three reductions are activated simultaneously through ``RelaxationConfig``:

.. code-block:: python

   from Irene.relaxations import RelaxationConfig
   from Irene.relaxation_api import RelaxationEngine

   config = RelaxationConfig(
       reduction_method="border_basis",      # or "groebner"
       quotient_basis="border",              # quotient-ring engine
       monomial_pruning=True,                # enable Newton polytope pruning
       sparsity_detection=True,              # enable correlative sparsity
       verbose_reduction=True,
   )
   engine = RelaxationEngine(prog, order=2, config=config)

When ``verbose_reduction=True``, the engine reports the combined effect::

   Reduction pipeline (order 2):
     Step 1 - Border basis:    28 → 22 monomials (1.27×)
     Step 2 - Newton pruning:  22 → 17 monomials (1.29×)  [cumulative 1.65×]
     Step 3 - Chordal decomp:  3 cliques, mean block size 6.3
       Estimated SDP speedup: ~8.4×

compare_all()
=============

The ``compare_all`` convenience function runs SOS, SONC, and SOSONC relaxations on the same problem and returns a comparison table:

.. code-block:: python

   from Irene.relaxation_api import compare_all

   results = compare_all(prog, order=2, solver="clarabel")
   # Returns dict with 'sos', 'sonc', 'sosonc' keys, each containing value/status/timing

This is useful for quick benchmarking and method comparison without writing separate
engine instances.
