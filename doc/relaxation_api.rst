========================================
Unified Relaxation API
========================================

The ``relaxation_api.py`` module provides a unified interface for running SOS, SONC,
and SOSONC relaxations through a single engine class. It wraps the individual
relaxation modules behind a consistent API and offers convenience functions for
comparing multiple relaxation methods side by side.

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
- **solver** (str, optional): Solver backend — ``"cvxopt"`` (default), ``"clarabel"``, ``"dsdp"``. Clarabel and CVXOPT are the primary backends for SDP relaxations; Clarabel is recommended on instances where structural feasibility matters (e.g., Motzkin / Choi–Lam at relaxation order :math:`\geq` 3, where some backends misreport infeasibility).
- **verbosity** (int, optional): Output level 0–2 (default: 1)
- **config** (RelaxationConfig, optional): Reduction pipeline configuration

The ``solve()`` Method
----------------------

.. code-block:: python

   result = engine.solve(method="sos")

Accepts ``method`` as one of:

- **``"sos"``**: Sum-of-squares relaxation via moment matrix PSD constraints
- **``"sonc"``**: SONC relaxation via geometric programming
- **``"sosonc_sos_first"``**: Combined SOS+SONC optimization, solving SOS first
- **``"sosonc_sonc_first"``**: Combined SOS+SONC optimization, solving SONC first

Returns a :class:`RelaxResult` with attributes:

- **value** (float): Lower bound on the global minimum (``-inf`` when the solver reports infeasibility)
- **status** (str): ``"optimal"``, ``"infeasible"``, or ``"error"``
- **method** (str): The relaxation method that was run
- **runtime**, **init_time** (float): Wall-clock seconds for solve and setup phases
- **certificate**: SOS certificate when available (SDP path)
- **solver_info** (dict): Backend diagnostics; on the SDP path includes the relaxation order actually used under ``"order"`` plus the raw solver status string

The unified engine does not expose a moment-basis size in this contract: basis-size telemetry comes from the reduction pipeline (set ``verbose_reduction=True`` to print measured lines such as ``[SDPRlx] Newton pruning: 28 -> 28``) or from the legacy ``SDPRelaxations`` interface documented in ``sdp.rst``, whose result dict includes ``basis_size``.

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
- **quotient_basis** (str): Quotient-ring reduction engine used by ``ReduceExp`` and ``ReducedMonomialBase``. ``"groebner"`` (default) uses the classical SymPy Groebner-basis reduction — the behavior of original Irene; ``"border"`` uses Irene's ``BorderBasis`` quotient-algebra reduction (numerically computed multiplication tables). The environment variable ``IRENE_QUOTIENT_BASIS=groebner|border`` sets the default when no explicit config is passed.
- **verbose_reduction** (bool): Print detailed reduction diagnostics during construction

By default, reductions are disabled for compatibility. Enable them explicitly with
``reduction_method`` or the corresponding boolean flags.

Two-Stage Hybrid Monoid-Graph Reduction Theorem
================================================

The real power of Irene's reduction pipeline lies in the **synergistic
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

When ``verbose_reduction=True``, the engine prints one line per reduction stage
that actually ran.  For example, on Motzkin at order 3 with the Newton-polytope
pipeline (``reduction_method="newton_polytope"``, ``monomial_pruning=True``,
``sparsity_detection=True``) the measured output is::

   [SDPRlx] Newton pruning: 28 -> 28 (100.00% retained)
   [SDPRlx] Newton pruning: 10 -> 10 (100.00% retained)

The two lines are the moment-matrix basis (degree :math:`2t = 6`, 28 monomials)
and the order-:math:`t` basis (degree 3, 10 monomials).  The border-basis stage
is a no-op on this instance: Motzkin has no free relations, so the quotient ring
is the full polynomial ring and no border basis is built.  Correlative-sparsity
decomposition likewise does not apply here: every non-constant term of Motzkin
couples both variables, so the sparsity graph is a single component covering
the whole basis and the moment matrix stays dense.  Newton pruning itself is
also a designed no-op on this dense instance -- until a certified SOS-support
reduction exists, the pipeline retains the full degree-bounded support as
``required_support``, which makes the relaxed SDP identical to the unpruned
baseline (bound ``-526.08114071``, status ``optimal`` at orders 1--3).
Measured runtimes in ``benchmarks/results/p3_vs_baseline.json`` give speedups of
0.69x--1.32x across the three test problems and all orders; no larger speedup
is claimed on instances of this size.

compare_all()
=============

The ``compare_all`` convenience function runs SOS, SONC, and SOSONC relaxations on the same problem and returns a comparison table:

.. code-block:: python

   from Irene.relaxation_api import compare_all

   results = compare_all(prog, order=2, solver="clarabel")
   # Returns dict with 'sos', 'sonc', 'sosonc' keys, each containing value/status/timing

This is useful for quick benchmarking and method comparison without writing separate
engine instances.
