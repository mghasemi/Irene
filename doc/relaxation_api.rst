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

compare_all()
=============

The ``compare_all`` convenience function runs SOS, SONC, and SOSONC relaxations on the same problem and returns a comparison table:

.. code-block:: python

   from Irene.relaxation_api import compare_all

   results = compare_all(prog, order=2, solver="clarabel")
   # Returns dict with 'sos', 'sonc', 'sosonc' keys, each containing value/status/timing

This is useful for quick benchmarking and method comparison without writing separate
engine instances.
