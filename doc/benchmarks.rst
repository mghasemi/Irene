========================================
Benchmarks and Performance Evaluation
========================================

This chapter documents the benchmarking infrastructure for IreneRewrite, including
the problem gallery system, performance comparison scripts, and representative
examples using the current API.

.. contents::
   :local:
   :depth: 2

Benchmark Problem Gallery
=========================

The ``benchmarks/gallery.yaml`` file defines a structured catalog of polynomial
optimization problems with known properties, expected relaxation results, and
metadata for filtering. Each entry specifies:

- **id**: Unique string identifier (e.g. ``motzkin``, ``choi_lam``)
- **name**: Human-readable name
- **description**: Mathematical description of the problem
- **variables**: List of variable names
- **degree**: Total degree of objective polynomial(s)
- **category**: One of ``unconstrained``, ``constrained``, ``separating``, ``mean_poly``
- **objective**: Polynomial expression in SymPy-compatible syntax
- **constraints**: Optional list of constraint dicts with ``expr`` and ``type`` keys
- **true_min**: Known global minimum (or ``null`` if unknown)
- **relaxations**: Expected relaxation results at various orders
- **tags**: Keywords for filtering

Gallery Runner
--------------

The ``benchmarks/run_gallery.py`` script loads the gallery, constructs each problem
via Irene's API, runs SOS/SONC/SOSONC relaxations at specified orders, and records
structured JSON results for regression tracking:

.. code-block:: bash

   # Run full gallery with default CLARABEL solver
   python benchmarks/run_gallery.py

   # Filter by tag, use SCS solver, custom tolerance
   python benchmarks/run_gallery.py --filter separating --solver scs --tolerance 1e-4

   # Custom output directory and timeout
   python benchmarks/run_gallery.py --output-dir ./results/ --timeout 600

The runner constructs problems using the semigroup algebra pattern:

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem

   sg = CommutativeSemigroup(variables)
   sga = SemigroupAlgebra(sg)
   sym_dict = {v: sga[v] for v in variables}

   objective = eval(obj_expr, {"__builtins__": {}}, sym_dict)
   prog = OptimizationProblem(sga)
   prog.set_objective(objective)

Representative Gallery Problems
-------------------------------

**Motzkin Polynomial**: The canonical separating example. Nonnegative by AM-GM but
not SOS. SONC certificate exists at order 6.

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem
   from Irene.sosonc import SOSONCRelaxations

   sg = CommutativeSemigroup(['x', 'y'])
   sga = SemigroupAlgebra(sg)
   x, y = sga['x'], sga['y']

   prog = OptimizationProblem(sga)
   motzkin = x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2
   prog.set_objective(motzkin)

   sosonc = SOSONCRelaxations(prog)
   result = sosonc.globalMinSOS(order=3)
   print(f"SOS lower bound (order 3): {result:.6f}")

**Choi-Lam Polynomial**: Nonnegative on :math:`\mathbb{R}^2`, not SOS. Zero at
:math:`(0,0), (\pm 1, 0), (0, \pm 1)`. Used in the Mean Polynomial paper as a
separating example.

.. code-block:: python

   choi_lam = x**4 * y**2 + x**2 * y**4 + x**2 * y**2 * (x**2 + y**2 - 1)
   prog.set_objective(choi_lam)
   result_sonc = sosonc.globalMinSONC(order=3)
   print(f"SONC lower bound (order 3): {result_sonc:.6f}")

**Robinson Polynomial**: A degree-6 bivariate with known minimum. Frequently used
as a stress test for hierarchy convergence.

.. code-block:: python

   robinson = x**4 * y**2 + x**2 * y**4 + x**4 + y**4 - x**2 - y**2
   prog.set_objective(robinson)
   result_combined = sosonc.globalMinSOSPSONC(order=3, first='sos')
   print(f"SOS+SONC lower bound: {result_combined:.6f}")

Phase 3 Optimization Benchmarks
===============================

The ``benchmarks/p3_vs_baseline.py`` script compares the baseline configuration
(no reduction pipeline) against the Phase 3 optimized configuration (Newton polytope
pruning + border basis + correlative sparsity detection):

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem
   from Irene.relaxation_api import RelaxationEngine
   from Irene.relaxations import RelaxationConfig

   # Build problem (e.g., Motzkin)
   sg = CommutativeSemigroup(["x", "y"])
   sa = SemigroupAlgebra(sg)
   x, y = sa["x"], sa["y"]
   prog = OptimizationProblem(sa)
   prog.set_objective(x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2)

   # Baseline: no reductions
   baseline_config = RelaxationConfig(
       reduction_method="none",
       monomial_pruning=False,
       sparsity_detection=False,
   )

   # Phase 3 optimized: full reduction pipeline
   p3_config = RelaxationConfig(
       reduction_method="newton_polytope",
       monomial_pruning=True,
       sparsity_detection=True,
       verbose_reduction=False,
   )

   engine = RelaxationEngine(prog, order=2, solver="clarabel", config=p3_config)
   result = engine.solve("sos")
   print(f"Value: {result.value:.8f}, Status: {result.status}")

The comparison measures matrix dimension, generation time, solve time, and final
bound for each configuration across orders 1&ndash;3.

Cross-Version Comparison
------------------------

The ``benchmarks/compare_irene_vs_rewrite.py`` script runs the same problems through
both the original Irene package and IreneRewrite to validate numerical consistency:

.. code-block:: bash

   python benchmarks/compare_irene_vs_rewrite.py

This verifies that Phase 3 optimizations (Newton pruning, border basis, sparsity)
produce bounds within tolerance of the baseline while reducing matrix dimensions.

Constrained Optimization Examples
=================================

The following examples demonstrate constrained polynomial optimization using the
current IreneRewrite API pattern with semigroup algebras.

Bounded Quartic Minimization
----------------------------

Minimize :math:`x^2 - 4x` subject to :math:`4 - x^2 \geq 0`:

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem
   from Irene.relaxations import SDPRelaxations

   sg = CommutativeSemigroup(['x'])
   sga = SemigroupAlgebra(sg)
   x = sga['x']

   prog = OptimizationProblem(sga)
   prog.set_objective(x**2 - 4 * x)
   prog.add_constraints([4 - x**2])

   sdp = SDPRelaxations(prog, verbosity=0)
   for t in range(1, 5):
       result = sdp.solve(order=t)
       print(f"Order {t}: bound = {result['value']:.6f}, "
             f"time = {result['time_solve']:.3f}s, "
             f"basis = {result['basis_size']}")

Trigonometric Polynomial via Algebraic Substitution
---------------------------------------------------

Minimize :math:`\cos^2(x) + \sin^2(y)` over :math:`-5 \leq x,y \leq 5` by
substituting :math:`s_1 = \sin(x), c_1 = \cos(x), s_2 = \sin(y), c_2 = \cos(y)`
and adding the algebraic relations :math:`s_i^2 + c_i^2 = 1`:

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem
   from Irene.relaxations import SDPRelaxations

   sg = CommutativeSemigroup(['s1', 'c1', 's2', 'c2'])
   sga = SemigroupAlgebra(sg)
   s1, c1, s2, c2 = sga['s1'], sga['c1'], sga['s2'], sga['c2']

   prog = OptimizationProblem(sga)
   prog.set_objective(c1**2 + s2**2)

   # Algebraic relations: sin^2 + cos^2 = 1 (encoded as equality via pair of inequalities)
   rel1 = 1 - s1**2 - c1**2
   rel2 = 1 - s2**2 - c2**2
   prog.add_constraints([rel1, -rel1, rel2, -rel2])

   # Box constraints: x^2 <= 25, y^2 <= 25 (approximated)
   prog.add_constraints([25 - s1**2, 25 - s2**2])

   sdp = SDPRelaxations(prog, verbosity=0)
   result = sdp.solve(order=2)
   print(f"Lower bound: {result['value']:.8f}")

Performance Profiling Tools
===========================

IreneRewrite includes several profiling scripts for diagnosing performance bottlenecks:

- **``instrument_relaxation_v2.py``**: Instruments the relaxation pipeline with per-stage timing (basis construction, matrix assembly, solver call)
- **``profile_symengine_overhead.py``**: Measures SymEngine vs SymPy overhead in polynomial arithmetic operations
- **``stage_bc_detailed.py``**: Fine-grained breakdown of border basis and sparsity detection stages
- **``p3_diagnose.py``**: Diagnostic script for Phase 3 reduction pipeline behavior

Solver Routing Behavior
=======================

IreneRewrite routes SDP solves through multiple backends:

1. **CVXPY + CLARABEL** (default): Robust handling of ill-conditioned moment matrices and reliable infeasibility detection
2. **Native CVXOPT**: Fallback path; note that infeasibility detection can differ from CLARABEL due to tolerance handling
3. **External solvers** (DSDP, SDPA, CSDP): For very large instances via CLI invocation

The solver is selected via the ``solver`` parameter:

.. code-block:: python

   result = sdp.solve(order=4, solver='clarabel')    # default
   result = sdp.solve(order=4, solver='cvxopt')      # native fallback
   result = sdp.solve(order=4, solver='dsdp')        # external CLI

Return Structure
----------------

The ``solve()`` method returns a dictionary with these keys:

- ``value`` (float): Primal objective value (lower bound on minimum)
- ``status`` (str): Solver status string ('optimal', 'infeasible', etc.)
- ``order`` (int): Relaxation order used
- ``basis_size`` (int): Number of moment variables
- ``time_init`` (float): SDP construction time in seconds
- ``time_solve`` (float): Solver runtime in seconds

References
==========

- Lasserre, J.-B. (2001). "Global optimization with polynomials and the problem of sums of squares." *SIAM Journal on Optimization*, 11(3), 793–812.
- Parrilo, P. A. (2000). "Structured semidefinite programs and semialgebraic geometry methods in robustness and optimization." *Caltech PhD Thesis*.
- De Klerk, E. & Pasechnik, D. V. (2002). "Computational antilipschitz optimization." *SIAM Journal on Optimization*, 12(4), 1072–1090.
