========================================
Legacy API Migration Guide
========================================

This chapter maps every construct from the original Irene API (``SDPRelaxations``,
``Mom()``, ``Probability=False``, etc.) to the modern IreneRewrite pipeline
(``OptimizationProblem`` + ``RelaxationEngine``).

The legacy API remains available for backward compatibility — existing code
will continue to run.  However, new development should use the modern pipeline
for its unified configuration, reduction pipeline integration, and consistent
result types.

Problem Construction
====================

.. list-table:: Legacy → Modern Mapping: Problem Setup
   :header-rows: 1

   * - Legacy API (original Irene)
     - Modern API (IreneRewrite)
   * - ``Rlx = SDPRelaxations([x, y, z])``
     - Use ``CommutativeSemigroup`` → ``SemigroupAlgebra`` → ``OptimizationProblem``
   * - ``Rlx = SDPRelaxations([x, y, f], relations=[...])``
     - ``sg = CommutativeSemigroup(['x','y','f']); sga = SemigroupAlgebra(sg);``
       ``sga.add_relations([...])``
   * - ``Rlx.SetObjective(f)``
     - ``prog.set_objective(f)``
   * - ``Rlx.AddConstraint(g >= 0)``
     - ``prog.add_constraint(g >= 0)``
   * - ``Rlx.SetMonoOrd('lex')``
     - Configured via ``RelaxationConfig(quotient_basis=...)``
   * - ``Rlx.MomentsOrd(3)``
     - ``engine = RelaxationEngine(prog, order=3)``

Objective and Constraints
==========================

.. code-block:: python
   :caption: Legacy

   from sympy import symbols
   x, y, z = symbols('x y z')
   Rlx = SDPRelaxations([x, y, z])
   Rlx.SetObjective(-2*x + y - z)
   Rlx.AddConstraint(x + y + z <= 4)
   Rlx.AddConstraint(x >= 0)

.. code-block:: python
   :caption: Modern

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem

   sg = CommutativeSemigroup(['x', 'y', 'z'])
   sga = SemigroupAlgebra(sg)
   x, y, z = sga['x'], sga['y'], sga['z']

   prog = OptimizationProblem(sga)
   prog.set_objective(-2*x + y - z)
   prog.add_constraint(x + y + z <= 4)
   prog.add_constraint(x >= 0)

Moment Constraints
==================

.. list-table:: Legacy → Modern Mapping: Moment Constraints
   :header-rows: 1

   * - Legacy API
     - Modern API
   * - ``Rlx.MomentConstraint(Mom(x*y) >= 0.5)``
     - ``prog.add_moment_constraint(...)``
   * - ``Rlx.MomentConstraint(Mom(x**2) == 1/3)``
     - ``prog.add_moment_constraint(..., equality=True)``

Solver Selection and Solving
=============================

.. list-table:: Legacy → Modern Mapping: Solving
   :header-rows: 1

   * - Legacy API
     - Modern API
   * - ``Rlx.SetSDPSolver('dsdp')``
     - ``RelaxationEngine(prog, solver='dsdp')``
   * - ``Rlx.InitSDP()``
     - Automatic on ``engine.solve()``
   * - ``Rlx.Minimize()``
     - ``engine.solve('sos')``
   * - ``print(Rlx.Solution)``
     - ``print(result)`` / ``result.value`` / ``result.status``
   * - ``Rlx.Solution[x*y]``
     - Result object attributes — see ``RelaxResult``

Probability and Moment Settings
================================

.. list-table:: Legacy → Modern Mapping: Settings
   :header-rows: 1

   * - Legacy API
     - Modern API
   * - ``Rlx.Probability = False``
     - Configured through ``OptimizationProblem`` problem-level settings
   * - ``Rlx.PSDMoment = True``
     - Always True in modern API (PSD constraint is always enforced)
   * - ``Rlx.ErrorTolerance``
     - Configured through solver-specific tolerance parameters
       (see :doc:`cvxpy_solver`)

SOS Decomposition
==================

.. code-block:: python
   :caption: Legacy

   Rlx.Minimize()
   V = Rlx.Decompose()
   # V = {0: [a01, a02, ...], 1: [a11, ...], ...}
   sos = expand(Rlx.ReduceExp(sum([p**2 for p in V[0]])))

.. code-block:: python
   :caption: Modern

   result = engine.solve('sos')
   # result.certificate contains the SOS decomposition
   # result.certificate = {'f_sos': ..., 'f_sonc': ...}

Solution Extraction
====================

.. list-table:: Legacy → Modern Mapping: Solution Extraction
   :header-rows: 1

   * - Legacy API
     - Modern API
   * - ``Rlx.Solution.ExtractSolution('LH', card)``
     - ``result`` attributes; see ``RelaxResult.solver_info``
   * - ``Rlx.Solution.ExtractSolution('scipy', card)``
     - Configured through solver backend options
   * - ``Rlx.Solution.Support``
     - ``result.solver_info`` dictionary

Complete Example: Legacy vs Modern
===================================

.. code-block:: python
   :caption: Legacy (original Irene)

   from sympy import symbols
   from Irene import SDPRelaxations, Mom

   x, y, z = symbols('x y z')
   Rlx = SDPRelaxations([x, y, z])
   Rlx.SetObjective(-2*x + y - z)
   Rlx.AddConstraint(24 - 20*x + 9*y - 13*z + 4*x**2
                     - 4*x*y + 4*x*z + 2*y**2 - 2*y*z + 2*z**2 >= 0)
   Rlx.AddConstraint(x + y + z <= 4)
   Rlx.AddConstraint(3*y + z <= 6)
   Rlx.MomentsOrd(3)
   Rlx.SetSDPSolver('dsdp')
   Rlx.InitSDP()
   Rlx.Minimize()
   print(Rlx.Solution)

.. code-block:: python
   :caption: Modern (IreneRewrite)

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem
   from Irene.relaxation_api import RelaxationEngine
   from Irene.relaxations import RelaxationConfig

   sg = CommutativeSemigroup(['x', 'y', 'z'])
   sga = SemigroupAlgebra(sg)
   x, y, z = sga['x'], sga['y'], sga['z']

   prog = OptimizationProblem(sga)
   prog.set_objective(-2*x + y - z)
   prog.add_constraint(24 - 20*x + 9*y - 13*z + 4*x**2
                       - 4*x*y + 4*x*z + 2*y**2 - 2*y*z + 2*z**2 >= 0)
   prog.add_constraint(x + y + z <= 4)
   prog.add_constraint(3*y + z <= 6)
   prog.add_constraint(x >= 0)
   prog.add_constraint(x <= 2)
   prog.add_constraint(y >= 0)
   prog.add_constraint(z >= 0)
   prog.add_constraint(z <= 3)

   config = RelaxationConfig(
       reduction_method="newton_polytope",
       monomial_pruning=True,
   )
   engine = RelaxationEngine(prog, order=3, solver='dsdp', config=config)
   result = engine.solve('sos')
   print(f"Lower bound: {result.value:.8f}")
   print(f"Status: {result.status}")
