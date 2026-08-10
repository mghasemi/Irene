========================================
Semidefinite Programming Relaxations
========================================

The SDP module implements Lasserre's hierarchy of semidefinite programming
relaxations for polynomial optimization problems. Given a problem

.. math::

   \min \{f(x) : g_1(x) \geq 0, \dots, g_m(x) \geq 0, x \in K\},

the hierarchy constructs a sequence of SDPs whose optimal values converge
monotonically to the true optimum under mild topological conditions.

Moment and Localizing Matrices
==============================

At relaxation order :math:`t`, the method introduces moment variables
:math:`y_\alpha` for each exponent :math:`\alpha \in \Lambda_t = \{\alpha : |\alpha| \leq t\}`
and requires that the **moment matrix** :math:`M_t(y)` and all **localizing matrices**
:math:`M_t(g_i y)` be positive semidefinite.

The moment matrix has entries indexed by monomials in the basis :math:`B_t`:

.. math::

   M_t(y)_{\alpha, \beta} = y_{\alpha + \beta}, \quad \alpha, \beta \in B_t.

For each constraint :math:`g_i(x) = \sum_\gamma h_{i,\gamma} x^\gamma`, the localizing
matrix is defined by:

.. math::

   M_t(g_i y)_{\alpha, \beta} = \sum_\gamma h_{i,\gamma} y_{\alpha + \beta + \gamma}.

The SDP at order :math:`t` reads:

.. math::

   \min y_f = \sum_\alpha f_\alpha y_\alpha \quad \text{s.t.} \quad M_t(y) \succeq 0, \;\; M_t(g_i y) \succeq 0.

API Overview
============

The ``SDPRelaxations`` class provides the primary interface:

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebraElement
   from Irene.program import OptimizationProblem
   from Irene.relaxations import SDPRelaxations

   # Define semigroup and variables
   sg = CommutativeSemigroup(['x', 'y'])
   x, y = sg.generators[0], sg.generators[1]

   # Build problem with semigroup algebra elements
   objective = SemigroupAlgebraElement(sg, {sg.one: 1, sg.monomial({0: 2}): -1})  # x^2
   constraint = SemigroupAlgebraElement(sg, {sg.one: 1, sg.monomial({0: 2, 1: 2}): 1})  # 1 + x^2*y^2

   prog = OptimizationProblem(sg, objective)
   prog.add_constraint(constraint >= 0)

   # Solve with SDP hierarchy
   sdp = SDPRelaxations(prog)
   result = sdp.solve(order=4)
   print(f"Lower bound: {result['value']:.6f}")
   print(f"Solver status: {result['status']}")

Solver Routing
==============

IreneRewrite routes SDP solves through multiple backends automatically:

**Primary path (CVXPY + CLARABEL)**: The default solver uses CVXPY's DCP-compliant
formulation with the CLARABEL conic interior-point method. This provides robust
handling of ill-conditioned moment matrices and reliable infeasibility detection.

**Fallback path (native CVXOPT)**: If CVXPY or CLARABEL are unavailable, the solver
falls back to the native CVXOPT implementation. Note that CVXOPT's infeasibility
detection can differ from CLARABEL — problems declared infeasible by CLARABEL may
return unbounded solutions in CVXOPT due to different tolerance handling.

**External solvers (DSDP, SDPA, CSDP)**: For very large instances, external CLI-based
solvers can be invoked. These require separate installation and are configured via
the solver parameter.

.. code-block:: python

   # Explicit solver selection
   result = sdp.solve(order=4, solver='clarabel')    # CLARABEL via CVXPY (default)
   result = sdp.solve(order=4, solver='cvxopt')      # Native CVXOPT path
   result = sdp.solve(order=4, solver='dsdp')        # External DSDP CLI

Return Structure
----------------

The ``solve()`` method returns a dictionary with the following keys:

- ``value`` (float): Primal objective value (lower bound on minimum)
- ``status`` (str): Solver status string ('optimal', 'infeasible', etc.)
- ``order`` (int): Relaxation order used
- ``basis_size`` (int): Number of moment variables
- ``time_init`` (float): SDP construction time in seconds
- ``time_solve`` (float): Solver runtime in seconds

Practical Example: Bounded Polynomial
=====================================

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebraElement
   from Irene.program import OptimizationProblem
   from Irene.relaxations import SDPRelaxations

   sg = CommutativeSemigroup(['x'])
   x = sg.generators[0]

   # Minimize (x - 2)^2 subject to x^2 <= 4
   objective = SemigroupAlgebraElement(sg, {sg.monomial({0: 1}): -4, sg.monomial({0: 2}): 1})  # x^2 - 4x
   # Add constant term separately if needed
   constraint = SemigroupAlgebraElement(sg, {sg.one: 4, sg.monomial({0: 2}): -1})  # 4 - x^2

   prog = OptimizationProblem(sg, objective)
   prog.add_constraint(constraint >= 0)

   sdp = SDPRelaxations(prog)
   for t in range(1, 5):
       result = sdp.solve(order=t)
       print(f"Order {t}: bound = {result['value']:.6f}, "
             f"time = {result['time_solve']:.3f}s, "
             f"basis = {result['basis_size']}")

Hierarchy Convergence
=====================

Under the Archimedean condition (the set :math:`K` is contained in a compact
spectrahedron), Putinar's Positivstellensatz guarantees that the hierarchy
terminates: for some finite order :math:`t^*`, the SDP at order :math:`t^*`
returns the exact global minimum. In practice, convergence is often achieved
at much lower orders than the theoretical bound suggests.

For non-Archimedean sets, the hierarchy still provides valid lower bounds that
converge asymptotically, but termination is not guaranteed at any finite order.

References
==========

- Lasserre, J.-B. (2001). "Global optimization with polynomials and the problem of sums of squares." *SIAM Journal on Optimization*, 11(3), 793–812.
- Parrilo, P. A. (2000). "Structured semidefinite programs and semialgebraic geometry methods in robustness and optimization." *Caltech PhD Thesis*.
- Laurent, M. (2009). "Sums of squares, moment matrices and optimization over polynomials." *Developments in Mathematics*, 14, 157–270.
