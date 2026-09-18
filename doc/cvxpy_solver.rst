========================================
CVXPY Solver Layer
========================================

The ``cvxpy_solver.py`` module provides a DCP-compliant solver layer that bridges
Irene's SDP relaxation constructs to modern convex optimization backends via CVXPY.
It supports CLARABEL (default), SCS, and native CVXOPT as solver backends.

.. contents::
   :local:
   :depth: 2

Architecture
============

The CVXPY layer sits between Irene's moment matrix construction and the actual
numerical solver. It handles:

1. **DCP formulation**: Translates moment matrix PSD constraints into CVXPY's disciplined convex programming framework
2. **Solver routing**: Selects and configures the backend solver based on problem size and user preference
3. **Result extraction**: Parses solver output back into Irene's result structure with timing metadata

Solver Routing Behavior
-----------------------

The default solver is **CLARABEL** (an interior-point method with robust handling of
ill-conditioned moment matrices). The routing logic:

.. code-block:: python

   from Irene.cvxpy_solver import CvxpySDP

   sdp = CvxpySDP(moment_matrix, localizing_matrices)
   result = sdp.solve(solver='CLARABEL')  # default
   result = sdp.solve(solver='SCS')       # ADMM-based, faster for large problems
   result = sdp.solve(solver='CVXOPT')    # native fallback

Infeasibility Detection and Positivstellensatz Duality
------------------------------------------------------

An important distinction between backends affects correctness of nonnegativity
certificates:

- **Native CVXOPT (C interface)**: Correctly reports ``'infeasible'`` for SDPs
  whose primal is infeasible.  This is the reliable path for SOS
  certification: when CVXOPT declares infeasibility, it means the moment
  matrix cannot be made PSD while satisfying constraints, which is equivalent
  to a **dual SOS proof of nonnegativity** via Putinar's
  Positivstellensatz.

- **CLARABEL (via CVXPY)**: May return finite weak bounds with status
  ``'optimal'`` for infeasible SDPs, because its interior-point method
  interprets primal infeasibility differently.  CLARABEL's dual
  unboundedness certificate is mathematically equivalent to a
  Putinar-type representation, but the solver may terminate with a weak
  bound rather than a clean ``'infeasible'`` status.

**Conic Duality Guide.** In the Moment-SOS hierarchy, the primal SDP minimizes
:math:`L(f)` subject to :math:`M_t(y) \succeq 0` and :math:`M_t(g_i y) \succeq 0`.
Its dual maximizes :math:`\gamma` such that :math:`f - \gamma` admits a
representation

.. math::

   f - \gamma = \sigma_0 + \sum_i \sigma_i g_i, \qquad
   \sigma_0, \dots, \sigma_m \in \sum \mathbb{R}[x]_{\le 2t}^2.

A **dual unboundedness certificate** from CLARABEL (or an **infeasible** status
from native CVXOPT) corresponds exactly to a certified SOS decomposition proving
:math:`f \ge \gamma` on the semialgebraic set :math:`K =
\{x : g_i(x) \ge 0\}`.  Both solvers therefore produce valid certificates; the
difference is only in how they report the status.

**Solver Selection Rule.**

- Use native CVXOPT when **correct infeasibility detection is critical**
  (e.g., proving a polynomial is NOT SOS, as in the Motzkin and Choi–Lam
  gallery problems).  The IreneRewrite engine routes ``solver='cvxopt'``
  through CVXOPT's native C interface for this reason.
- Prefer CLARABEL for **large-scale feasible problems** (up to ~500 moment
  variables) where its robust interior-point convergence is valuable and
  infeasibility is not expected.
- Use SCS (``solver='scs'``) when moment matrix dimension exceeds
  :math:`500 \times 500`; its first-order ADMM method scales better but
  requires tighter tolerances for certificate-quality bounds.

**Numerical Parameter Recommendations.**  For high-order relaxations
(:math:`t \ge 3`) where moment matrices become ill-conditioned:

.. list-table:: Solver tolerance settings for high-order SDP
   :header-rows: 1

   * - Solver
     - Tolerance parameter(s)
     - Typical value
   * - CLARABEL
     - ``tol_gap_abs``, ``tol_feas``
     - ``1e-8``
   * - SCS
     - ``eps_abs``
     - ``1e-6`` (tighten to ``1e-7`` when :math:`M_t(y) > 500 \\times 500`)
   * - CVXOPT (native)
     - ``abstol``, ``reltol``, ``feastol``
     - defaults adequate up to :math:`t=3`; raise ``feastol`` to ``1e-7`` for :math:`t \\ge 4`

API Reference
=============

CvxpySDP Class
--------------

.. code-block:: python

   from Irene.cvxpy_solver import CvxpySDP

   # Construct from moment matrix and constraint blocks
   sdp = CvxpySDP(M, A_blocks, c_vector)

   # Solve with default solver (CLARABEL)
   result = sdp.solve()

   # Solve with specific backend
   result = sdp.solve(solver='SCS', verbose=True)

The constructor accepts:

- **M** (matrix): The moment matrix template for PSD constraints
- **A_blocks** (list): Linear constraint blocks :math:`\sum_i y_i A_i`
- **c_vector** (array): Objective coefficients

Returns a result dictionary with keys: ``value``, ``status``, ``time_solve``, ``solver_used``.

Performance Notes
=================

1. **CLARABEL** is recommended for problems up to ~500 moment variables (orders 2–3 in bivariate settings)
2. **SCS** becomes competitive for larger instances due to its first-order method scaling, but may require tighter tolerances for certificate-quality bounds
3. The CVXPY layer adds ~10–20% overhead vs direct solver calls due to DCP graph construction, but provides uniform API and robust error handling
