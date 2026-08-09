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

Infeasibility Detection Differences
-----------------------------------

An important distinction between backends:

- **CLARABEL**: Detects infeasibility reliably via dual unboundedness certificates. Returns ``'infeasible'`` status when the moment matrix PSD constraint cannot be satisfied.
- **Native CVXOPT**: May return ``'unknown'`` or timeout on near-infeasible problems due to different tolerance handling in the cone solver.

This means CLARABEL is preferred for Positivstellensatz applications where proving
infeasibility (i.e., nonnegativity certificates) is as important as finding bounds.

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
