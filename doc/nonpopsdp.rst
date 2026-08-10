========================================
Non-Polynomial Optimization (NonPOPSDP)
========================================

The ``nonpopsdp.py`` module (ported from original Irene in 2026-08-09) applies
Lasserre's moment-SOS hierarchy to optimization problems whose objective or
constraints involve transcendental functions (``exp``, ``sin``, ``cos``, ...).

Pipeline
========

1. **Approximate** each transcendental function by a polynomial surrogate
   (Taylor or Chebyshev).
2. **Substitute** the surrogates into the objective/constraints, producing a
   polynomial optimization problem (POP).
3. **Relax** the POP via ``SDPRelaxations`` (Lasserre hierarchy).
4. **Solve** the SDP (CVXOPT by default; other backends via
   ``SDPRelaxations``).

Per Josz--Henrion (2014), a redundant ball constraint is ALWAYS added to the
relaxation to guarantee strong duality (no primal--dual gap).

Quick Example
=============

.. code-block:: python

   from math import exp
   import sympy as sp
   from Irene.nonpopsdp import NonPOPSDP

   x = sp.symbols("x")
   exp_sym = sp.symbols("exp")            # bare symbol named after the function

   pop = NonPOPSDP(
       x,
       {"exp": {"func": exp, "method": "chebyshev",
                "domain": (-1.0, 1.0), "degree": 6}},
       relax_order=2, ball_radius=1.0, verbosity=0,
   )
   pop.set_objective(exp_sym)             # min exp(x) on [-1, 1]
   lb = pop.solve()                       # ~ 0.3679 (true min exp(-1))

Function surrogates are referenced by **bare symbols named after the function**
(``symbols('sin')``), not by ``sp.sin(x)`` — ``TranscendentalApproximator.substitute``
replaces the named symbol with the polynomial surrogate.

API
===

- ``taylor_approx(func, var, center, degree)`` — Taylor surrogate with Lagrange
  remainder bound.
- ``chebyshev_approx(func, var, domain, degree)`` — Chebyshev surrogate on a
  domain with empirical max error.
- ``TranscendentalApproximator(var, approx_map)`` — builds and substitutes
  several surrogates at once.
- ``NonPOPSDP(var, approx_map, relax_order, ball_radius, ...)`` — single-variable
  pipeline (``set_objective``, ``add_constraint``, ``solve``).
- ``NonPOPSDP_Multi(vars, approx_map, ...)`` — multi-variable pipeline with
  per-function ``var_idx``.

Numerical Fixes in the Port (vs original Irene)
===============================================

The original implementation had two latent numerical bugs, both fixed in this
port (verified against original Irene):

1. **Chebyshev coefficients** were extracted with an incorrectly scaled raw
   FFT, producing catastrophically wrong surrogates (max error ~61.5 for
   ``exp`` of degree 6 on ``[-2, 2]``; the true error is ~5e-4). The port uses
   ``numpy.polynomial.chebyshev.chebfit`` and also fixed an off-by-one in the
   error-evaluation grid (``fine_t = 2(x-mid)/(b-a) - 1`` mapped ``[a,b]``
   onto ``[-2, 0]``).
2. **Taylor coefficients** were computed with naive central finite differences
   (error ~1e36 for ``exp`` at degree 6). The port uses a high-order
   central-difference stencil with Richardson extrapolation at 60-digit
   precision (``_mp_derivative``), accurate to ~1e-6 at degree 7.

The pipeline API and SDP construction are otherwise faithful to the original.

Integration Notes
=================

- ``NonPOPSDP.solve`` accepts an optional ``config`` (``RelaxationConfig``),
  so the surrogate POP inherits the quotient-basis and reduction-pipeline
  options (see :doc:`relaxation_api`).
- The module is backend-agnostic: surrogates are SymPy expressions, and the
  SDP construction routes through the user-selectable symbolic engine.
