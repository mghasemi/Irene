=================================
SOS+SONC Relaxations
=================================

The module ``sosonc.py`` implements the SOS+SONC two-step optimization
framework from Moritz Schick's PhD thesis (*Sums of squares plus sums of
nonnegative circuit polynomials*, Universität Konstanz). It provides combined
SOS+SONC lower bounds for unconstrained polynomial optimization, translating
Schick's MATLAB toolbox to Python and integrating it into the Irene framework.

.. contents::
   :local:
   :depth: 2


Theory
=================================

The SOS+SONC Cone :math:`\Sigma + C`
-------------------------------------

Let :math:`\Sigma_{n,2d}` denote the cone of sums of squares (SOS) of
polynomials in :math:`n` variables of degree at most :math:`2d`, and let
:math:`C_{n,2d}` denote the cone of sums of nonnegative circuit
polynomials (SONC) of the same degree bound. Both cones are proper subsets of
the PSD cone, and neither contains the other in general
(see Corollary 6.14 of the companion manuscript on mean polynomials).

The Minkowski sum

.. math::

   (\Sigma + C)_{n,2d} = \{s + c \mid s \in \Sigma_{n,2d},\; c \in C_{n,2d}\}

is a natural certificate family that strictly contains both SOS and SONC.
For an unconstrained polynomial optimization problem

.. math::

   f^* = \inf_{x \in \mathbb{R}^n} f(x),

the SOS+SONC relaxation computes

.. math::

   f_{\Sigma + C}^* = \sup\{\lambda \in \mathbb{R} \mid f - \lambda \in (\Sigma + C)_{n,2d}\}.

Because :math:`\Sigma \subseteq \Sigma + C` and :math:`C \subseteq \Sigma + C`,
we always have

.. math::

   \max\{f_\Sigma^*,\; f_C^*\} \;\leq\; f_{\Sigma + C}^* \;\leq\; f^*.

Two-Step Preprocessing (Algorithms 4 & 5)
-------------------------------------------

Schick's thesis introduces two complementary strategies that avoid solving
the full SDP-plus-exponential-cone feasibility problem.

**Algorithm 4 — SOS-first (SOS preprocessing → SONC relaxation)**

1. Find :math:`g^* \in \Sigma_{n,2d}` minimising a convex distance
   :math:`\varphi(f, g^*)` (e.g., the :math:`\ell_2`-norm of the
   coefficient vector of :math:`f - g^*`). The intuition is to
   "cover" as much of :math:`f` as possible with an SOS certificate,
   leaving a residual that is well-suited for SONC.

2. Solve the SONC relaxation for the residual :math:`h = f - g^*`,
   obtaining :math:`\mu^* = h_C^* = \sup\{\mu \mid h - \mu \in C\}`.

3. The combined lower bound is

   .. math::

      f_{\Sigma+C}^* \geq \mu^*,

   with the decomposition :math:`f - \mu^* = g^* + (h - \mu^*) \in \Sigma + C`.

**Algorithm 5 — SONC-first (SONC preprocessing → SOS relaxation)**

The roles of SOS and SONC are swapped: first find :math:`g^* \in C`
minimising :math:`\psi(f, g^*)`, then solve the SOS relaxation on
:math:`h = f - g^*`.

Computational Complexity
-------------------------

- **Pure SOS:** semidefinite programming — :math:`O(n^{6r})` in the Lasserre
  relaxation order :math:`r`.
- **Pure SONC:** signomial geometric programming — polynomial-time per
  sequential GP iteration. Very fast for ST-polynomials; insensitive to
  degree increases.
- **SOS+SONC (two-step):** the cost of one SDP plus one signomial program.
  The preprocessing overhead is minimal.

When SOS is infeasible (the polynomial is not SOS), the SOS-first two-step
falls back to pure SONC, guaranteeing at least the SONC bound.
Symmetrically for the SONC-first variant.

Implementation in Irene
=================================

The central class is :class:`~Irene.sosonc.SOSONCRelaxations`.

.. autoclass:: Irene.sosonc.SOSONCRelaxations
   :members: globalMinSOS, globalMinSONC, globalMinSOSPSONC
   :noindex:

Constructor
---------------------------------

.. code-block:: python

   from Irene.sosonc import SOSONCRelaxations
   from Irene.program import OptimizationProblem

   engine = SOSONCRelaxations(
       prog,                         # OptimizationProblem instance
       error_bound=1e-10,
       verbosity=1,
       solver='cvxopt',              # SDP solver
       use_local_solve=True,         # signomial GP local solve
       relaxation_order=1,           # Lasserre relaxation order
   )

Detailed Method Reference
---------------------------------------

:meth:`~Irene.sosonc.SOSONCRelaxations.globalMinSOS`
   Computes :math:`\lambda^* = \sup\{\lambda \mid f - \lambda \in \Sigma\}`
   by wrapping :class:`~Irene.relaxations.SDPRelaxations`. Uses a Gram-matrix
   SDP via the selected solver (CVXOPT, CSDP, SDPA, or DSDP).

   Returns :class:`~Irene.sosonc.SOSONCRelaxSol` with ``.val``, ``.status``,
   ``.error_code``, and ``.runtime``.

:meth:`~Irene.sosonc.SOSONCRelaxations.globalMinSONC`
   Computes :math:`\lambda^* = \sup\{\lambda \mid f - \lambda \in C\}`
   by wrapping :class:`~Irene.sonc.SONCRelaxations`. Uses signomial
   geometric programming via GPkit with the CVXOPT backend.

   Returns :class:`~Irene.sosonc.SOSONCRelaxSol`.

:meth:`~Irene.sosonc.SOSONCRelaxations.globalMinSOSPSONC`
   Implements the two-step SOS+SONC lower bound.

   - ``first='sos'`` → Algorithm 4 (SOS preprocessing → SONC residual)
   - ``first='sonc'`` → Algorithm 5 (SONC preprocessing → SOS residual)

   The residual :math:`h = f - \lambda^*` is constructed by shifting the
   constant term of the :class:`~Irene.grouprings.SemigroupAlgebraElement`.
   If residual construction fails, the method falls back to
   :math:`\max\{\lambda_{\text{SOS}}, \lambda_{\text{SONC}}\}`.

Result Container
---------------------------------------

.. autoclass:: Irene.sosonc.SOSONCRelaxSol
   :members: val, method, f_sos, f_sonc, status, error_code, runtime, message
   :noindex:

Convenience Function
---------------------------------------

.. autofunction:: Irene.sosonc.sosonc_bounds
   :noindex:

Returns a dictionary with four keys: ``'sos'``, ``'sonc'``,
``'sos_first'``, ``'sonc_first'``, each mapping to the corresponding
lower bound.

Implementation Pipeline
=================================

The internal pipeline of ``globalMinSOSPSONC`` when called with
``first='sos'`` is:

1. **Solve** SOS relaxation on the original problem → :math:`\lambda_{\text{SOS}}`.
2. **Build residual** :math:`h = f - \lambda_{\text{SOS}}` by cloning the
   objective's coefficient-content list and subtracting
   :math:`\lambda_{\text{SOS}}` from the identity monomial's coefficient.
3. **Construct** a new :class:`~Irene.program.OptimizationProblem` with
   :math:`h` as the objective and the same constraints (if any).
4. **Solve** SONC on the residual → :math:`\mu^*`.
5. **Combine** :math:`\lambda_{\text{SOS}} + \mu^*` as the final lower bound.
6. **Fallback:** if any step raises an exception, return
   :math:`\max\{\lambda_{\text{SOS}}, \lambda_{\text{SONC}}\}`.

The SONC-first variant is symmetric.

Relation to the Mean Polynomial Hierarchy
=========================================

The SOS+SONC cone :math:`\Sigma + C` is a proper subset of the mean
polynomial preprime :math:`T_{\text{mean}}` introduced in Sections 6--7
of the companion manuscript (Ghasemi & Kuhlmann, 2026). Consequently,
the SOS+SONC lower bounds from this module are dominated by the
mean-polynomial hierarchy bounds described in
``algorithm_mean_polynomial_hierarchy.md``.

The natural extension of this module to the full mean-polynomial
hierarchy replaces :math:`C_{n,2d}` (SONC) with
:math:`T_{\text{mean},r}^{(1)}` (single mean forms) in the signomial
program, and uses :math:`M_{2,1}` forms to capture the SOS component
directly without requiring an SDP. This extension is planned for a
future ``mean_polynomial.py`` module.

Example
=================================

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem
   from Irene.sosonc import SOSONCRelaxations, sosonc_bounds

   # x^4 - x^2 on ℝ (global minimum = -0.25)
   sg = CommutativeSemigroup(['x'])
   sga = SemigroupAlgebra(sg)
   x = sga['x']
   prog = OptimizationProblem(sga)
   prog.set_objective(x ** 4 - x ** 2)

   engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=2)
   result = engine.globalMinSOSPSONC(first='sos')
   print(result)

   # Compare all four bounds
   bounds = sosonc_bounds(prog, verbosity=0)
   for method, val in bounds.items():
       print(f"{method}: {val:.6f}")

Test Suite
=================================

The test file ``tests/test_sosonc.py`` exercises the module with:

- **Container validation:** default values and string representation of
  :class:`~Irene.sosonc.SOSONCRelaxSol`.
- **SOS integration:** :math:`x^2 + y^2` (minimum 0), :math:`x^4 - x^2`
  (minimum -0.25) with relaxation order 2.
- **SONC integration:** Motzkin polynomial (a known SONC form).
- **Two-step pipeline:** SOS-first and SONC-first on quadratic problems.
- **Convenience wrapper:** ``sosonc_bounds`` returning all four keys.
- **Error handling:** invalid ``first`` argument rejection.

Run with::

   cd Irene && python -m pytest tests/test_sosonc.py -v


Further Reading
=================================

1. M. Schick, *Sums of squares plus sums of nonnegative circuit
   polynomials*, PhD dissertation, Universität Konstanz, 2025.
   `GitHub repository <https://github.com/schick-moritz/SOS_plus_SONC_toolbox>`__.

2. M. Dressler, S. Iliman, and T. de Wolff, *An Approach to Constrained
   Polynomial Optimization via Nonnegative Circuit Polynomials and
   Geometric Programming*, Journal of Symbolic Computation 91 (2019),
   149--172.

3. M. Ghasemi and S. Kuhlmann, *The Cone Generated by Positive
   Semidefinite Mean Polynomials*, 2026 (companion manuscript,
   Section 7).
