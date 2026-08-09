=================================
SONC Relaxations
=================================

The module ``sonc.py`` implements constrained SONC relaxations as geometric programs.

Formulation Focus
=================================

The implementation follows the Section 3 constrained SONC structure used in this
project and supports the Example 3.3 workflow in the repository examples.

The central transformed object is:

.. math::

   G(\mu) = f - \sum_i \mu_i g_i.

For each relevant exponent term, the model introduces auxiliary variables and
constraints that bound positive and negative parts of coefficients and connect
them through barycentric weights.

For each :math:`\beta` in the active delta set and each support point
:math:`\alpha(j)`, the implementation uses the standard constrained SONC
pattern with variables :math:`a_{\beta,j}`, :math:`b_\beta`, and multipliers
:math:`\mu`. To align with the Section 3 constrained formulation
(equation (3.3)), the key families implemented in code are:

Variable Roles
=================================

1. :math:`\mu`: Lagrange-style multipliers combining objective and constraints
   through :math:`G(\mu)`.
2. :math:`a_{\beta,j}`: circuit weights attached to support points for each
   active :math:`\beta`.
3. :math:`b_\beta`: auxiliary upper bounds controlling positive and negative
   parts of :math:`G(\mu)_\beta`.

**(F1) support-point inequalities**

.. math::

   \sum_j a_{\beta,j} \le G(\mu)_{\alpha(j)},

**(F2) circuit-product inequality**

.. math::

   \prod_j \left(\frac{a_{\beta,j}}{\lambda_j^{(\beta)}}\right)^{\lambda_j^{(\beta)}} \ge b_\beta,

**(F3) positive-part bound**

.. math::

   G(\mu)_\beta^+ \le b_\beta,

**(F4) negative-part bound**

.. math::

   G(\mu)_\beta^- \le b_\beta.

The GP objective minimized in this chapter is denoted by
:math:`p(\mu, a, b)`, consistent with the Section 3 notation used by the
constrained SONC formulation.

In implementation-oriented form, the objective can be read as:

.. math::

    p(\mu,a,b)
    = \sum_{i=1}^{m} \mu_i g_i^+(\alpha_0)
    + \sum_{\beta: \lambda_0^{(\beta)} > 0}
       \lambda_0^{(\beta)}
       b_\beta^{1/\lambda_0^{(\beta)}}
       \prod_{j\neq 0}
       \left(\frac{\lambda_j^{(\beta)}}{a_{\beta,j}}\right)^{\lambda_j^{(\beta)}/\lambda_0^{(\beta)}}.

This is the objective assembled in ``_build_objective`` after barycentric data
is available.

The barycentric weights :math:`\lambda^{(\beta)}` are computed from support points via
convex-combination routines before model assembly.

Branching on :math:`\lambda_0^{(\beta)}`
=========================================

The implementation distinguishes two geometric cases:

1. :math:`\lambda_0^{(\beta)} > 0`: objective includes a weighted
   exponential/product term for :math:`\beta`.
2. :math:`\lambda_0^{(\beta)} = 0`: circuit-product inequality (F2) controls
   :math:`b_\beta` directly without the :math:`\lambda_0` objective branch.

This split matches the constrained SONC structure implemented in the solver
assembly pipeline.

Implementation Stages in ``SONCRelaxations``
============================================

1. ``_build_delta_sets``: collects candidate terms.
2. ``_build_support_points``: computes support points/Newton vertices.
3. ``_build_beta_info``: computes barycentric coordinates.
4. ``_initialize_variables``: allocates variables for multipliers and circuit terms.
5. ``_build_constraints`` and ``_build_objective``: constructs the constrained GP.
6. ``solve``: solves and returns a lower bound.

Theory-to-Code Anchors
=================================

1. ``_build_delta_sets`` corresponds to active :math:`\beta` extraction.
2. ``_build_support_points`` and ``_build_beta_info`` provide support vertices and
   barycentric weights :math:`\lambda^{(\beta)}`.
3. ``_g_split`` computes positive and negative parts of coefficients in :math:`G(\mu)`.
4. ``_build_constraints`` encodes the constrained SONC families.
5. ``_build_objective`` builds the GP objective over :math:`\mu`, :math:`a`, and :math:`b` terms.

Geometric Interpretation
=================================

At a high level, SONC uses support geometry to construct nonnegativity
certificates from circuit-like structures rather than semidefinite moment
matrices. Irene's pipeline computes that geometry first, then injects it into
the constrained GP model.

Repository Anchors
=================================

1. ``tests/test_sonc_section3.py``: unit tests for barycentric weights, support
   points, delta sets, and end-to-end solve behavior.
2. The benchmark suite in ``benchmarks/`` includes SONC traces via the gallery system.

Runnable Example
=================================

The following example reproduces Example 3.3 from the constrained SONC paper,
minimizing :math:`1 + 2x^2y^4 + \tfrac{1}{2}x^3y^2` subject to
:math:`\tfrac{1}{3} - x^6y^2 \geqslant 0`::

    from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
    from Irene.program import OptimizationProblem
    from Irene.sonc import SONCRelaxations

    sg = CommutativeSemigroup(['x', 'y'])
    sga = SemigroupAlgebra(sg)
    x, y = sga['x'], sga['y']

    prog = OptimizationProblem(sga)
    prog.set_objective(1 + 2 * x**2 * y**4 + 0.5 * x**3 * y**2)
    prog.add_constraints([(1.0 / 3.0) - x**6 * y**2])

    sonc = SONCRelaxations(prog, verbosity=0)
    lower_bound = sonc.solve(verbosity=0)
    print(f"SONC lower bound: {lower_bound:.6f}")

The ``verbosity`` keyword controls GP solver output. The returned value is a
certified lower bound on the global minimum over the feasible set. To verify
barycentric weight correctness, inspect ``sonc._build_beta_info(...)`` which
returns convex-combination weights :math:`\sum_j \lambda_j^{(\beta)} = 1` for
each active term :math:`\beta`.
