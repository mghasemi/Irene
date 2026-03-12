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

The barycentric weights :math:`\lambda^{(\beta)}` are computed from support points via
convex-combination routines before model assembly.

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

Repository Anchors
=================================

1. ``examples/SONCExample.py``: minimal SONC run path.
2. ``examples/SONCExample33.py``: Section 3.3-style benchmark trace.
3. ``tests/test_sonc_section3.py``: checks for barycentric weights, setup, and solve behavior.
