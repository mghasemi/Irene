=================================
Geometric Programming Relaxations
=================================

The module ``geometric.py`` implements a geometric-programming lower-bound strategy
for constrained polynomial optimization over basic semialgebraic sets.

Theory to Implementation
=================================

The implementation follows the Section 4 construction used in the repository
notes, including a transformed program defined by a matrix :math:`H` and a
GP objective/constraint system over auxiliary variables.

Let :math:`g_0=-f` and let :math:`g_1,\dots,g_m` be the constraint polynomials.
The geometric pipeline builds transformed expressions :math:`h_j` using
:math:`H` and then solves for a lower bound through GP variables
:math:`\mu`, :math:`w_\alpha`, and :math:`z_\alpha`.

At a high level, the transformed family is:

.. math::

	h_j = \sum_k H_{k,j} g_k.

The relaxation objective then combines transformed positive parts and residual
terms indexed by active support elements. In implementation-oriented notation,
the objective has the form

.. math::

	 p(\mu, w, z)
	 = \sum_{j=1}^{m} h_+(h_j)\mu_j
	 + \sum_{\alpha \in \Delta^{<d}} (d-\deg\alpha)
		 \left(\frac{w_\alpha}{d}\right)^{\frac{d}{d-\deg\alpha}}
		 \prod_i \left(\frac{\alpha_i}{z_{\alpha,i}}\right)^{\frac{\alpha_i}{d-\deg\alpha}}.

Here :math:`d` denotes the effective even relaxation order used by the problem.

The implementation follows the lower-bound construction associated with Section 4,
including equation (3) as implemented in ``geometric.py`` for transformed objective
and constraint families.

The main implementation stages are:

1. ``auto_transform_matrix``: builds a transformation matrix satisfying the required sign structure.
2. ``transform_program``: constructs transformed expressions from the original objective/constraints.
3. ``_build_delta_sets`` and ``_initialize_variables``: prepare monomial sets and GP variables.
4. ``_build_objective`` and constraint assembly in ``solve``: formulate and solve the GP model.

Interpretation of the Matrix Transform
======================================

The matrix :math:`H` is not only a numerical preconditioner. It defines which
linear combinations of objective/constraint polynomials are exposed to the GP
model and therefore determines the sign and support structure of the final
inequalities. The ``auto_transform_matrix`` routine computes a feasible
transformation using diagonal/sign conditions encoded in the implementation.

Constraint Families in ``solve``
=================================

The method assembles several GP/signomial constraint blocks:

1. Normalization and variable bounds (including :math:`\mu_0=1`).
2. Generator-wise inequalities over transformed coefficients.
3. Degree-matching constraints for terms in :math:`\Delta^{=d}`.
4. Positive/negative coefficient balancing for all active terms.

These blocks correspond to the computational implementation of the Section 4
equation family and are solved through ``gpkit.Model``.

Lower-Bound Meaning
=================================

The solved value is used as a lower bound certificate for the original POP.
As with any hierarchy-style relaxation strategy, bound quality depends on model
order, transform quality, and problem structure.

Practical Notes
=================================

1. The method returns a lower bound as a floating-point value.
2. Solver availability and numerical conditioning can affect runtime behavior.
3. ``examples/GPExample.py`` provides a complete end-to-end usage pattern.
4. If automatic transformation is unstable for a given instance, a custom
   matrix can be supplied by setting ``gp.H`` before calling ``solve``.
