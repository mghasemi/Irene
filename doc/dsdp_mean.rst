========================================
Differential SDP and Mean Relaxations
========================================

The ``dsdp.py`` module implements differential semidefinite programming relaxations
bridged through SymEngine for symbolic computation. It extends the standard moment/SDP
hierarchy to handle optimization problems where polynomial terms include functions
that are solutions of algebraic differential equations (ADEs).

.. contents::
   :local:
   :depth: 2

Differential SDP Connection
===========================

Standard SDP hierarchies for polynomial optimization work with the cone of sums of
squares and moment matrices over monomial bases. The **differential SDP** extension
handles a broader class of problems by incorporating:

1. **Jet prolongation**: Extending variables to include derivatives :math:`y, y', y'', \dots` up to a fixed order
2. **ADE constraints**: Encoding algebraic differential equations as polynomial constraints on the jet space
3. **Differential moment matrices**: Moment structures that respect the derivation operator

The module provides SymEngine-bridged implementations of these constructs, enabling
symbolic manipulation of differential polynomials before numerical SDP formulation.

Mean Polynomial Forms
=====================

The mean relaxation uses weighted power mean forms as certificates of nonnegativity:

.. math::

   M_{q,p}(X, w) = \sum_{i} w_i X_i^p \left(\sum_{j} w_j X_j^q\right)^{\frac{p-q}{q}}.

These forms generalize both SOS and SONC certificates. The mean polynomial cone
:math:`\mathcal{M}_{n,2d}` contains the SOS cone when :math:`p` divides :math:`2d`,
and strictly contains it for other parameter choices (see the companion manuscript
on mean polynomials, Chapters 1–7).

The DSDP module constructs relaxations using:

- **Mean-based moment matrices**: PSD constraints on generalized moment structures derived from power means
- **SymEngine polynomial arithmetic**: Exact symbolic manipulation before numerical evaluation
- **Derivation-aware basis construction**: Monomial bases that respect the derivation operator structure

API Overview
============

.. code-block:: python

   from Irene.dsdp import DSDPRelaxations

   # Construct differential SDP relaxation for a problem with ADE constraints
   dsdp = DSDPRelaxations(prog, jet_order=2)

   # Solve at specified order
   result = dsdp.solve(order=2)
   print(f"Differential SDP bound: {result['value']:.6f}")

The ``DSDPRelaxations`` class accepts an ``OptimizationProblem`` and a ``jet_order``
parameter controlling the derivative depth. The solve method returns results in the
same dictionary format as standard relaxations (keys: ``value``, ``status``, ``order``,
``basis_size``, timing fields).

SymEngine Bridging
==================

The module uses SymEngine for symbolic polynomial arithmetic with automatic fallback
to SymPy when SymEngine's Poly API lacks a required operation. This dual-engine design
ensures correctness while maximizing performance for the common case. Key bridged
operations include:

- Polynomial multiplication and addition in the jet space
- Derivation operator application (:math:`d_x, d_y` as operators, not Leibniz fractions)
- Ideal membership testing via border basis reduction

Practical Notes
===============

1. The DSDP module is research-grade — it implements the theoretical framework from
   the differential Positivstellensatz rough ideas but should be validated against
   known benchmarks before production use
2. Jet prolongation increases problem dimension by a factor of :math:`(jet\_order + 1)`
   per differentiated variable
3. The SymEngine bridge adds minimal overhead for small problems but provides significant
   speedups for symbolic preprocessing at higher orders

References
==========

- Ghasemi, M. (2026). "Mean Polynomials: Generalizing SOS and SONC via Power Mean Forms." *Companion manuscript*, Chapters 1–7.
- Ritt, J. F. (1950). "Differential Algebra." American Mathematical Society Colloquium Publications.
- Kolchin, E. R. (1973). "Differential Algebra and Algebraic Groups." Academic Press.
