.. IreneRewrite documentation master file.
   Updated for Phase 3 (2026).

Welcome to IreneRewrite's Documentation!
========================================

IreneRewrite is a Python toolkit for polynomial optimization via semidefinite
programming, geometric programming, and SONC/SOS hierarchies. It implements
Lasserre's moment-SDP hierarchy, circuit-based SONC relaxations, mean polynomial
forms, correlative sparsity detection, Newton polytope pruning, and differential
SDP extensions.

Contents:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   introduction
   architecture
   migration

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   algebra
   program
   sdp
   geometric
   sonc
   sosonc
   optim

.. toctree::
   :maxdepth: 2
   :caption: Phase 3 — Algebraic Reductions

   border_basis
   sparsity
   newton_polytope
   relaxation_api

.. toctree::
   :maxdepth: 2
   :caption: Transcendental & Differential Algebraic Optimization

   approx
   nonpopsdp
   dsdp_mean

.. toctree::
   :maxdepth: 2
   :caption: Solver Layer & Numerical Methods

   cvxpy_solver

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks and Examples

   benchmarks
   examples

.. toctree::
   :maxdepth: 2
   :caption: pyProximation

   pyprox_intro
   pyprox_measures
   pyprox_hilbert
   pyprox_interpolation
   pyprox_code

.. toctree::
   :maxdepth: 2
   :caption: Reference

   code
   rev
   appendix


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
