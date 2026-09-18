============================
Examples and Validation
============================

This chapter lists runnable entry points that exercise the three method families.
All example scripts live in the ``examples/`` directory at the repository root
(except the benchmark runners, which live in ``benchmarks/``).

Recommended Example Sequence
=============================

1. ``examples/Rosenbrock.py`` — SDP hierarchy on a classic benchmark.
2. ``examples/GPExample.py`` — geometric relaxation flow via GP.
3. ``examples/SONCExample.py`` and ``examples/SONCExample33.py`` — SONC circuit-polynomial relaxations.

API Quick Reference
===================

.. csv-table::
   :header: "Script", "Primary classes", "Solver dependency"

   "``examples/Rosenbrock.py``", "``SDPRelaxations``", "CVXPY/CLARABEL (default) or CVXOPT"
   "``examples/GPExample.py``", "``OptimizationProblem``, ``GPRelaxations``", "GP solver backend"
   "``examples/SONCExample.py``", "``OptimizationProblem``, ``SONCRelaxations``", "GP solver backend"
   "``examples/SONCExample33.py``", "``OptimizationProblem``, ``SONCRelaxations``", "GP solver backend"

SDP Example
===========

Run from the repository root with the virtual environment activated::

    source .venv/bin/activate
    python examples/Rosenbrock.py

Expected behavior:

1. Initializes an ``SDPRelaxations`` object with semigroup-algebra expressions.
2. Solves an SDP lower-bound problem via the selected solver (CLARABEL by default).
3. Prints solver summary, objective values, and timing information.

Geometric Programming Example
==============================

Run::

    python examples/GPExample.py

Expected behavior:

1. Builds an ``OptimizationProblem`` from semigroup-algebra expressions.
2. Constructs a ``GPRelaxations`` model with transformation matrix :math:`H`.
3. Prints transformation matrix information and GP solution details.

SONC Examples
=============

Run::

    python examples/SONCExample.py
    python examples/SONCExample33.py

Expected behavior:

1. Builds constrained SONC models from semigroup-algebra expressions.
2. Prints a certified lower bound when the solver succeeds.
3. Reports runtime and solver status if GP solving is unavailable in the current environment.

Example 3.3 Traceability
========================

The script ``examples/SONCExample33.py`` is aligned with the Section 3.3 benchmark
used in the repository and is paired with checks in ``tests/test_sonc_section3.py``.

Benchmark Gallery System
========================

IreneRewrite includes a gallery-based benchmark runner for systematic comparison
across relaxation families:

* ``benchmarks/gallery.yaml`` — YAML configuration defining problem sets, parameters, and solver options.
* ``benchmarks/run_gallery.py`` — Entry point that loads the gallery config and runs all configured benchmarks.

Run::

    python benchmarks/run_gallery.py

This exercises SDP, GP, SONC, and SOSONC relaxations on a curated set of problems
and writes structured results to ``benchmarks/results/``.

Backend Comparison Benchmark
============================

The comprehensive cross-version / cross-backend benchmark runs the same feature
set through original Irene (SymPy), IreneRewrite with SymEngine, and IreneRewrite
forced to the SymPy backend::

    benchmarks/benchmark_backends.py --mode irene
    benchmarks/benchmark_backends.py --mode irene_rewrite
    benchmarks/benchmark_backends.py --mode irene_rewrite_sympy

Each mode covers SOS/SONC/SOSONC relaxations, GP, DSDP mean and KKT relaxations,
ADE relation building, border bases, correlative sparsity, Newton polytope
pruning, and symbolic-engine micro-benchmarks.

Additional Examples
===================

The following scripts exercise mean polynomial forms, separating polynomials,
and other specialized relaxation techniques:

* ``examples/pqforms.py`` — Power-mean form certificates for nonnegativity.
* ``examples/SOSONCSchickSeparating.py`` — Schick's SOS+SONC separating example.
* ``examples/Rosenbrock.py``, ``examples/Giunta.py``, ``examples/Parsopoulos.py`` — Classic benchmark problems.
* ``examples/McCormick.py`` — McCormick relaxation on non-convex objective.
* ``benchmarks/compare_irene_vs_rewrite.py`` — Cross-version comparison of original Irene vs IreneRewrite results.

Regression Validation
=====================

Run the test suite from the repository root with the virtual environment activated::

    source .venv/bin/activate
    python -m pytest Irene/tests/ tests/ -q

This is the recommended consistency check after modifying optimization modules
or documentation examples. Individual test files can be run separately, e.g.::

    python -m pytest tests/test_sosonc.py -v
