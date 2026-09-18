=====================
Introduction
=====================

This is a brief documentation for using *Irene*.
Irene was originally written to find reliable approximations
for optimum value of an arbitrary optimization problem.
It implements a modification of Lasserre's SDP Relaxations based
on generalized truncated moment problem to handle general optimization
problems algebraically.

Documentation Scope
===============================

The documentation now covers three complementary pathways for constrained
polynomial optimization:

1. SDP hierarchy methods (moment/SOS based).
2. Geometric programming relaxations.
3. SONC relaxations.

The recommended reading path is:

1. ``architecture`` for conceptual map.
2. ``algebra`` and ``program`` for representation layer.
3. ``sdp``, ``geometric``, and ``sonc`` for method-specific formulations.

Requirements and dependencies
===============================

This is a python package, so clearly python is an obvious requirement.
Irene relies on the following packages:

    + for vector calculations:
        - `NumPy <http://www.numpy.org/>`_.
        - `SciPy <https://www.scipy.org/>`_.
    + for symbolic computations:
        - `SymPy <http://www.sympy.org/>`_.
        - `SymEngine <https://symengine.org/>`_ (primary engine; SymPy fallback).
    + for semidefinite optimization (choose one path):
        - **CVXPY** (recommended default) with `CLARABEL <https://github.com/oxfordcontrol/Clarabel.rs>`_ backend,
        - `cvxopt <http://cvxopt.org/>`_ (legacy native path),
        - `dsdp <http://www.mcs.anl.gov/hs/software/DSDP/>`_,
        - `sdpa <http://sdpa.sourceforge.net/>`_,
        - `csdp <https://projects.coin-or.org/Csdp/>`_.

Dependency Matrix by Method Family
----------------------------------

.. list-table::
     :header-rows: 1

     * - Method family
       - Core Python packages
       - Optional packages
       - External solver requirement
     * - SDP relaxations
       - numpy, scipy, sympy, symengine
       - cvxpy, clarabel
       - one of cvxpy (default), cvxopt, dsdp, sdpa, csdp
     * - Geometric relaxations
       - numpy, scipy, sympy
       - gpkit
       - gpkit-supported GP backend
     * - SONC relaxations
       - numpy, scipy, sympy
       - gpkit
       - gpkit-supported GP backend

Solver Prerequisites
--------------------

Before running examples, verify available solvers from Python::

    from Irene.base import base
    print(base().AvailableSDPSolvers())

Quick Validation Workflow
-------------------------

After installation, the following commands provide a practical smoke test::

    python benchmarks/Rosenbrock.py
    python benchmarks/GPExample.py
    python benchmarks/SONCExample.py
    python -m pytest tests/ -q

Solver Troubleshooting
----------------------

Common runtime signatures and first actions:

1. ``AvailableSDPSolvers()`` returns an empty list.

    This indicates that no configured SDP backend is currently reachable.
    Install at least one supported solver and verify it is available on ``PATH``
    (or configured in solver path settings on platforms that require explicit paths).

2. ``RuntimeError: GP solve failed`` or ``RuntimeError: SONC GP solve failed``.

    These messages usually indicate missing GP backend support, an unavailable solver,
    or an infeasible/numerically unstable relaxation for the selected formulation.
    Start with the shipped examples, lower verbosity, and simplified instances.

3. ``ModuleNotFoundError: No module named 'gpkit'``.

    Install ``gpkit`` before running geometric or SONC examples.

4. SDP solve runs but returns non-optimal status.

    Try another supported SDP solver, inspect constraints for scaling issues,
    and compare with a lower relaxation order before increasing model complexity.


Download
================

`IreneRewrite <https://github.com/mghasemi/Irene>`_ can be obtained from GitHub.

Installation
=========================

**Using ``venv`` (recommended)**::

    cd IreneRewrite
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -e ".[dev]"

**Using ``uv`` (faster alternative)**::

    cd IreneRewrite
    uv venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"

Both methods create an editable installation with development dependencies.
The virtual environment isolates solver backends and avoids system-wide conflicts.

Documentation
--------------------------
The documentation of `Irene` is prepared via `Sphinx <http://www.sphinx-doc.org/>`_.

To compile html version of the documentation run::

    cd doc
    make html

To make a pdf file, subject to existence of ``latexpdf`` run::

    cd doc
    make latexpdf

License
=======================
`Irene` is distributed under `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_:

MIT License
------------------

    Copyright (c) 2016-2026 Mehdi Ghasemi

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
