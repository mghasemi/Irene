=============================
Examples and Validation
=============================

This chapter lists runnable entry points that exercise the three method families.

Recommended Example Sequence
=============================

1. ``examples/Example01.py`` for SDP hierarchy flow.
2. ``examples/GPExample.py`` for geometric relaxation flow.
3. ``examples/SONCExample.py`` and ``examples/SONCExample33.py`` for SONC flow.

API Quick Reference
=============================

.. csv-table::
   :header: "Script", "Primary classes", "Solver dependency"

   "``examples/Example01.py``", "``SDPRelaxations``", "SDP solver (for example ``csdp``, ``sdpa``, ``dsdp``, or ``cvxopt``)"
   "``examples/GPExample.py``", "``OptimizationProblem``, ``GPRelaxations``", "``gpkit`` backend"
   "``examples/SONCExample.py``", "``OptimizationProblem``, ``SONCRelaxations``", "``gpkit`` backend"
   "``examples/SONCExample33.py``", "``OptimizationProblem``, ``SONCRelaxations``", "``gpkit`` backend"

SDP Example
=============================

Run::

   python examples/Example01.py

Expected behavior:

1. Initializes an ``SDPRelaxations`` object with symbolic relations.
2. Solves an SDP lower-bound problem via selected solver.
3. Prints solver summary and objective values.

Geometric Programming Example
=============================

Run::

   python examples/GPExample.py

Expected behavior:

1. Builds an ``OptimizationProblem`` from semigroup-algebra expressions.
2. Constructs a ``GPRelaxations`` model.
3. Prints transformation matrix information and GP solution details.

SONC Examples
=============================

Run::

   python examples/SONCExample.py
   python examples/SONCExample33.py

Expected behavior:

1. Builds constrained SONC models from semigroup-algebra expressions.
2. Prints a lower bound when solver/model setup succeeds.
3. Reports runtime solver status if GP solving is not available in the current environment.

Example 3.3 Traceability
=============================

The script ``examples/SONCExample33.py`` is aligned with the Section 3.3 benchmark
used in the repository and is paired with checks in ``tests/test_sonc_section3.py``.

Regression Validation
=============================

Run the test suite from the repository root::

   python -m unittest discover tests/

This is the recommended consistency check after modifying optimization modules
or documentation examples.
