=============================
IreneRewrite
=============================

IreneRewrite is the actively developed modernization of Irene, a Python toolkit for constrained
polynomial optimization over commutative real algebras.

It supports multiple relaxation families and backends:

- SOS and moment-SDP relaxations
- SONC relaxations
- hybrid SOS+SONC workflows
- legacy and CVXPY-based SDP solver paths

--------------------------------------
Repository Status (2026-08-09)
--------------------------------------

- Development status: active
- Packaging target: ``2.0.0.dev0`` (defined in ``pyproject.toml``)
- Phases 1-3 modernization work: completed
- Current focus: integration hardening, CI/benchmark automation, and remaining reduction wiring

Implemented modernization highlights in this repository:

- Selectable symbolic backend via ``IRENE_SYMBOLIC_BACKEND`` (SymEngine primary, SymPy fallback)
- CVXPY solver abstraction layer (with Clarabel/SCS/CVXOPT integration)
- Structural reduction modules:

  - border basis
  - correlative sparsity detection
  - Newton polytope pruning

- Unified relaxation entrypoint in ``Irene/relaxation_api.py``
- Runtime telemetry helpers in ``Irene/telemetry.py``

Known current gap:

- Phase 3 reduction modules are implemented and tested, but full end-to-end integration through all
  legacy ``relaxations.py`` code paths is still in progress.

Requirements
=============================

Core runtime (from ``pyproject.toml``):

- Python >= 3.10
- sympy, numpy, scipy
- cvxpy, cvxopt
- gpkit, multiprocess

Optional extras:

- ``.[symengine]`` for SymEngine acceleration
- ``.[solvers]`` for additional conic/QP solvers (clarabel, scs, osqp)
- ``.[dev]`` for testing and coverage tooling

Symbolic backend selection:

- ``IRENE_SYMBOLIC_BACKEND=symengine`` (default when available)
- ``IRENE_SYMBOLIC_BACKEND=sympy``
- ``IRENE_SYMBOLIC_BACKEND=auto``

Installation
=============================

Create and activate a virtual environment, then install from source.

Base install::

	pip install .

Install with SymEngine + solver extras::

	pip install .[symengine,solvers]

Install development dependencies::

	pip install .[dev,symengine,solvers]

Testing
=============================

Run test suites configured in ``pyproject.toml``::

	pytest

The project includes tests under ``Irene/tests/`` and ``tests/``.

Quick Start (Modern API)
=============================

.. code-block:: python

   from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
   from Irene.program import OptimizationProblem
   from Irene.relaxation_api import RelaxationEngine

   sg = CommutativeSemigroup(["x", "y", "z"])
   sga = SemigroupAlgebra(sg)
   x, y, z = sga["x"], sga["y"], sga["z"]

   prog = OptimizationProblem(sga)
   prog.set_objective(-2*x + y - z)
   prog.add_constraint(x + y + z <= 4)
   prog.add_constraint(x >= 0)

   engine = RelaxationEngine(prog, order=2, solver="cvxpy")
   result = engine.solve("sos")
   print(result.status, result.value)

Legacy API Compatibility
=============================

The legacy API remains available for backward compatibility (for example ``SDPRelaxations``).
For new code, prefer ``OptimizationProblem`` + ``RelaxationEngine``.

Documentation
=============================

Documentation sources are in ``doc/`` and include architecture, migration, solver, and benchmark
chapters.

Build docs locally::

	make -C doc html

License
=============================

Irene is distributed under the `MIT License <https://en.wikipedia.org/wiki/MIT_License>`_.
