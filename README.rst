=============================
Irene
=============================

Irene is a Python toolkit for constrained
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
- Versioning: stable 1.x line — current version defined in ``pyproject.toml`` (see ``CHANGELOG.md``)
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

Structural reduction modules (border basis, correlative sparsity detection, Newton polytope
pruning) are implemented, unit-tested, and integrated into the unified relaxation pipeline
(``relaxation_api.py`` / ``relaxations.py``); they are opt-in via ``RelaxationConfig`` for
backward compatibility. On dense instances the certified-safe Newton-pruning retention set
currently retains the full basis by design (no size reduction) until certified sparsity sets
are available for sparse instances — see the paper's Section 4 for measured behavior.

Reduction defaults:

- Structural reductions are opt-in for compatibility. Pass ``RelaxationConfig`` to
  ``RelaxationEngine`` to enable Newton-polytope pruning, border-basis reduction, or
  correlative sparsity detection.

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
============================

From PyPI::

    pip install irene

or from source (development)::

    pip install .[symengine,solvers]

Development dependencies::

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

Read the hosted documentation at `irene.readthedocs.io <https://irene.readthedocs.io/>`_.
Documentation sources are in ``doc/`` and include architecture, migration, solver, and benchmark
chapters.

Build docs locally::

	make -C doc html

Related Work
============================

The theoretical foundations of the relaxation hierarchies implemented here are developed in:

- M. Ghasemi, J. B. Lasserre, M. Marshall, *Lower Bounds on the Global Minimum of a Polynomial*, Computational Optimization and Applications 56(1), 2013. `arXiv:1209.3049 <https://arxiv.org/abs/1209.3049>`_
- M. Ghasemi, M. Marshall, *Lower Bounds for Polynomials Using Geometric Programming*, SIAM Journal on Optimization 22(2), 460–473, 2012. `DOI:10.1137/110836869 <https://doi.org/10.1137/110836869>`_

License
=============================

Irene is distributed under the `MIT License <https://en.wikipedia.org/wiki/MIT_License>`_.
