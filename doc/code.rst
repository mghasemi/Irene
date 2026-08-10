Code Documentation
=================================

.. automodule:: Irene.base
   :members:

.. automodule:: Irene.relaxations
   :members:

.. automodule:: Irene.sdp
   :members:

.. automodule:: Irene.program
   :members:

.. automodule:: Irene.grouprings
   :members:

.. automodule:: Irene.geometric
   :members:

.. automodule:: Irene.sonc
   :members:

.. automodule:: Irene.sosonc
   :members:

.. automodule:: Irene.border_basis
   :members:

.. automodule:: Irene.sparsity
   :members:

.. automodule:: Irene.newton_polytope
   :members:

.. automodule:: Irene.relaxation_api
   :members:

.. automodule:: Irene.symbolic_engine
   :members:

.. automodule:: Irene.cvxpy_solver
   :members:

.. automodule:: Irene.dsdp
   :members:

.. automodule:: Irene.telemetry
   :members:

.. automodule:: Irene.matrices
   :members:

.. automodule:: Irene.invariant
   :members:

Doctest Integration
===================

To ensure that code snippets in docstrings remain synchronized with the
IreneRewrite codebase, Sphinx can be configured to run ``doctest`` blocks
during documentation builds.

Enable in ``conf.py``:

.. code-block:: python

   extensions = [
       # ... other extensions ...
       'sphinx.ext.doctest',
   ]

Then run as part of the documentation build pipeline:

.. code-block:: bash

   cd doc
   make doctest

Alternatively, run via pytest against the installed package:

.. code-block:: bash

   .venv/bin/python3 -m pytest --doctest-modules Irene/

The following modules are doctest-ready (their docstrings contain executable
examples): ``relaxation_api.py``, ``program.py``, ``border_basis.py``.  See
:doc:`benchmarks` for the gallery-based integration test suite that serves as
the primary verification layer.
