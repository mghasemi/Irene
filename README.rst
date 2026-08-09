=============================
Irene
=============================

*Irene* is a python package that aims to be a toolkit for global optimization problems that can be
realized algebraically. It generalizes Lasserre's Relaxation method to handle theoretically any
optimization problem with bounded feasibility set. The method is based on solutions of generalized 
truncated moment problem over commutative real algebras.

IreneRewrite (this tree) is the modernized implementation: it adds a user-selectable
symbolic backend (SymEngine primary, SymPy fallback — or pure SymPy), a CVXPY solver
abstraction layer, structural reductions (border bases, correlative sparsity, Newton
polytope pruning), a unified relaxation API, and execution telemetry.

Requirements
=============================

For symbolic computations *Irene* depends on `SymPy <http://www.sympy.org/en/index.html>`_ and
optionally on `SymEngine <https://symengine.org/>`_ (the C++ backend, enabled by default when
installed). The symbolic backend is user-selectable:

- ``IRENE_SYMBOLIC_BACKEND=symengine`` (default when installed) — C++ engine with SymPy fallback
- ``IRENE_SYMBOLIC_BACKEND=sympy`` — pure SymPy mode
- ``IRENE_SYMBOLIC_BACKEND=auto`` — prefer SymEngine when available

For numeric computations it uses `NumPy <http://www.numpy.org/>`_.

To solve semidefinite programs, at least one of the following solvers must be available:
	- `cvxopt <http://cvxopt.org/>`_,
	- `cvxpy <https://www.cvxpy.org/>`_ with CLARABEL/SCS (modern path),
	- `dsdp <http://www.mcs.anl.gov/hs/software/DSDP/>`_,
	- `sdpa <http://sdpa.sourceforge.net/>`_,
	- `csdp <https://projects.coin-or.org/Csdp/>`_.

Installation
=============================

To obtain *Irene* visit `https://github.com/mghasemi/Irene <https://github.com/mghasemi/Irene>`_.

For more details refer to the `documentation <http://irene.readthedocs.io/>`_.

For system-wide installation run::

	sudo python setup.py install

To install with the SymEngine C++ backend enabled (recommended for performance)::

	pip install .[symengine]

License
=============================
`Irene` is distributed under `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_.
