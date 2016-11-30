=============================
Irene
=============================

*Irene* is a python package that aims to be a toolkit for global optimization problems that can be
realized algebraically. It generalizes Lasserre's Relaxation method to handle theoretically any
optimization problem with bounded feasibility set. The method is based on solutions of generalized 
truncated moment problem over commutative real algebras.

Requirements
=============================

For symbolic computations *Irene* depends on `SymPy <http://www.sympy.org/en/index.html>`_ and for 
numeric computations uses `NumPy <http://www.numpy.org/>`_.

To solve semidefinite programs, at least one of the following solvers must be available:
	- `cvxopt <http://cvxopt.org/>`_,
	- `dsdp <http://www.mcs.anl.gov/hs/software/DSDP/>`_,
	- `sdpa <http://sdpa.sourceforge.net/>`_,
	- `csdp <https://projects.coin-or.org/Csdp/>`_.

Installation
=============================

To obtain *Irene* visit `https://github.com/mghasemi/Irene <https://github.com/mghasemi/Irene>`_.

For system-wide installation run::

	sudo python setup.py install

License
=============================
`Irene` is distributed under `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_.
