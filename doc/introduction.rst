=====================
Introduction
=====================

This is a brief documentation for using *Irene*.
Irene was originally written to find reliable approximations
for optimum value of an arbitrary optimization problem.
It implements a modification of Lasserre's SDP Relaxations based
on generalized truncated moment problem to handle general optimization
problems algebraically.

Requirements and dependencies
===============================

This is a python package, so clearly python is an obvious requirement.
Irene relies on the following packages:

	+ for vector calculations:
		- `NumPy <http://www.numpy.org/>`_.
	+ for symbolic computations:
		- `SymPy <http://www.sympy.org/>`_.
	+ for semidefinite optimization, at least one of the following is required:
		- `cvxopt <http://cvxopt.org/>`_,
		- `dsdp <http://www.mcs.anl.gov/hs/software/DSDP/>`_,
		- `sdpa <http://sdpa.sourceforge.net/>`_,
		- `csdp <https://projects.coin-or.org/Csdp/>`_.


Download
================

`Irene` can be obtained from `https://github.com/mghasemi/Irene <https://github.com/mghasemi/Irene>`_.

Installation
=========================

To install `Irene`, run the following in terminal::

	sudo python setup.py install

Documentation
--------------------------
The documentation of `Irene` is prepared via `sphinx <http://www.sphinx-doc.org/>`_.

To compile html version of the documentation run::

	$Irene/doc/make html

To make a pdf file,subject to existence of ``latexpdf`` run::

	$Irene/doc/make latexpdf


License
=======================
`Irene` is distributed under `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_:

MIT License
------------------

	Copyright (c) 2016 Mehdi Ghasemi

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