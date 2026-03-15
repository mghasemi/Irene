===================
Appendix
===================

.. _pyProximationRef:

pyProximation
===================

`pyProximation <https://github.com/mghasemi/pyProximation>`_ is a companion library
for function approximation in Hilbert spaces. Full documentation is included in this
manual — see the :doc:`pyProximation <pyprox_intro>` section.

.. _pyOptRef:

pyOpt
===================

`pyOpt <http://www.pyopt.org/>`_ is a Python-based package for formulating and solving 
nonlinear constrained optimization problems in an efficient, reusable and portable manner.
It is an open-source software distributed under the terms of the 
`GNU Lesser General Public License <http://www.gnu.org/licenses/lgpl.html>`_.

`pyOpt` provides unified interface to the following nonlinear optimizers:
	+ SNOPT - Sparse NOlinear OPTimizer
	+ NLPQL - Non-Linear Programming by Quadratic Lagrangian
	+ NLPQLP - NonLinear Programming with Non-Monotone and Distributed Line Search
	+ FSQP - Feasible Sequential Quadratic Programming
	+ SLSQP - Sequential Least Squares Programming
	+ PSQP - Preconditioned Sequential Quadratic Programming
	+ ALGENCAN - Augmented Lagrangian with GENCAN
	+ FILTERSD
	+ MMA - Method of Moving Asymptotes
	+ GCMMA - Globally Convergent Method of Moving Asymptotes
	+ CONMIN - CONstrained function MINimization
	+ MMFD - Modified Method of Feasible Directions
	+ KSOPT - Kreisselmeier–Steinhauser Optimizer
	+ COBYLA - Constrained Optimization BY Linear Approximation
	+ SDPEN - Sequential Penalty Derivative-free method for Nonlinear constrained optimization
	+ SOLVOPT - SOLver for local OPTimization problems
	+ ALPSO - Augmented Lagrangian Particle Swarm Optimizer
	+ NSGA2 - Non Sorting Genetic Algorithm II
	+ ALHSO - Augmented Lagrangian Harmony Search Optimizer
	+ MIDACO - Mixed Integer Distributed Ant Colony Optimization

Basic usage:
-------------------
`pyOpt` is design to solve general constrained nonlinear optimization problems:

.. math::
	\left\lbrace
	\begin{array}{lll}
		\min & f(x) & \\
		\textrm{Subject to} & & \\
		& g_j(x) = 0 & j=1,\dots,m_e\\
		& g_j(x)\leq0 & j=m_e+1,\dots,m\\
		& l_i\leq x_i\leq u_i & i=1,\dots,n,
	\end{array}
	\right.

where:
	+ :math:`x` is the vector of design variables
	+ :math:`f(x)` is a nonlinear function
	+ :math:`g(x)` is a linear or nonlinear function
	+ :math:`n` is the number of design variables
	+ :math:`m_e` is the number of equality constraints
	+ :math:`m` is the total number of constraints (number of equality constraints: :math:`m_i=m-m_e`).

The following is a pseudo-code demonstrating the basic usage of ``pyOpt``::

	# General Objective Function Template:
	def obj_fun(x, *args, **kwargs):
		"""
		f: objective value
		g: array (or list) of constraint values
		fail: 0 for successful function evaluation, 1 for unsuccessful function evaluation (test must be provided by user)
		If the Optimization problem is unconstrained, g must be returned as an empty list or array: g = []
		Inequality constraints are handled as `<=`.
		"""
		fail = 0
		f = function(x,*args,**kwargs)
		g = function(x,*args,**kwargs)

		return f,g,fail

	# Instantiating an Optimization Problem:
	opt_prob = Optimization('name', obj_fun)
	# Assigning Objective:
	opt_prob.addObj('name', value=0.0, optimum=0.0)
	# Single Design variable:
	opt_prob.addVar('name', type='c', value=0.0, lower=-inf, upper=inf, choices=listochoices)
	# A Group of Design Variables:
	opt_prob.addVarGroup('name', numerinGroup, type='c', value=value, lower=lb, upper=up, choices=listochoices)
	# where `value`, `lb`, `ub` (float or int or list or 1Darray).
	# and supported Types are ‘c’: continuous design variable;
	# `i`: integer design variable; 
	# `d`: discrete design variable (based on choices, e.g.: list/dict of materials).
	# Assigning Constraints:
	## Single Constraint:
	opt_prob.addCon('name', type='i', lower=-inf, upper=inf, equal=0.0)
	## A Group of Constraints:
	opt_prob.addConGroup('name', numberinGroup, type='i', lower=lb, upper=up, equal=eq)
	# where `lb`, `ub`, `eq` are (float or int or list or 1Darray).
	# and supported types are 
	# `i` - inequality constraint;
	# `e` - equality constraint.

	# Instantiating an Optimizer (e.g.: Snopt):
	opt = pySNOPT.SNOPT()
	# Solving the Optimization Problem:
	opt(opt_prob, sens_type='FD', disp_opts=False, sens_mode='', *args, **kwargs)
	# Output:
	print opt_prob

For more details, see `pyOpt documentation <http://www.pyopt.org/quickguide/quickguide.html>`_.

LaTeX support
===================

There is a method implemented for every class of Irene module that generates a LaTeX code related
to the object. This code can be retrieved by calling ``LaTeX`` function on the instance.

Calling ``LaTeX`` on a

	- ``SDPRelaxations`` instance returns the code which demonstrates user entered optimization problem.
	- ``SDRelaxSol`` instance returns the code for the moment matrix of the solution.
	- ``Mom`` instance (moment object) returns the LaTeX code of the object.
	- ``sdp`` instance returns the code for the SDP.

Moreover, if ``LaTeX`` function is called on a ``SymPy`` object it calls ``sympy.latex`` function and returns 
the output.

The ``LaTeX`` function is a member of ``Irene.base`` module which calls ``__latex__`` method of its classes.