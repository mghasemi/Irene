=============================
Approximating Optimum Value
=============================

In various cases, separating functions and symbols are either very difficult or impossible.
For example :math:`x, \sin x` and :math:`e^x` are not algebraically independent, but their 
dependency can not be easily expressed in finitely many relations. 
One possible approach to these problems is replacing transcendental terms with a reasonably
good approximation. This certainly will introduce more numerical error, but at least gives
a reliable estimate for the optimum value.


Using ``pyProximation.OrthSystem``
====================================

A simple and common method to approximate transcendental functions is using truncated Taylor
expansions. In spite of its simplicity, there are various pitfalls which needs to be avoided.
The most common is that the radius of convergence of the Taylor expansion may be smaller than
the feasibility region of the optimization problem.

Example:
-------------------

*Find the minimum of* :math:`x + e^{x\sin x}` *where* :math:`-\pi\leq x\leq \pi`.

The objective function includes terms of :math:`x` and transcendental functions. So, it is 
difficult to find a suitable algebraic representation to transform this optimization problem.
Let us try to use Taylor expansion of :math:`e^{x\sin x}` to find an approximation for the 
optimum and compare the result with ``scipy.optimize`` and ``pyswarm.pso``::

	from sympy import *
	from Irene import *
	# introduce symbols and functions
	x = Symbol('x')
	e = Function('e')(x)
	# transcendental term of objective
	f = exp(x * sin(x))
	# Taylor expansion
	f_app = f.series(x, 0, 12).removeO()
	# initiate the Relaxation object
	Rlx = SDPRelaxations([x])
	# set the objective
	Rlx.SetObjective(x + f_app)
	# add support constraints
	Rlx.AddConstraint(pi**2 - x**2 >= 0)
	# initialize the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution
	# using scipy
	from scipy.optimize import minimize
	fun = lambda x: x[0] + exp(x[0] * sin(x[0]))
	cons = (
	    {'type': 'ineq', 'fun': lambda x: pi**2 - x[0]**2},
	)
	sol1 = minimize(fun, (0, 0), method='COBYLA', constraints=cons)
	sol2 = minimize(fun, (0, 0), method='SLSQP', constraints=cons)
	print "solution according to 'COBYLA':"
	print sol1
	print "solution according to 'SLSQP':"
	print sol2
	# particle swarm optimization
	from pyswarm import pso
	lb = [-3.3, -3.3]
	ub = [3.3, 3.3]
	cns = [cons[0]['fun']]
	print "PSO:"
	print pso(fun, lb, ub, ieqcons=cns)

The output will look like::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 0.270121097565 seconds
	              Run Time: 0.012974 seconds
	Primal Objective Value: -416.628918881
	  Dual Objective Value: -416.628917197
	Feasible solution for moments of order 5

	solution according to 'COBYLA':
	     fun: 0.76611902154788758
	   maxcv: 0.0
	 message: 'Optimization terminated successfully.'
	    nfev: 34
	  status: 1
	 success: True
	       x: array([ -4.42161128e-01,  -9.76206736e-05])
	solution according to 'SLSQP':
	     fun: 0.766119450232887
	     jac: array([ 0.00154828,  0.        ,  0.        ])
	 message: 'Optimization terminated successfully.'
	    nfev: 17
	     nit: 4
	    njev: 4
	  status: 0
	 success: True
	       x: array([-0.44164406,  0.        ])
	PSO:
	Stopping search: Swarm best objective change less than 1e-08
	(array([-3.14159265,  1.94020281]), -2.1415926274654815)

Now instead of Taylor expansion, we use Legendre polynomials to estimate :math:`e^{x\sin x}`::

	from sympy import *
	from Irene import *
	from pyProximation import OrthSystem
	# introduce symbols and functions
	x = Symbol('x')
	e = Function('e')(x)
	# transcendental term of objective
	f = exp(x * sin(x))
	# Legendre polynomials via pyProximation
	D = [(-pi, pi)]
	S = OrthSystem([x], D)
	# set B = {1, x, x^2, ..., x^12}
	B = S.PolyBasis(12)
	# link B to S
	S.Basis(B)
	# generate the orthonormal basis
	S.FormBasis()
	# extract the coefficients of approximation
	Coeffs = S.Series(f)
	# form the approximation
	f_app = sum([S.OrthBase[i] * Coeffs[i] for i in range(len(S.OrthBase))])
	# initiate the Relaxation object
	Rlx = SDPRelaxations([x])
	# set the objective
	Rlx.SetObjective(x + f_app)
	# add support constraints
	Rlx.AddConstraint(pi**2 - x**2 >= 0)
	# set the solver
	Rlx.SetSDPSolver('dsdp')
	# initialize the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution

The output will be::

	Solution of a Semidefinite Program:
	                Solver: DSDP
	                Status: Optimal
	   Initialization Time: 0.722383022308 seconds
	              Run Time: 0.077674 seconds
	Primal Objective Value: -2.26145824829
	  Dual Objective Value: -2.26145802066
	Feasible solution for moments of order 6

By a small modification of the above code, we can employ Chebyshev polynomials for approximation::

	from sympy import *
	from Irene import *
	from pyProximation import Measure, OrthSystem
	# introduce symbols and functions
	x = Symbol('x')
	e = Function('e')(x)
	# transcendental term of objective
	f = exp(x * sin(x))
	# Chebyshev polynomials via pyProximation
	D = [(-pi, pi)]
	# the Chebyshev weight
	w = lambda x: 1. / sqrt(pi**2 - x**2)
	M = Measure(D, w)
	S = OrthSystem([x], D)
	# link the measure to S
	S.SetMeasure(M)
	# set B = {1, x, x^2, ..., x^12}
	B = S.PolyBasis(12)
	# link B to S
	S.Basis(B)
	# generate the orthonormal basis
	S.FormBasis()
	# extract the coefficients of approximation
	Coeffs = S.Series(f)
	# form the approximation
	f_app = sum([S.OrthBase[i] * Coeffs[i] for i in range(len(S.OrthBase))])
	# initiate the Relaxation object
	Rlx = SDPRelaxations([x])
	# set the objective
	Rlx.SetObjective(x + f_app)
	# add support constraints
	Rlx.AddConstraint(pi**2 - x**2 >= 0)
	# set the solver
	Rlx.SetSDPSolver('dsdp')
	# initialize the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution

which returns::

	Solution of a Semidefinite Program:
	                Solver: DSDP
	                Status: Optimal
	   Initialization Time: 0.805300951004 seconds
	              Run Time: 0.066767 seconds
	Primal Objective Value: -2.17420785198
	  Dual Objective Value: -2.17420816422
	Feasible solution for moments of order 6

This gives a better approximation for the optimum value. The optimum values found via
Legendre and Chebyshev polynomials are certainly better than Taylor expansion and the
results of ``scipy.optimize``.