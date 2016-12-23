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

Example 1:
-------------------

*Find the minimum of* :math:`x + e^{x\sin x}` *where* :math:`-\pi\leq x\leq \pi`.

The objective function includes terms of :math:`x` and transcendental functions. So, it is 
difficult to find a suitable algebraic representation to transform this optimization problem.
Let us try to use Taylor expansion of :math:`e^{x\sin x}` to find an approximation for the 
optimum and compare the result with ``scipy.optimize``, ``pyOpt.ALPSO`` and ``pyOpt.NSGA2``::

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

	# pyOpt
	from pyOpt import *


	def objfunc(x):
	    from numpy import sin, exp, pi
	    f = x[0] + exp(x[0] * sin(x[0]))
	    g = [
	        x[0]**2 - pi**2
	    ]
	    fail = 0
	    return f, g, fail

	opt_prob = Optimization('A mixed function', objfunc)
	opt_prob.addVar('x1', 'c', lower=-pi, upper=pi, value=0.0)
	opt_prob.addObj('f')
	opt_prob.addCon('g1', 'i')
	# Augmented Lagrangian Particle Swarm Optimizer
	alpso = ALPSO()
	alpso(opt_prob)
	print opt_prob.solution(0)
	# Non Sorting Genetic Algorithm II
	nsg2 = NSGA2()
	nsg2(opt_prob)
	print opt_prob.solution(1)

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
	
	ALPSO Solution to A mixed function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.0683
	    Total Function Evaluations:      1240
	    Lambda:     [ 0.]
	    Seed: 1482112089.31088901

	    Objectives:
	        Name        Value        Optimum
		     f        -2.14159             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	     -3.141593      -3.14e+00     3.14e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= 0.000000 <= 0.00e+00

	--------------------------------------------------------------------------------


	NSGA-II Solution to A mixed function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.4231
	    Total Function Evaluations:          

	    Objectives:
	        Name        Value        Optimum
		     f        -2.14159             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	     -3.141593      -3.14e+00     3.14e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= 0.000000 <= 0.00e+00

	--------------------------------------------------------------------------------


Now instead of Taylor expansion, we use Legendre polynomials to estimate :math:`e^{x\sin x}`.
To find Legendre estimators, we use `pyProximation <https://github.com/mghasemi/pyProximation>`_ which
implements general Hilbert space methods (see Appendix-:ref:`pyProximationRef`)::

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

Example 2:
-------------------

*Find the minimum of* :math:`x\sinh y + e^{y\sin x}` *where* :math:`-\pi\leq x, y\leq\pi`.

Again, we use Legendre approximations for :math:`\sinh y` and :math:`e^{y\sin x}`::

	from sympy import *
	from Irene import *
	from pyProximation import OrthSystem
	# introduce symbols and functions
	x = Symbol('x')
	y = Symbol('y')
	sh = Function('sh')(y)
	ch = Function('ch')(y)
	# transcendental term of objective
	f = exp(y * sin(x))
	g = sinh(y)
	# Legendre polynomials via pyProximation
	D_f = [(-pi, pi), (-pi, pi)]
	D_g = [(-pi, pi)]
	Orth_f = OrthSystem([x, y], D_f)
	Orth_g = OrthSystem([y], D_g)
	# set bases
	B_f = Orth_f.PolyBasis(10)
	B_g = Orth_g.PolyBasis(10)
	# link B_f to Orth_f and B_g to Orth_g
	Orth_f.Basis(B_f)
	Orth_g.Basis(B_g)
	# generate the orthonormal bases
	Orth_f.FormBasis()
	Orth_g.FormBasis()
	# extract the coefficients of approximations
	Coeffs_f = Orth_f.Series(f)
	Coeffs_g = Orth_g.Series(g)
	# form the approximations
	f_app = sum([Orth_f.OrthBase[i] * Coeffs_f[i]
	             for i in range(len(Orth_f.OrthBase))])
	g_app = sum([Orth_g.OrthBase[i] * Coeffs_g[i]
	             for i in range(len(Orth_g.OrthBase))])
	# initiate the Relaxation object
	Rlx = SDPRelaxations([x, y])
	# set the objective
	Rlx.SetObjective(x * g_app + f_app)
	# add support constraints
	Rlx.AddConstraint(pi**2 - x**2 >= 0)
	Rlx.AddConstraint(pi**2 - y**2 >= 0)
	# set the sdp solver
	Rlx.SetSDPSolver('cvxopt')
	# initialize the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution
	# using scipy
	from scipy.optimize import minimize
	fun = lambda x: x[0] * sinh(x[1]) + exp(x[1] * sin(x[0]))
	cons = (
	    {'type': 'ineq', 'fun': lambda x: pi**2 - x[0]**2},
	    {'type': 'ineq', 'fun': lambda x: pi**2 - x[1]**2}
	)
	sol1 = minimize(fun, (0, 0), method='COBYLA', constraints=cons)
	sol2 = minimize(fun, (0, 0), method='SLSQP', constraints=cons)
	print "solution according to 'COBYLA':"
	print sol1
	print "solution according to 'SLSQP':"
	print sol2

	# pyOpt
	from pyOpt import *


	def objfunc(x):
	    from numpy import sin, sinh, exp, pi
	    f = x[0] * sinh(x[1]) + exp(x[1] * sin(x[0]))
	    g = [
	        x[0]**2 - pi**2,
	        x[1]**2 - pi**2
	    ]
	    fail = 0
	    return f, g, fail

	opt_prob = Optimization(
	    'A trigonometric-hyperbolic-exponential function', objfunc)
	opt_prob.addVar('x1', 'c', lower=-pi, upper=pi, value=0.0)
	opt_prob.addVar('x2', 'c', lower=-pi, upper=pi, value=0.0)
	opt_prob.addObj('f')
	opt_prob.addCon('g1', 'i')
	opt_prob.addCon('g2', 'i')
	# Augmented Lagrangian Particle Swarm Optimizer
	alpso = ALPSO()
	alpso(opt_prob)
	print opt_prob.solution(0)
	# Non Sorting Genetic Algorithm II
	nsg2 = NSGA2()
	nsg2(opt_prob)
	print opt_prob.solution(1)
	
The result will be::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 4.09241986275 seconds
	              Run Time: 0.123869 seconds
	Primal Objective Value: -35.3574475835
	  Dual Objective Value: -35.3574473266
	Feasible solution for moments of order 5

	solution according to 'COBYLA':
	     fun: 1.0
	   maxcv: 0.0
	 message: 'Optimization terminated successfully.'
	    nfev: 13
	  status: 1
	 success: True
	       x: array([ 0.,  0.])
	solution according to 'SLSQP':
	     fun: 1
	     jac: array([ 0.,  0.,  0.])
	 message: 'Optimization terminated successfully.'
	    nfev: 4
	     nit: 1
	    njev: 1
	  status: 0
	 success: True
	       x: array([ 0.,  0.])

	ALPSO Solution to A trigonometric-hyperbolic-exponential function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.0946
	    Total Function Evaluations:      1240
	    Lambda: [ 0.  0.]
	    Seed: 1482112613.82665610

	    Objectives:
	        Name        Value        Optimum
		     f        -35.2814             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	     -3.141593      -3.14e+00     3.14e+00 
		     x2       c	      3.141593      -3.14e+00     3.14e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= 0.000000 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= 0.000000 <= 0.00e+00

	--------------------------------------------------------------------------------


	NSGA-II Solution to A trigonometric-hyperbolic-exponential function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.5331
	    Total Function Evaluations:          

	    Objectives:
	        Name        Value        Optimum
		     f        -35.2814             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      3.141593      -3.14e+00     3.14e+00 
		     x2       c	     -3.141593      -3.14e+00     3.14e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= 0.000000 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= 0.000000 <= 0.00e+00

	--------------------------------------------------------------------------------

which shows a significant improvement compare to results of ``scipi.minimize``.