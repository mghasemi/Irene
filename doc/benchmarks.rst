==================================
Benchmark Optimization Problems
==================================

There are benchmark problems to evaluated how good an optimization method works.
We apply the generalized relaxation method to some of these benchmarks that are
mainly taken from [MJXY]_.

.. [MJXY] M\. Jamil, Xin-She Yang, *A literature survey of benchmark functions for global optimization problems*, IJMMNO, Vol. 4(2), 2013.

Rosenbrock Function
==================================

The original Rosenbrock function is :math:`f(x, y)=(1-x)^2 + 100(y-x^2)^2` 
which is a sums of squares and attains its minimum at :math:`(1, 1)`.
The global minimum is inside a long, narrow, parabolic shaped flat valley. 
To find the valley is trivial. To converge to the global minimum, however, 
is difficult.
The same holds for a generalized form of Rosenbrock function which is defined as:

.. math::
	f(x_1,\dots,x_n) = \sum_{i=1}^{n-1} 100(x_{i+1} - x_i^2)^2+(1-x_i)^2.

Since :math:`f` is a sum of squares, and :math:`f(1,\dots,1)=0`, the global 
minimum is equal to 0. The following code compares various optimization 
methods including the relaxation method, to find a minimum for :math:`f`
where :math:`9-x_i^2\ge0` for :math:`i=1,\dots,9`::

	from sympy import *
	from Irene import SDPRelaxations
	NumVars = 9
	# define the symbolic variables and functions
	x = [Symbol('x%d' % i) for i in range(NumVars)]

	print "Relaxation method:"
	# initiate the SDPRelaxations object
	Rlx = SDPRelaxations(x)
	# Rosenbrock function
	rosen = sum([100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])
	             ** 2 for i in range(NumVars - 1)])
	# set the objective
	Rlx.SetObjective(rosen)
	# add constraints
	for i in range(NumVars):
	    Rlx.AddConstraint(9 - x[i]**2 >= 0)
	# set the sdp solver
	Rlx.SetSDPSolver('cvxopt')
	# initiate the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution
	# solve with scipy
	from scipy.optimize import minimize
	fun = lambda x: sum([100 * (x[i + 1] - x[i]**2)**2 +
	                     (1 - x[i])**2 for i in range(NumVars - 1)])
	cons = [
	    {'type': 'ineq', 'fun': lambda x: 9 - x[i]**2} for i in range(NumVars)]
	x0 = tuple([0 for _ in range(NumVars)])
	sol1 = minimize(fun, x0, method='COBYLA', constraints=cons)
	sol2 = minimize(fun, x0, method='SLSQP', constraints=cons)

	print "solution according to 'COBYLA':"
	print sol1
	print "solution according to 'SLSQP':"
	print sol2

	# pyOpt
	from pyOpt import *


	def objfunc(x):
	    f = sum([100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])
	             ** 2 for i in range(NumVars - 1)])
	    g = [x[i]**2 - 9 for i in range(NumVars)]
	    fail = 0
	    return f, g, fail

	opt_prob = Optimization(
	    'The Rosenbrock function', objfunc)
	opt_prob.addObj('f')
	for i in range(NumVars):
	    opt_prob.addVar('x%d' % (i + 1), 'c', lower=-3, upper=3, value=0.0)
	    opt_prob.addCon('g%d' % (i + 1), 'i')

	# Augmented Lagrangian Particle Swarm Optimizer
	alpso = ALPSO()
	alpso(opt_prob)
	print opt_prob.solution(0)
	# Non Sorting Genetic Algorithm II
	nsg2 = NSGA2()
	nsg2(opt_prob)
	print opt_prob.solution(1)

The result is::

	Relaxation method:
	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 750.234924078 seconds
	              Run Time: 8.43369 seconds
	Primal Objective Value: 1.67774267808e-08
	  Dual Objective Value: 1.10015692778e-08
	Feasible solution for moments of order 2

	solution according to 'COBYLA':
	     fun: 4.4963584556077389
	   maxcv: 0.0
	 message: 'Maximum number of function evaluations has been exceeded.'
	    nfev: 1000
	  status: 2
	 success: False
	       x: array([  8.64355944e-01,   7.47420978e-01,   5.59389194e-01,
	         3.16212252e-01,   1.05034350e-01,   2.05923923e-02,
	         9.44389237e-03,   1.12341021e-02,  -7.74530516e-05])
	     fun: 1.3578865444308464e-07
	     jac: array([ 0.00188377,  0.00581741, -0.00182463,  0.00776938, -0.00343305,
	       -0.00186283,  0.0020364 ,  0.00881489, -0.0047164 ,  0.        ])
	solution according to 'SLSQP':
	 message: 'Optimization terminated successfully.'
	    nfev: 625
	     nit: 54
	    njev: 54
	  status: 0
	 success: True
	       x: array([ 1.00000841,  1.00001216,  1.00000753,  1.00001129,  1.00000134,
	        1.00000067,  1.00000502,  1.00000682,  0.99999006])

	ALPSO Solution to The Rosenbrock function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                   10.6371
	    Total Function Evaluations:     48040
	    Lambda: [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
	    Seed: 1482114864.60097694

	    Objectives:
	        Name        Value        Optimum
		     f        0.590722             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      0.992774      -3.00e+00     3.00e+00 
		     x2       c	      0.986019      -3.00e+00     3.00e+00 
		     x3       c	      0.970756      -3.00e+00     3.00e+00 
		     x4       c	      0.942489      -3.00e+00     3.00e+00 
		     x5       c	      0.886910      -3.00e+00     3.00e+00 
		     x6       c	      0.787367      -3.00e+00     3.00e+00 
		     x7       c	      0.618875      -3.00e+00     3.00e+00 
		     x8       c	      0.382054      -3.00e+00     3.00e+00 
		     x9       c	      0.143717      -3.00e+00     3.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -8.014399 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -8.027767 <= 0.00e+00
		     g3   	  i       -1.00e+21 <= -8.057633 <= 0.00e+00
		     g4   	  i       -1.00e+21 <= -8.111714 <= 0.00e+00
		     g5   	  i       -1.00e+21 <= -8.213391 <= 0.00e+00
		     g6   	  i       -1.00e+21 <= -8.380053 <= 0.00e+00
		     g7   	  i       -1.00e+21 <= -8.616994 <= 0.00e+00
		     g8   	  i       -1.00e+21 <= -8.854035 <= 0.00e+00
		     g9   	  i       -1.00e+21 <= -8.979345 <= 0.00e+00

	--------------------------------------------------------------------------------


	NSGA-II Solution to The Rosenbrock function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.6244
	    Total Function Evaluations:          

	    Objectives:
	        Name        Value        Optimum
		     f          5.5654             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      0.727524      -3.00e+00     3.00e+00 
		     x2       c	      0.537067      -3.00e+00     3.00e+00 
		     x3       c	      0.296186      -3.00e+00     3.00e+00 
		     x4       c	      0.094420      -3.00e+00     3.00e+00 
		     x5       c	      0.017348      -3.00e+00     3.00e+00 
		     x6       c	      0.009658      -3.00e+00     3.00e+00 
		     x7       c	      0.015372      -3.00e+00     3.00e+00 
		     x8       c	      0.009712      -3.00e+00     3.00e+00 
		     x9       c	      0.001387      -3.00e+00     3.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -8.470708 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -8.711559 <= 0.00e+00
		     g3   	  i       -1.00e+21 <= -8.912274 <= 0.00e+00
		     g4   	  i       -1.00e+21 <= -8.991085 <= 0.00e+00
		     g5   	  i       -1.00e+21 <= -8.999699 <= 0.00e+00
		     g6   	  i       -1.00e+21 <= -8.999907 <= 0.00e+00
		     g7   	  i       -1.00e+21 <= -8.999764 <= 0.00e+00
		     g8   	  i       -1.00e+21 <= -8.999906 <= 0.00e+00
		     g9   	  i       -1.00e+21 <= -8.999998 <= 0.00e+00

	--------------------------------------------------------------------------------

The relaxation method returns values very close to the actual minimum but 
two out of other three methods fail to estimate the minimum correctly.

Giunta Function
==================================

Giunta is an example of continuous, differentiable, separable, scalable, 
multimodal function defined by:

.. math::
	\begin{array}{lcl}
	f(x_1, x_2) & = & \frac{3}{5} + \sum_{i=1}^2[\sin(\frac{16}{15}x_i-1)\\
		& + & \sin^2(\frac{16}{15}x_i-1)\\
		& + & \frac{1}{50}\sin(4(\frac{16}{15}x_i-1))].
	\end{array}


The following code optimizes :math:`f` when :math:`1-x^2\ge0` and :math:`1-y^2\ge0`::

	from sympy import *
	from Irene import *
	x = Symbol('x')
	y = Symbol('y')
	s1 = Symbol('s1')
	c1 = Symbol('c1')
	s2 = Symbol('s2')
	c2 = Symbol('c2')
	f = .6 + (sin(x - 1) + (sin(x - 1))**2 + .02 * sin(4 * (x - 1))) + \
	    (sin(y - 1) + (sin(y - 1))**2 + .02 * sin(4 * (y - 1)))
	f = expand(f, trig=True)
	f = N(f.subs({sin(x): s1, cos(x): c1, sin(y): s2, cos(y): c2}))
	rels = [s1**2 + c1**2 - 1, s2**2 + c2**2 - 1]
	Rlx = SDPRelaxations([s1, c1, s2, c2], rels)
	Rlx.SetObjective(f)
	Rlx.AddConstraint(1 - s1**2 >= 0)
	Rlx.AddConstraint(1 - s2**2 >= 0)
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution
	# solve with scipy
	from scipy.optimize import minimize
	fun = lambda x: .6 + (sin((16. / 15.) * x[0] - 1) + (sin((16. / 15.) * x[0] - 1))**2 + .02 * sin(4 * ((16. / 15.) * x[0] - 1))) + (
	    sin((16. / 15.) * x[1] - 1) + (sin((16. / 15.) * x[1] - 1))**2 + .02 * sin(4 * ((16. / 15.) * x[1] - 1)))
	cons = [
	    {'type': 'ineq', 'fun': lambda x: 1 - x[i]**2} for i in range(2)]
	x0 = tuple([0 for _ in range(2)])
	sol1 = minimize(fun, x0, method='COBYLA', constraints=cons)
	sol2 = minimize(fun, x0, method='SLSQP', constraints=cons)
	print "solution according to 'COBYLA':"
	print sol1
	print "solution according to 'SLSQP':"
	print sol2

	# pyOpt
	from pyOpt import *


	def objfunc(x):
	    f = .6 + (sin((16. / 15.) * x[0] - 1) + (sin((16. / 15.) * x[0] - 1))**2 + .02 * sin(4 * ((16. / 15.) * x[0] - 1))) + (
	        sin((16. / 15.) * x[1] - 1) + (sin((16. / 15.) * x[1] - 1))**2 + .02 * sin(4 * ((16. / 15.) * x[1] - 1)))
	    g = [x[i]**2 - 1 for i in range(2)]
	    fail = 0
	    return f, g, fail

	opt_prob = Optimization(
	    'The Giunta function', objfunc)
	opt_prob.addObj('f')
	for i in range(2):
	    opt_prob.addVar('x%d' % (i + 1), 'c', lower=-1, upper=1, value=0.0)
	    opt_prob.addCon('g%d' % (i + 1), 'i')

	# Augmented Lagrangian Particle Swarm Optimizer
	alpso = ALPSO()
	alpso(opt_prob)
	print opt_prob.solution(0)
	# Non Sorting Genetic Algorithm II
	nsg2 = NSGA2()
	nsg2(opt_prob)
	print opt_prob.solution(1)

and the result is::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 2.53814482689 seconds
	              Run Time: 0.041321 seconds
	Primal Objective Value: 0.0644704534329
	  Dual Objective Value: 0.0644704595475
	Feasible solution for moments of order 2

	solution according to 'COBYLA':
	     fun: 0.064470430891900576
	   maxcv: 0.0
	 message: 'Optimization terminated successfully.'
	    nfev: 40
	  status: 1
	 success: True
	       x: array([ 0.46730658,  0.4674184 ])
	solution according to 'SLSQP':
	     fun: 0.0644704633430450
	     jac: array([-0.00029983, -0.00029983,  0.        ])
	 message: 'Optimization terminated successfully.'
	    nfev: 13
	     nit: 3
	    njev: 3
	  status: 0
	 success: True
	       x: array([ 0.46717727,  0.46717727])

	ALPSO Solution to The Giunta function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                   10.6180
	    Total Function Evaluations:      1240
	    Lambda: [ 0.  0.]
	    Seed: 1482115204.08583212

	    Objectives:
	        Name        Value        Optimum
		     f       0.0644704             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      0.467346      -1.00e+00     1.00e+00 
		     x2       c	      0.467369      -1.00e+00     1.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -0.781588 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -0.781566 <= 0.00e+00

	--------------------------------------------------------------------------------


	NSGA-II Solution to The Giunta function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                   50.9196
	    Total Function Evaluations:          

	    Objectives:
	        Name        Value        Optimum
		     f       0.0644704             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      0.467403      -1.00e+00     1.00e+00 
		     x2       c	      0.467324      -1.00e+00     1.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -0.781535 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -0.781608 <= 0.00e+00

	--------------------------------------------------------------------------------


Parsopoulos Function
==================================

Parsopoulos is defined as :math:`f(x,y)=\cos^2(x)+\sin^2(y)`.
The following code computes its minimum where :math:`-5\leq x,y\leq5`::

	from sympy import *
	from Irene import *
	x = Symbol('x')
	y = Symbol('y')
	s1 = Symbol('s1')
	c1 = Symbol('c1')
	s2 = Symbol('s2')
	c2 = Symbol('c2')
	f = c1**2 + s2**2
	rels = [s1**2 + c1**2 - 1, s2**2 + c2**2 - 1]
	Rlx = SDPRelaxations([s1, c1, s2, c2], rels)
	Rlx.SetObjective(f)
	Rlx.AddConstraint(1 - s1**2 >= 0)
	Rlx.AddConstraint(1 - s2**2 >= 0)
	Rlx.MomentsOrd(2)
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution
	# solve with scipy
	from scipy.optimize import minimize
	fun = lambda x: cos(x[0])**2 + sin(x[1])**2
	cons = [
	    {'type': 'ineq', 'fun': lambda x: 25 - x[i]**2} for i in range(2)]
	x0 = tuple([0 for _ in range(2)])
	sol1 = minimize(fun, x0, method='COBYLA', constraints=cons)
	sol2 = minimize(fun, x0, method='SLSQP', constraints=cons)
	print "solution according to 'COBYLA':"
	print sol1
	print "solution according to 'SLSQP':"
	print sol2

	# pyOpt
	from pyOpt import *


	def objfunc(x):
	    f = cos(x[0])**2 + sin(x[1])**2
	    g = [x[i]**2 - 25 for i in range(2)]
	    fail = 0
	    return f, g, fail

	opt_prob = Optimization(
	    'The Parsopoulos function', objfunc)
	opt_prob.addObj('f')
	for i in range(2):
	    opt_prob.addVar('x%d' % (i + 1), 'c', lower=-5, upper=5, value=0.0)
	    opt_prob.addCon('g%d' % (i + 1), 'i')

	# Augmented Lagrangian Particle Swarm Optimizer
	alpso = ALPSO()
	alpso(opt_prob)
	print opt_prob.solution(0)
	# Non Sorting Genetic Algorithm II
	nsg2 = NSGA2()
	nsg2(opt_prob)
	print opt_prob.solution(1)

which returns::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 2.48692297935 seconds
	              Run Time: 0.035358 seconds
	Primal Objective Value: -3.74719295193e-10
	  Dual Objective Value: 5.43053240402e-12
	Feasible solution for moments of order 2

	solution according to 'COBYLA':
	     fun: 1.83716742579312e-08
	   maxcv: 0.0
	 message: 'Optimization terminated successfully.'
	    nfev: 35
	  status: 1
	 success: True
	       x: array([  1.57072551e+00,   1.15569800e-04])
	solution according to 'SLSQP':
	     fun: 1
	     jac: array([ -1.49011612e-08,   1.49011612e-08,   0.00000000e+00])
	 message: 'Optimization terminated successfully.'
	    nfev: 4
	     nit: 1
	    njev: 1
	  status: 0
	 success: True
	       x: array([ 0.,  0.])

	ALPSO Solution to The Parsopoulos function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    4.4576
	    Total Function Evaluations:      1240
	    Lambda: [ 0.  0.]
	    Seed: 1482115438.17070389

	    Objectives:
	        Name        Value        Optimum
		     f     5.68622e-09             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	     -4.712408      -5.00e+00     5.00e+00 
		     x2       c	     -0.000073      -5.00e+00     5.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -2.793212 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -25.000000 <= 0.00e+00

	--------------------------------------------------------------------------------


	NSGA-II Solution to The Parsopoulos function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                   17.7197
	    Total Function Evaluations:          

	    Objectives:
	        Name        Value        Optimum
		     f     2.37167e-08             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	     -1.570676      -5.00e+00     5.00e+00 
		     x2       c	      3.141496      -5.00e+00     5.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -22.532977 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -15.131000 <= 0.00e+00

	--------------------------------------------------------------------------------


Shubert Function
==================================

Shubert function is defined by:

.. math::
	f(x_1,\dots,x_n) = \prod_{i=1}^n\left(\sum_{j=1}^5\cos((j+1)x_i+i)\right).

It is a continuous, differentiable, separable, non-scalable, multimodal function.
The following code compares the result of five optimizers when :math:`-10\leq x_i\leq10`
and :math:`n=2`::

	from sympy import *
	from Irene import *
	x = Symbol('x')
	y = Symbol('y')
	s1 = Symbol('s1')
	c1 = Symbol('c1')
	s2 = Symbol('s2')
	c2 = Symbol('c2')
	f = sum([cos((j + 1) * x + j) for j in range(1, 6)]) * \
	    sum([cos((j + 1) * y + j) for j in range(1, 6)])
	obj = N(expand(f, trig=True).subs(
	    {sin(x): s1, cos(x): c1, sin(y): s2, cos(y): c2}))
	rels = [s1**2 + c1**2 - 1, s2**2 + c2**2 - 1]
	Rlx = SDPRelaxations([s1, c1, s2, c2], rels)
	Rlx.SetObjective(obj)
	Rlx.AddConstraint(1 - s1**2 >= 0)
	Rlx.AddConstraint(1 - s2**2 >= 0)
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution
	g = lambda x: sum([cos((j + 1) * x[0] + j) for j in range(1, 6)]) * \
	    sum([cos((j + 1) * x[1] + j) for j in range(1, 6)])
	x0 = (-5, 5)
	from scipy.optimize import minimize
	cons = (
	    {'type': 'ineq', 'fun': lambda x: 100 - x[0]**2},
	    {'type': 'ineq', 'fun': lambda x: 100 - x[1]**2})
	sol1 = minimize(g, x0, method='COBYLA', constraints=cons)
	sol2 = minimize(g, x0, method='SLSQP', constraints=cons)
	print "solution according to 'COBYLA':"
	print sol1
	print "solution according to 'SLSQP':"
	print sol2

	from sage.all import *
	m1 = minimize_constrained(g, cons=[cn['fun'] for cn in cons], x0=x0)
	m2 = minimize_constrained(g, cons=[cn['fun']
	                                   for cn in cons], x0=x0, algorithm='l-bfgs-b')
	print "Sage:"
	print "minimize_constrained (default):", m1, g(m1)
	print "minimize_constrained (l-bfgs-b):", m2, g(m2)

	# pyOpt
	from pyOpt import *


	def objfunc(x):
	    f = sum([cos((j + 1) * x[0] + j) for j in range(1, 6)]) * \
	        sum([cos((j + 1) * x[1] + j) for j in range(1, 6)])
	    g = [x[i]**2 - 100 for i in range(2)]
	    fail = 0
	    return f, g, fail

	opt_prob = Optimization(
	    'The Shubert function', objfunc)
	opt_prob.addObj('f')
	for i in range(2):
	    opt_prob.addVar('x%d' % (i + 1), 'c', lower=-10, upper=10, value=0.0)
	    opt_prob.addCon('g%d' % (i + 1), 'i')

	# Augmented Lagrangian Particle Swarm Optimizer
	alpso = ALPSO()
	alpso(opt_prob)
	print opt_prob.solution(0)
	# Non Sorting Genetic Algorithm II
	nsg2 = NSGA2()
	nsg2(opt_prob)
	print opt_prob.solution(1)

The result is::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 730.02412415 seconds
	              Run Time: 5.258507 seconds
	Primal Objective Value: -18.0955649723
	  Dual Objective Value: -18.0955648855
	Feasible solution for moments of order 6
	Scipy 'COBYLA':
	     fun: -3.3261182321238367
	   maxcv: 0.0
	 message: 'Optimization terminated successfully.'
	    nfev: 39
	  status: 1
	 success: True
	       x: array([-3.96201407,  4.81176624])
	Scipy 'SLSQP':
	     fun: -0.856702387212005
	     jac: array([-0.00159422,  0.00080796,  0.        ])
	 message: 'Optimization terminated successfully.'
	    nfev: 35
	     nit: 7
	    njev: 7
	  status: 0
	 success: True
	       x: array([-4.92714381,  4.81186391])
	Sage:
	minimize_constrained (default): (-3.962032420336303, 4.811734682897321) -3.32611819422
	minimize_constrained (l-bfgs-b): (-3.962032420336303, 4.811734682897321) -3.32611819422

	ALPSO Solution to The Shubert function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                   37.7526
	    Total Function Evaluations:      2200
	    Lambda: [ 0.  0.]
	    Seed: 1482115770.57303905

	    Objectives:
	        Name        Value        Optimum
		     f        -18.0956             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	     -7.061398      -1.00e+01     1.00e+01 
		     x2       c	     -1.471424      -1.00e+01     1.00e+01 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -50.136654 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -97.834910 <= 0.00e+00

	--------------------------------------------------------------------------------


	NSGA-II Solution to The Shubert function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                   97.6291
	    Total Function Evaluations:          

	    Objectives:
	        Name        Value        Optimum
		     f        -18.0955             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	     -0.778010      -1.00e+01     1.00e+01 
		     x2       c	     -7.754277      -1.00e+01     1.00e+01 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -99.394700 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -39.871193 <= 0.00e+00

	--------------------------------------------------------------------------------


We note that four out of six other optimizers stuck at a local minimum and 
return incorrect values.

Moreover, we employed 20 different optimizers included in `pyOpt <http://www.pyopt.org/>`_
and only 4 of them returned the correct optimum value.