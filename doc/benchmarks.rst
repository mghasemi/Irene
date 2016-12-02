==================================
Benchmark Optimization Problems
==================================

There are benchmark problems to evaluated how good an optimization method works.
We apply the generalized relaxation method to some of these benchmarks that are
mainly taken from [MJXY]_.

.. [MJXY] M. Jamil, Xin-She Yang, *A literature survey of benchmark functions for global optimization problems*, IJMMNO, Vol. 4(2), 2013.

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
	print  "solution according to 'COBYLA':"
	sol1 = minimize(fun, x0, method='COBYLA', constraints=cons)
	print "solution according to 'SLSQP':"
	sol2 = minimize(fun, x0, method='SLSQP', constraints=cons)
	print sol1
	print sol2
	# solve with pso
	print "PSO:"
	from pyswarm import pso
	lb = [-3 for i in range(NumVars)]
	ub = [3 for i in range(NumVars)]
	cns = [cn['fun'] for cn in cons]
	print pso(fun, lb, ub, ieqcons=cns)

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
	PSO:
	Stopping search: maximum iterations reached --> 100
	(array([-0.30495496,  0.10698904, -0.129344  ,  0.07972014,  0.027356  ,
	        0.02170117, -0.0036854 ,  0.10778454,  0.04141022]), 12.067752160169965)

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
	print  "solution according to 'COBYLA':"
	print sol1
	print "solution according to 'SLSQP':"
	print sol2
	# solve with pso
	print "PSO:"
	from pyswarm import pso
	lb = [-1 for i in range(2)]
	ub = [1 for i in range(2)]
	cns = [cn['fun'] for cn in cons]
	print pso(fun, lb, ub, ieqcons=cns)

and the result is::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 3.03993797302 seconds
	              Run Time: 0.015762 seconds
	Primal Objective Value: 0.0644704534329
	  Dual Objective Value: 0.0644704595475
	Feasible solution for moments of order 2

	solution according to 'COBYLA':
	     fun: 0.064470433766030344
	   maxcv: 0.0
	 message: 'Optimization terminated successfully.'
	    nfev: 45
	  status: 1
	 success: True
	       x: array([ 0.49835509,  0.49847982])
	solution according to 'SLSQP':
	     fun: 0.0644705317528075
	     jac: array([ 0.00045323,  0.00045323,  0.        ])
	 message: 'Optimization terminated successfully.'
	    nfev: 17
	     nit: 4
	    njev: 4
	  status: 0
	 success: True
	       x: array([ 0.4987201,  0.4987201])
	PSO:
	Stopping search: Swarm best objective change less than 1e-08
	(array([ 0.49858097,  0.49835221]), 0.064470444814555605)

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
	# solve with pso
	print "PSO:"
	from pyswarm import pso
	lb = [-5 for i in range(2)]
	ub = [5 for i in range(2)]
	cns = [cn['fun'] for cn in cons]
	print pso(fun, lb, ub, ieqcons=cns)

which returns::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 3.01474308968 seconds
	              Run Time: 0.013299 seconds
	Primal Objective Value: -3.7471929546e-10
	  Dual Objective Value: 5.43046022792e-12
	Feasible solution for moments of order 2

	solution according to 'COBYLA':
	     fun: 1.8371674257900859e-08
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
	PSO:
	Stopping search: Swarm best objective change less than 1e-08
	(array([ 4.71233715,  3.14155673]), 3.9770280273184657e-09)

Shubert Function
==================================

Subert function is defined by:

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
	print "PSO:"
	from pyswarm import pso
	lb = [-10, -10]
	ub = [10, 10]
	cns = [cn['fun'] for cn in cons]
	print pso(g, lb, ub, ieqcons=cns)

The result is::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 1129.02412415 seconds
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
	PSO:
	Stopping search: Swarm best objective change less than 1e-08
	(array([-0.77822054,  4.8118272 ]), -18.095565036766224)

We note that four out of five optimizers stuck at a local minimum and 
return incorrect values.