=============================
Optimization
=============================

Let :math:`X` be a nonempty topological space and :math:`A` be a unital sub-algebra of continuous functions over :math:`X`
which separates points of :math:`X`. We consider the following optimization problem:

.. math::
	\left\lbrace
	\begin{array}{lll}
		\min & f(x) & \\
		\textrm{subject to} & & \\
		& g_i(x)\ge 0 & i=1,\dots,m.
	\end{array}
	\right.

Denote the feasibility set of the above program by :math:`K` (i.e., :math:`K=\{x\in X:g_i(x)\ge 0,~ i=1,\dots,m\}`).
Let :math:`\rho` be the optimum value of the above program and :math:`\mathcal{M}_1^+(K)` be the space of all probability Borel 
measures supported on :math:`K`. One can show that:

.. math::
	\rho = \inf_{\mu\in\mathcal{M}_1^+(K)}\int f~d\mu.

This associates a :math:`K`-positive linear functional :math:`L_{\mu}` to every measure :math:`\mu\in\mathcal{M}_1^+(K)`. 
Let us denote the set of all elements of :math:`A` nonnegative on :math:`K` by :math:`Psd_A(K)`.
If :math:`\exists p\in Psd_A(K)` such that :math:`p^{-1}([0, n])` is compact for each :math:`n\in\mathbb{N}`, then one can show that
every :math:`K`-positive linear functional admits an integral representation via a Borel measure on :math:`K` (Marshall's generalization
of Haviland's theorem).
Let :math:`Q_{\bf g}` be the quadratic module generated by :math:`g_1,\dots,g_m`, i.e, the set of all elements in :math:`A` of the form 

.. math::
	\sigma_0+\sigma_1 g_1+\dots+\sigma_m g_m,
	:label: sosdecomp

where :math:`\sigma_0,\dots,\sigma_m\in\sum A^2` are sums of squares of elements of :math:`A`. A quadratic module :math:`Q` is said to be 
Archimedean if for every :math:`h\in A` there exists :math:`M>0` such that :math:`M\pm h\in Q`. By Jacobi's representation theorem, 
if :math:`Q` is Archimedean and :math:`h>0` on :math:`K`, where :math:`K=\{x\in X:g(x)\ge0~\forall g\in Q\}`, then :math:`h\in Q`.
Since :math:`Q` is Archimedean, :math:`K` is compact and this implies that if a linear functional on :math:`A` is nonnegative on :math:`Q`, 
then it is :math:`K`-positive and hence admits an integral representation. Therefore:

.. math::
	\rho = \inf_{\tiny\begin{array}{c}L(Q)\ge 0\\ L(1)=1\end{array}}L(f).

Let :math:`Q=Q_{\bf g}` and :math:`L(Q)\subseteq[0,\infty)`. Then clearly :math:`L(\sum A^2)\subseteq[0,\infty)` which means :math:`L` is
positive semidefinite. Moreover, for each :math:`i=1,\dots,m`, :math:`L(g_i\sum A^2)\subseteq[0,\infty)` which means the maps

.. math::
	\begin{array}{rcl}
		L_{g_i}:A & \longrightarrow & \mathbb{R}\\
		h & \mapsto & L(g_i h)
	\end{array}

are positive semidefinite. So the optimum value of the following program is still equal to :math:`\rho`:

.. math::
	\left\lbrace
	\begin{array}{lll}
		\min & L(f) & \\
		\textrm{subject to} & & \\
		& L\succeq 0 & \\
		& L_{g_i}\succeq0 & i=1,\dots,m.
	\end{array}
	\right.
	:label: infsdp

This, still is not a semidefinite program as each constraint is infinite dimensional. One plausible idea is to consider functionals on
finite dimensional subspaces of :math:`A` containing :math:`f, g_1,\dots,g_m`. This was done by Lasserre for polynomials [JBL]_.

Let :math:`B\subseteq A` be a linear subspace. If :math:`L:A\longrightarrow\mathbb{R}` is :math:`K`-positive, so is its restriction 
on :math:`B`. But generally, :math:`K`-positive maps on :math:`B` do not extend to :math:`K`-positive one on :math:`A` and hence
existence of integral representations are not guaranteed. Under a rather mild condition, this issue can be resolved:

**Theorem.** [GIKM]_ Let :math:`K\subseteq X` be compact, :math:`B\subseteq A` a linear subspace such that there exists :math:`p\in B` strictly 
positive on :math:`K`. Then every linear functional :math:`L:B\longrightarrow\mathbb{R}` satisfying :math:`L(Psd_B(K))\subseteq[0,\infty)` 
admits an integral representation via a Borel measure supported on :math:`K`.

Now taking :math:`B` to be a finite dimensional linear space containing :math:`f, g_1,\dots,g_m` and satisfying the assumptions of the 
above theorem,  turns :eq:`infsdp` into a semidefinite program. Note that this does not imply that the optimum value of the resulting 
SDP is equal to :math:`\rho` since

	+ :math:`Q_{\bf g}\cap B\neq Psd_{B}(K)` and,
	+ there may not exist a decomposition of :math:`f-\rho` as in :eq:`sosdecomp` inside :math:`B` (i.e., the summands may not belong to :math:`B`).

Thus, the optimum value just gives a lower bound for :math:`\rho`. But walking through a :math:`K`-frame, as explained in [GIKM]_ constructs 
a net of lower bounds for :math:`\rho` which approaches :math:`\rho`, eventually.

I practice, one only needs to find a sufficiently big finite dimensional linear space which contains :math:`f, g_1,\dots,g_m` and a :eq:`sosdecomp`
decomposition of :math:`f-\rho` can be found within that space. Therefore, the convergence happens in finitely many steps, subject to finding a 
suitable :math:`K`-frame for the problem.  

The significance of this method is that it converts any optimization problem into finitely many semidefinite programs whose optimum values approaches 
the optimum value of the original program and semidefinite programs can be solved in polynomial time. Although, this suggests that the NP-complete 
problem of optimization can be solved in P-time, but since the number of SDPs that is required to reach the optimum is unknown and such a bound does
not exists when dealing with Archimedean modules.

.. note::
	

	1. One behavior that distinguishes this method from others is that using SDP relaxations always provides a lower bound for the\ 
	minimum value of the objective function over the feasibility set. While other methods usually involve evaluation of the objective\ 
	and hence the result is always an upper bound for the minimum.
	
	2. The SDP relaxation method relies on symbolic computations which could be quite costly and slow. Therefore, dealing with rather large
	problems -although `Irene` takes advantage from multiple cores- can be rather slow.

.. [GIKM] M\. Ghasemi, M. Infusino, S. Kuhlmann and M. Marshall, *Truncated Moment Problem for unital commutative real algebras*, to appear.
.. [JBL] J-B. Lasserre, *Global optimization with polynomials and the problem of moments*, SIAM J. Optim. 11(3) 796-817 (2000).

Polynomial Optimization
=============================

The SDP relaxation method was originally introduced by Lasserre [JBL]_ for polynomial optimization problem and excellent software packages such
as `GloptiPoly <http://homepages.laas.fr/henrion/software/gloptipoly/>`_ and `ncpol2sdpa <https://github.com/peterwittek/ncpol2sdpa>`_
exist to handle constraint polynomial optimization problems. 

`Irene` uses `sympy <http://www.sympy.org/>`_ for symbolic computations, so, it always need to be imported and the symbolic variables must be
introduced. Once these steps are done, the objective and constraints should be entered using ``SetObjective`` and `AddConstraint` methods.
the method ``MomentsOrd`` takes the relaxation degree upon user's request, otherwise the minimum relaxation degree will be used.
The default SDP solver is ``CVXOPT`` which can be modified via ``SetSDPSolver`` method. Currently ``CVXOPT``, ``DSDP``, ``SDPA`` and ``CSDP`` are supported.
Next step is initialization of the SDP by ``InitSDP`` and finally solving the SDP via ``Minimize`` and the output will be stored in the ``Solution``
variable as a python dictionary.

**Example** Solve the following polynomial optimization problem:

.. math::
	\left\lbrace
	\begin{array}{ll}
		\min & -2x+y-z\\
		\textrm{subject to} & 24-20x+9y-13z+4x^2-4xy \\
		& +4xz+2y^2-2yz+2z^2\ge0\\
		& x+y+z\leq 4\\
		& 3y+z\leq 6\\
		& 0\leq x\leq 2\\
		& y\ge 0\\
		& 0\leq z\leq 3.
	\end{array}\right.

The following program uses relaxation of degree 3 and `sdpa` to solve the above problem::

	from sympy import *
	from Irene import *
	# introduce variables
	x = Symbol('x')
	y = Symbol('y')
	z = Symbol('z')
	# initiate the Relaxation object
	Rlx = SDPRelaxations([x, y, z])
	# set the objective
	Rlx.SetObjective(-2 * x + y - z)
	# add support constraints
	Rlx.AddConstraint(24 - 20 * x + 9 * y - 13 * z + 4 * x**2 -
	                  4 * x * y + 4 * x * z + 2 * y**2 - 2 * y * z + 2 * z**2 >= 0)
	Rlx.AddConstraint(x + y + z <= 4)
	Rlx.AddConstraint(3 * y + z <= 6)
	Rlx.AddConstraint(x >= 0)
	Rlx.AddConstraint(x <= 2)
	Rlx.AddConstraint(y >= 0)
	Rlx.AddConstraint(z >= 0)
	Rlx.AddConstraint(z <= 3)
	# set the relaxation order
	Rlx.MomentsOrd(3)
	# set the solver
	Rlx.SetSDPSolver('dsdp')
	# initialize the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	# output
	print Rlx.Solution

The output looks like::
	
	Solution of a Semidefinite Program:
	                Solver: DSDP
	                Status: Optimal
	   Initialization Time: 8.04711222649 seconds
	              Run Time: 1.056733 seconds
	Primal Objective Value: -4.06848294478
	  Dual Objective Value: -4.06848289445
	Feasible solution for moments of order 3

Moment Constraints
-----------------------------
Initially the only constraints forced on the moments are those  in :eq:`infsdp`. We can also force user defined constraints on the moments
by calling ``MomentConstraint`` on a ``Mom`` object. The following adds two constraints :math:`\int xy~d\mu\ge\frac{1}{2}` and 
:math:`\int yz~d\mu + \int z~d\mu\ge 1` to the previous example::

	from sympy import *
	from Irene import *
	# introduce variables
	x = Symbol('x')
	y = Symbol('y')
	z = Symbol('z')
	# initiate the Relaxation object
	Rlx = SDPRelaxations([x, y, z])
	# set the objective
	Rlx.SetObjective(-2 * x + y - z)
	# add support constraints
	Rlx.AddConstraint(24 - 20 * x + 9 * y - 13 * z + 4 * x**2 -
	                  4 * x * y + 4 * x * z + 2 * y**2 - 2 * y * z + 2 * z**2 >= 0)
	Rlx.AddConstraint(x + y + z <= 4)
	Rlx.AddConstraint(3 * y + z <= 6)
	Rlx.AddConstraint(x >= 0)
	Rlx.AddConstraint(x <= 2)
	Rlx.AddConstraint(y >= 0)
	Rlx.AddConstraint(z >= 0)
	Rlx.AddConstraint(z <= 3)
	# add moment constraints
	Rlx.MomentConstraint(Mom(x * y) >= .5)
	Rlx.MomentConstraint(Mom(y * z) + Mom(z) >= 1)
	# set the relaxation order
	Rlx.MomentsOrd(3)
	# set the solver
	Rlx.SetSDPSolver('dsdp')
	# initialize the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	# output
	print Rlx.Solution
	print "Moment of x*y:", Rlx.Solution[x * y]
	print "Moment of y*z + z:", Rlx.Solution[y * z] + Rlx.Solution[z]

Solution is::

	Solution of a Semidefinite Program:
	                Solver: DSDP
	                Status: Optimal
	   Initialization Time: 7.91646790504 seconds
	              Run Time: 1.041935 seconds
	Primal Objective Value: -4.03644346623
	  Dual Objective Value: -4.03644340796
	Feasible solution for moments of order 3

	Moment of x*y: 0.500000001712
	Moment of y*z + z: 2.72623169152

Equality Constraints
-----------------------------
Although it is possible to add equality constraints via ``AddConstraint`` and ``MomentConstraint``, but 
`SDPRelaxation` converts them to two inequalities and considers a certain margin of error. 
For :math:`A=B`, it considers :math:`A\ge B - \varepsilon` and :math:`A\leq B + \varepsilon`.
In this case the value of :math:`\varepsilon` can be modified by setting `SDPRelaxation.ErrorTolerance`
which its default value is :math:`10^{-6}`.

Truncated Moment Problem
==================================
It must be clear that we can use ``SDPRelaxations.MomentConstraint`` to introduce a typical truncated
moment problem over polynomials as described in [JNie]_.

**Example** Find the support of a measure :math:`\mu` whose support is a subset of :math:`[-1,1]^2` and the followings hold:

.. math::
	\begin{array}{cc}
		\int x^2d\mu=\int y^2d\mu=\frac{1}{3} & \int x^2yd\mu=\int xy^2d\mu=0\\
		\int x^2y^2d\mu=\frac{1}{9} & \int x^4y^2d\mu=\int x^2y^4d\mu=\frac{1}{15}.
	\end{array}

The following code does the job::

	from sympy import *
	from Irene import *
	# introduce variables
	x = Symbol('x')
	y = Symbol('y')
	# initiate the Relaxation object
	Rlx = SDPRelaxations([x, y])
	# add support constraints
	Rlx.AddConstraint(1 - x**2 >= 0)
	Rlx.AddConstraint(1 - y**2 >= 0)
	# add moment constraints
	Rlx.MomentConstraint(Mom(x**2) == 1. / 3.)
	Rlx.MomentConstraint(Mom(y**2) == 1. / 3.)
	Rlx.MomentConstraint(Mom(x**2 * y) == 0.)
	Rlx.MomentConstraint(Mom(x * y**2) == 0.)
	Rlx.MomentConstraint(Mom(x**2 * y**2) == 1. / 9.)
	Rlx.MomentConstraint(Mom(x**4 * y**2) == 1. / 15.)
	Rlx.MomentConstraint(Mom(x**2 * y**4) == 1. / 15.)
	# set the solver
	Rlx.SetSDPSolver('dsdp')
	# initialize the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	# output
	Rlx.Solution.ExtractSolution('lh', 2)
	print Rlx.Solution

and the result is::

	Solution of a Semidefinite Program:
	                Solver: DSDP
	                Status: Optimal
	   Initialization Time: 1.08686900139 seconds
	              Run Time: 0.122459 seconds
	Primal Objective Value: 0.0
	  Dual Objective Value: -9.36054051771e-09
	               Support:
			(0.40181215311129925, 0.54947643681480196)
			(-0.40181215311127805, -0.54947643681498193)
	        Support solver: Lasserre--Henrion
	Feasible solution for moments of order 3

Note that the solution is not necessarily unique.

.. [JNie] J\. Nie, *The A-Truncated K-Moment Problem*, Found. Comput. Math., Vol.14(6), 1243-1276 (2014).

Optimization of Rational Functions
==================================

Given two polynomials :math:`p(X), q(X), g_1(X),\dots,g_m(X)`, the minimum of :math:`\frac{p(X)}{q(X)}` over
:math:`K=\{x:g_i(x)\ge0,~i=1,\dots,m\}` is equal to 

.. math::

	\left\lbrace
	\begin{array}{ll}
		\min & \int p(X)~d\mu \\
		\textrm{subject to} & \\
			& \int q(X)~d\mu = 1, \\
			& \mu\in\mathcal{M}^+(K).
	\end{array}\right.

Note that in this case :math:`\mu` is not taken to be a probability measure, but instead :math:`\int q(X)~d\mu = 1`.
We can use ``SDPRelaxations.Probability = False`` to relax the probability condition on :math:`\mu` and use moment
constraints to enforce :math:`\int q(X)~d\mu = 1`. The following example explains this.

**Example:** Find the minimum of :math:`\frac{x^2-2x}{x^2+2x+1}`::

	from sympy import *
	from Irene import *
	# define the symbolic variable
	x = Symbol('x')
	# initiate the SDPRelaxations object
	Rlx = SDPRelaxations([x])
	# settings
	Rlx.Probability = False
	# set the objective
	Rlx.SetObjective(x**2 - 2*x)
	# moment constraint
	Rlx.MomentConstraint(Mom(x**2+2*x+1) == 1)
	# set the sdp solver
	Rlx.SetSDPSolver('cvxopt')
	# initiate the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution

The result is::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 0.167912006378 seconds
	              Run Time: 0.008987 seconds
	Primal Objective Value: -0.333333666913
	  Dual Objective Value: -0.333333667469
	Feasible solution for moments of order 1

.. note::

	Beside ``SDPRelaxations.Probability`` there is another attribute ``SDPRelaxations.PSDMoment``
	which by default is set to ``True`` and makes sure that the sdp solver assumes positivity for
	the moment matrix.

Optimization over Varieties
=============================

Now we employ the results of [GIKM]_ to solve more complex optimization problems. The main idea is to represent the given function space 
as a quotient of a suitable polynomial algebra.

Suppose that we want to optimize the function :math:`\sqrt[3]{(xy)^2}-x+y^2` over the closed disk with radius 3.
In order to deal with the term :math:`\sqrt[3]{(xy)^2}`, we introduce an algebraic relation to ``SDPRelaxations`` object and give a 
monomial order for Groebner basis computations (default is `lex` for lexicographic order).
Clearly :math:`xy-\sqrt[3]{(xy)}^3=0`. Therefore by introducing an auxiliary variable or function symbol, say :math:`f(x,y)` the problem
can be stated in the quotient of :math:`\frac{\mathbb{R}[x,y,f]}{\langle xy-f^3\rangle}`. To check the result of ``SDPRelaxations`` we
employ ``scipy.optimize.minimize`` with two solvers ``COBYLA`` and ``COBYLA`` as well as two solvers, `Augmented Lagrangian Particle Swarm 
Optimizer` and `Non Sorting Genetic Algorithm II` from `pyOpt <http://www.pyopt.org/>`_::

	from sympy import *
	from Irene import *
	# introduce variables
	x = Symbol('x')
	y = Symbol('y')
	f = Function('f')(x, y)
	# define algebraic relations
	rel = [x * y - f**3]
	# initiate the Relaxation object
	Rlx = SDPRelaxations([x, y, f], rel)
	# set the monomial order
	Rlx.SetMonoOrd('lex')
	# set the objective
	Rlx.SetObjective(f**2 - x + y**2)
	# add support constraints
	Rlx.AddConstraint(9 - x**2 - y**2 >= 0)
	# set the solver
	Rlx.SetSDPSolver('cvxopt')
	# Rlx.MomentsOrd(2)
	# initialize the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	# output
	print Rlx.Solution
	# using scipy
	from numpy import power
	from scipy.optimize import minimize
	fun = lambda x: power(x[0]**2 * x[1]**2, 1. / 3.) - x[0] + x[1]**2
	cons = (
	    {'type': 'ineq', 'fun': lambda x: 9 - x[0]**2 - x[1]**2})
	sol1 = minimize(fun, (0, 0), method='COBYLA', constraints=cons)
	sol2 = minimize(fun, (0, 0), method='SLSQP', constraints=cons)
	print "solution according to 'COBYLA'"
	print sol1
	print "solution according to 'SLSQP'"
	print sol2

	# pyOpt
	from pyOpt import *

	def objfunc(x):
		from numpy import power
		f = power(x[0]**2 * x[1]**2, 1. / 3.) - x[0] + x[1]**2
		g = [x[0]**2 + x[1]**2 - 9]
		fail = 0
		return f, g, fail

	opt_prob = Optimization('A third root function', objfunc)
	opt_prob.addVar('x1', 'c', lower=-3, upper=3, value=0.0)
	opt_prob.addVar('x2', 'c', lower=-3, upper=3, value=0.0)
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

The output will be::
	
	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 0.12473487854 seconds
	              Run Time: 0.004865 seconds
	Primal Objective Value: -2.99999997394
	  Dual Objective Value: -2.9999999473
	Feasible solution for moments of order 1

	solution according to 'COBYLA'
	     fun: -0.99788411120450926
	   maxcv: 0.0
	 message: 'Optimization terminated successfully.'
	    nfev: 25
	  status: 1
	 success: True
	       x: array([  9.99969494e-01,   9.52333693e-05])
	 solution according to 'SLSQP'
	     fun: -2.9999975825413681
	     jac: array([  -0.99999923,  689.00398242,    0.        ])
	 message: 'Optimization terminated successfully.'
	    nfev: 64
	     nit: 13
	    njev: 13
	  status: 0
	 success: True
	       x: array([  3.00000000e+00,  -1.25290367e-09])
	
	ALPSO Solution to A third root function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.1174
	    Total Function Evaluations:      1720
	    Lambda: [ 0.00023458]
	    Seed: 1482111093.38230896

	    Objectives:
	        Name        Value        Optimum
		     f        -2.99915             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      3.000000      -3.00e+00     3.00e+00 
		     x2       c	      0.000008      -3.00e+00     3.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= 0.000000 <= 0.00e+00

	--------------------------------------------------------------------------------


	NSGA-II Solution to A third root function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.3833
	    Total Function Evaluations:          

	    Objectives:
	        Name        Value        Optimum
		     f        -2.99898             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      3.000000      -3.00e+00     3.00e+00 
		     x2       c	     -0.000011      -3.00e+00     3.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -0.000000 <= 0.00e+00

	--------------------------------------------------------------------------------



Optimization over arbitrary functions
======================================

Any given algebra can be represented as a quotient of a suitable polynomial algebra (on possibly infinitely many variables).
Since optimization problems usually involve finitely many functions and constraints, we can apply the technique introduced in the previous
section, as soon as we figure out the quotient representation of the function space.

Let us walk through the procedure by solving some examples.

**Example 1.** Find the optimum value of the following program:

.. math::
	\left\lbrace
	\begin{array}{ll}
		\min & -(\sin(x)-1)^3-(\sin(x)-\cos(y))^4-(\cos(y)-3)^2\\
		\textrm{subject to } & \\
		& 10 - (\sin(x) - 1)^2\ge 0,\\
		& 10 - (\sin(x) - \cos(y))^2\ge 0,\\
		& 10 - (\cos(y) - 3)^2\ge 0.
	\end{array}
	\right.

Let us introduce four symbols to represent trigonometric functions:

.. math::
	\begin{array}{|cc|cc|}
		\hline
		f : & \sin(x) & g : & \cos(x)\\
		\hline
		h : & \sin(y) & k : & \cos(y)\\
		\hline
	\end{array}

Then the quotient algebra :math:`\frac{\mathbb{R}[f,g,h,k]}{I}` where :math:`I=\langle f^2+g^2-1, h^2+k^2-1\rangle` is the right framework to solve 
the optimization problem. We also compare the outcome of ``SDPRelaxations`` with ``scipy`` and ``pyswarm``::

	from sympy import *
	from Irene import *
	# introduce variables
	x = Symbol('x')
	f = Function('f')(x)
	g = Function('g')(x)
	h = Function('h')(x)
	k = Function('k')(x)
	# define algebraic relations
	rels = [f**2 + g**2 - 1, h**2 + k**2 - 1]
	# initiate the Relaxation object
	Rlx = SDPRelaxations([f, g, h, k], rels)
	# set the monomial order
	Rlx.SetMonoOrd('lex')
	# set the objective
	Rlx.SetObjective(-(f - 1)**3 - (f - k)**4 - (k - 3)**2)
	# add support constraints
	Rlx.AddConstraint(10 - (f - 1)**2 >= 0)
	Rlx.AddConstraint(10 - (f - k)**2 >= 0)
	Rlx.AddConstraint(10 - (k - 3)**2 >= 0)
	# set the solver
	Rlx.SetSDPSolver('csdp')
	# initialize the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	# output
	print Rlx.Solution
	# using scipy
	from scipy.optimize import minimize
	fun = lambda x: -(sin(x[0]) - 1)**3 - (sin(x[0]) -
	                                       cos(x[1]))**4 - (cos(x[1]) - 3)**2
	cons = (
	    {'type': 'ineq', 'fun': lambda x: 10 - (sin(x[0]) - 1)**2},
	    {'type': 'ineq', 'fun': lambda x: 10 - (sin(x[0]) - cos(x[1]))**2},
	    {'type': 'ineq', 'fun': lambda x: 10 - (cos(x[1]) - 3)**2})
	sol1 = minimize(fun, (0, 0), method='COBYLA', constraints=cons)
	sol2 = minimize(fun, (0, 0), method='SLSQP', constraints=cons)
	print "solution according to 'COBYLA':"
	print sol1
	print "solution according to 'SLSQP':"
	print sol2
	# pyOpt
	from pyOpt import *


	def objfunc(x):
	    from numpy import sin, cos
	    f = -(sin(x[0]) - 1)**3 - (sin(x[0]) - cos(x[1]))**4 - (cos(x[1]) - 3)**2
	    g = [
	        (sin(x[0]) - 1)**2 - 10,
	        (sin(x[0]) - cos(x[1]))**2 - 10,
	        (cos(x[1]) - 3)**2 - 10
	    ]
	    fail = 0
	    return f, g, fail

	opt_prob = Optimization('A trigonometric function', objfunc)
	opt_prob.addVar('x1', 'c', lower=-10, upper=10, value=0.0)
	opt_prob.addVar('x2', 'c', lower=-10, upper=10, value=0.0)
	opt_prob.addObj('f')
	opt_prob.addCon('g1', 'i')
	opt_prob.addCon('g2', 'i')
	opt_prob.addCon('g3', 'i')
	# Augmented Lagrangian Particle Swarm Optimizer
	alpso = ALPSO()
	alpso(opt_prob)
	print opt_prob.solution(0)
	# Non Sorting Genetic Algorithm II
	nsg2 = NSGA2()
	nsg2(opt_prob)
	print opt_prob.solution(1)

Solutions are::

	Solution of a Semidefinite Program:
	                Solver: CSDP
	                Status: Optimal
	   Initialization Time: 3.22915506363 seconds
	              Run Time: 0.016662 seconds
	Primal Objective Value: -12.0
	  Dual Objective Value: -12.0
	Feasible solution for moments of order 2

	solution according to 'COBYLA':
	     fun: -11.824901993777621
	   maxcv: 1.7763568394002505e-15
	 message: 'Optimization terminated successfully.'
	    nfev: 42
	  status: 1
	 success: True
	       x: array([ 1.57064986,  1.7337948 ])
	solution according to 'SLSQP':
	     fun: -11.9999999999720
	     jac: array([ -2.94446945e-05,  -1.78813934e-05,   0.00000000e+00])
	 message: 'Optimization terminated successfully.'
	    nfev: 23
	     nit: 5
	    njev: 5
	  status: 0
	 success: True
	       x: array([ -1.57079782e+00,  -6.42618794e-07])

	ALPSO Solution to A trigonometric function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.3503
	    Total Function Evaluations:      3640
	    Lambda: [ 0.         0.         2.0077542]
	    Seed: 1482111691.32805490

	    Objectives:
	        Name        Value        Optimum
		     f        -11.8237             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      7.854321      -1.00e+01     1.00e+01 
		     x2       c	      4.549489      -1.00e+01     1.00e+01 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -10.000000 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -8.649336 <= 0.00e+00
		     g3   	  i       -1.00e+21 <= -0.000612 <= 0.00e+00

	--------------------------------------------------------------------------------


	NSGA-II Solution to A trigonometric function
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.7216
	    Total Function Evaluations:          

	    Objectives:
	        Name        Value        Optimum
		     f             -12             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	     -7.854036      -1.00e+01     1.00e+01 
		     x2       c	      0.000004      -1.00e+01     1.00e+01 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -6.000000 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -6.000000 <= 0.00e+00
		     g3   	  i       -1.00e+21 <= -6.000000 <= 0.00e+00

	--------------------------------------------------------------------------------

SOS Decomposition
======================================

Let :math:`f_*` be the result of ``SDPRelaxations.Minimize()``, then :math:`f-f_*\in Q_{\bf g}`.
Therefore, there exist :math:`\sigma_0,\sigma_1,\dots,\sigma_m\in \sum A^2` such that
:math:`f-f_*=\sigma_0+\sum_{i=1}^m\sigma_i g_i`. Once the ``Minimize()`` is called, the method
``SDPRelaxations.Decompose()`` returns this a dictionary of elements of :math:`A` of the form
``{0:[a(0, 1), ..., a(0, k_0)], ..., m:[a(m, 1), ..., a(m, k_m)}`` such that

.. math::
	f-f_* = \sum_{i=0}^{m}g_i\sum_{j=1}^{k_i} a^2_{ij},

where :math:`g_0=1`.

Usually there are extra coefficients that are very small in absolute value as a result of 
round off error that should be ignored.

The following example shows how to employ this functionality::

	from sympy import *
	from Irene import SDPRelaxations
	# define the symbolic variables and functions
	x = Symbol('x')
	y = Symbol('y')
	z = Symbol('z')

	Rlx = SDPRelaxations([x, y, z])
	Rlx.SetObjective(x**3 + x**2 * y**2 + z**2 * x * y - x * z)
	Rlx.AddConstraint(9 - (x**2 + y**2 + z**2) >= 0)
	# initiate the SDP
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	print Rlx.Solution
	# extract decomposition
	V = Rlx.Decompose()
	# test the decomposition
	sos = 0
	for v in V:
	    # for g0 = 1
	    if v == 0:
	        sos = expand(Rlx.ReduceExp(sum([p**2 for p in V[v]])))
	    # for g1, the constraint
	    else:
	        sos = expand(Rlx.ReduceExp(
	            sos + Rlx.Constraints[v - 1] * sum([p**2 for p in V[v]])))
	sos = sos.subs(Rlx.RevSymDict)
	pln = Poly(sos).as_dict()
	pln = {ex:round(pln[ex],5) for ex in pln}
	print Poly(pln, (x,y,z)).as_expr()

The output looks like this::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 0.875229120255 seconds
	              Run Time: 0.031426 seconds
	Primal Objective Value: -27.4974076889
	  Dual Objective Value: -27.4974076213
	Feasible solution for moments of order 2

	1.0*x**3 + 1.0*x**2*y**2 + 1.0*x*y*z**2 - 1.0*x*z + 27.49741

The ``Resume`` method
=======================================

It happens from time to time that one needs to stop the process of ``SDPRelaxations`` to look into 
its progress and/or run the code later. This has been accommodated thanks to python's support for
serialization and error handling.
Since the initialization of the final SDP is the most time consuming part of the process, if one 
breaks this via `Ctrl-c`, the object will save all the computation that has been done so far in 
a `.rlx` file named with the name of the object. So, if one wants to resume the process later, it
suffices to call the ``Resume`` method after instantiation and leave the program out and continue 
the initialization via calling ``InitSDP`` method.

The ``SDRelaxSol``
=======================================

This object is a container for the solution of ``SDPRelaxation`` objects.
It contains the following informations:
	
	- `Primal`: the value of the SDP in primal form,
	- `Dual`: the value of the SDP in dual form,
	- `RunTime`: the run time of the sdp solver,
	- `InitTime`: the total time consumed for initialization of the sdp,
	- `Solver`: the name of sdp solver,
	- `Status`: final status of the sdp solver,
	- `RelaxationOrd`: order of relaxation,
	- `TruncatedMmntSeq`: a dictionary of resulted moments,
	- `MomentMatrix`: the resulted moment matrix,
	- `ScipySolver`: the scipy solver to extract solutions,
	- `err_tol`: the minimum value which is considered to be nonzero,
	- `Support`: the support of discrete measure resulted from ``SDPRelaxation.Minimize()``,
	- `Weights`: corresponding weights for the Dirac measures.

The ``SDRelaxSol`` after initiation is an iterable object. The moments can be retrieved by
passing the index to the iterable ``SDRelaxSol[idx]``.

Extracting solutions
---------------------------------------
By default, the support of the measure is not calculated, but it can be approximated by calling 
the method ``SDRelaxSol.ExtractSolution()``. 

There exists an exact theoretical method for extracting the support of the solution measure as explained 
in [HL]_. But because of the numerical error of sdp solvers, computing rank and hence the support is quite 
difficult. So, ``SDRelaxSol.ExtractSolution()`` estimates the rank numerically by assuming that eigenvalues 
with absolute value less than ``err_tol`` which by default is set to ``SDPRelaxation.ErrorTolerance``.

Two methods are implemented for extracting solutions:

	- **Lasserre-Henrion** method as explained in [HL]_. To employ this method simply call ``SDRelaxSol.ExtractSolution('LH', card)``, where ``card`` is the maximum cardinality of the support.

	- **Moment Matching** method which employs ``scipy.optimize.root`` to approximate the support. The default ``scipy`` solver is set to `lm`, but other solvers can be selected using ``SDRelaxSol.SetScipySolver(solver)``. It is not guaranteed that scipy solvers return a reliable answer, but modifying sdp solvers and other parameters like ``SDPRelaxation.ErrorTolerance`` may help to get better results. To use this method call ``SDRelaxSol.ExtractSolution('scipy', card)`` where ``card`` is as above.

**Example 1.** Solve and find minimizers of :math:`x^2+y^2+z^4` where :math:`x+y+z=4`::

	from sympy import *
	from Irene import *

	x, y, z = symbols('x,y,z')

	Rlx = SDPRelaxations([x, y, z])
	Rlx.SetSDPSolver('cvxopt')
	Rlx.SetObjective(x**2 + y**2 + z**4)
	Rlx.AddConstraint(Eq(x + y + z, 4))
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	# extract support
	Rlx.Solution.ExtractSolution('LH', 1)
	print Rlx.Solution

	# pyOpt
	from pyOpt import *

	def objfunc(x):
		f = x[0]**2 + x[1]**2 + x[2]**4
		g = [x[0] + x[1] + x[2] - 4]
		fail = 0
		return f, g, fail

	opt_prob = Optimization('Testing solutions', objfunc)
	opt_prob.addVar('x1', 'c', lower=-4, upper=4, value=0.0)
	opt_prob.addVar('x2', 'c', lower=-4, upper=4, value=0.0)
	opt_prob.addVar('x3', 'c', lower=-4, upper=4, value=0.0)
	opt_prob.addObj('f')
	opt_prob.addCon('g1', 'e')
	# Augmented Lagrangian Particle Swarm Optimizer
	alpso = ALPSO()
	alpso(opt_prob)
	print opt_prob.solution(0)

The output is::

	Solution of a Semidefinite Program:
	                Solver: CVXOPT
	                Status: Optimal
	   Initialization Time: 1.59334087372 seconds
	              Run Time: 0.021102 seconds
	Primal Objective Value: 5.45953579912
	  Dual Objective Value: 5.45953586121
	               Support:
			(0.91685039306810523, 1.541574317520042, 1.5415743175200163)
	        Support solver: Lasserre--Henrion
	Feasible solution for moments of order 2

	ALPSO Solution to Testing solutions
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.1443
	    Total Function Evaluations:      1720
	    Lambda: [-3.09182651]
	    Seed: 1482274189.55335808

	    Objectives:
	        Name        Value        Optimum
		     f         5.46051             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      1.542371      -4.00e+00     4.00e+00 
		     x2       c	      1.541094      -4.00e+00     4.00e+00 
		     x3       c	      0.916848      -4.00e+00     4.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1       e                0.000314 = 0.00e+00

	--------------------------------------------------------------------------------

**Example 2.** Minimize :math:`-(x-1)^2-(x-y)^2-(y-3)^2` where :math:`1-(x-1)^2\ge0`, 
:math:`1-(x-y)^2\ge0` and :math:`1-(y-3)^2\ge0`. It has three minimizers 
:math:`(2, 3), (1, 2)`, and :math:`(2, 2)`::

	from sympy import *
	from Irene import *

	x, y = symbols('x, y')

	Rlx = SDPRelaxations([x, y])
	Rlx.SetSDPSolver('csdp')
	Rlx.SetObjective(-(x - 1)**2 - (x - y)**2 - (y - 3)**2)
	Rlx.AddConstraint(1 - (x - 1)**2 >= 0)
	Rlx.AddConstraint(1 - (x - y)**2 >= 0)
	Rlx.AddConstraint(1 - (y - 3)**2 >= 0)
	Rlx.MomentsOrd(2)
	Rlx.InitSDP()
	# solve the SDP
	Rlx.Minimize()
	# extract support
	Rlx.Solution.ExtractSolution('LH')
	print Rlx.Solution

	# pyOpt
	from pyOpt import *


	def objfunc(x):
	    f = -(x[0] - 1)**2 - (x[0] - x[1])**2 - (x[1] - 3)**2
	    g = [
	        (x[0] - 1)**2 - 1,
	        (x[0] - x[1])**2 - 1,
	        (x[1] - 3)**2 - 1
	    ]
	    fail = 0
	    return f, g, fail

	opt_prob = Optimization("Lasserre's Example", objfunc)
	opt_prob.addVar('x1', 'c', lower=-3, upper=3, value=0.0)
	opt_prob.addVar('x2', 'c', lower=-3, upper=3, value=0.0)
	opt_prob.addObj('f')
	opt_prob.addCon('g1', 'i')
	opt_prob.addCon('g2', 'i')
	opt_prob.addCon('g3', 'i')
	# Augmented Lagrangian Particle Swarm Optimizer
	alpso = ALPSO()
	alpso(opt_prob)
	print opt_prob.solution(0)
	# Non Sorting Genetic Algorithm II
	nsg2 = NSGA2()
	nsg2(opt_prob)
	print opt_prob.solution(1)

which results in::

	Solution of a Semidefinite Program:
	                Solver: CSDP
	                Status: Optimal
	   Initialization Time: 0.861004114151 seconds
	              Run Time: 0.00645 seconds
	Primal Objective Value: -2.0
	  Dual Objective Value: -2.0
	               Support:
			(2.000000006497352, 3.000000045123556)
			(0.99999993829586131, 1.9999999487412694)
			(1.9999999970209055, 1.9999999029899564)
	        Support solver: Lasserre--Henrion
	Feasible solution for moments of order 2


	ALPSO Solution to Lasserre's Example
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.1353
	    Total Function Evaluations:      1720
	    Lambda: [ 0.08278879  0.08220848  0.        ]
	    Seed: 1482307696.27431393

	    Objectives:
	        Name        Value        Optimum
		     f              -2             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      1.999967      -3.00e+00     3.00e+00 
		     x2       c	      3.000000      -3.00e+00     3.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -0.000065 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= 0.000065 <= 0.00e+00
		     g3   	  i       -1.00e+21 <= -1.000000 <= 0.00e+00

	--------------------------------------------------------------------------------


	NSGA-II Solution to Lasserre's Example
	================================================================================

	        Objective Function: objfunc

	    Solution: 
	--------------------------------------------------------------------------------
	    Total Time:                    0.2406
	    Total Function Evaluations:          

	    Objectives:
	        Name        Value        Optimum
		     f        -1.99941             0

		Variables (c - continuous, i - integer, d - discrete):
	        Name    Type       Value       Lower Bound  Upper Bound
		     x1       c	      1.999947      -3.00e+00     3.00e+00 
		     x2       c	      2.000243      -3.00e+00     3.00e+00 

		Constraints (i - inequality, e - equality):
	        Name    Type                    Bounds
		     g1   	  i       -1.00e+21 <= -0.000106 <= 0.00e+00
		     g2   	  i       -1.00e+21 <= -1.000000 <= 0.00e+00
		     g3   	  i       -1.00e+21 <= -0.000486 <= 0.00e+00

	--------------------------------------------------------------------------------

`Irene` detects all minimizers correctly, but each `pyOpt` solvers only detect one.
Note that we did not specify number of solutions, but the solver extracted them all.

.. [HL] D\. Henrion and J-B. Lasserre, *Detecting Global Optimality and Extracting Solutions in GloptiPoly*, Positive Polynomials in Control, LNCIS 312, 293-310 (2005).