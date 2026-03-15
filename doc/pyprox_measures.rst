=========================
Measures
=========================

Density function
=========================

The `pyProximation` implements two different scenarios for measure spaces:
	1. Continuous case, where support of the measure is given as a compact subspace (box) of :math:`\mathbb{R}^n`, and
	2. Discrete case, where a finite set of points and their weights are given.

Continuous measure spaces
-------------------------
For the continuous case, generally assume that the support of the measure is a product of closed interval, i.e.,
:math:`\prod_{i=1}^{n}[a_i, b_i]`, where for each `i`, :math:`a_i<b_i`.
Such a set can be defined as a list of ordered pairs of real numbers as ``[(a1, b1), (a2, b2), ..., (an, bn)]``.
Moreover, when we speak about a subset of the support, we always refer to a box, defined as a list of 2-tuples.

In this case, the measure is given implicitly as a density function :math:`w(x)`. So the measure of a set `S` is given by

.. math::
	\mu(S) = \int_S w(x)dx.

For example, the following code defines the Lebesgue measure on :math:`[-1, 1]\times[-1, 1]` and finds the measure of 
the set :math:`[0, 1]\times[-1, 0]`::

	# import the Measure class
	from pyProximation import Measure
	# define the support of the measure
	D = [(-1, 1), (-1, 1)]
	# define the measure with the constant density 1
	M = Measure(D, 1)
	# define a set called S
	S = [(0, 1), (-1, 0)]
	# find the measure of the set S
	print M.measure(S)

Discrete measure spaces
-------------------------
In this case, the measure is basically a convex combination of Dirac measure. Given a set :math:`X=\{x_1, \dots, x_n\}` and
corresponding non-negative weights :math:`w_1,\dots, w_n`, one defines a measure as :math:`\mu = \sum_{i=1}^n w_i \delta_{x_i}`.
Then the measure of a subset :math:`S=\{x_{i_1},\dots,x_{i_k}\}` of `X` is given by

.. math::
	\mu(S) = \int_S d\mu = \sum_{j=1}^k w_{i_j}

The following is a sample code for discrete case::
	
	# import the Measure class
	from pyProximation import Measure
	# define the support and density
	D = {'x1':1, 'x2':.5, 'x3':1.1, 'x4':.6}
	# define the measure
	M = Measure(D)
	# define a set called S
	S = ['x2', 'x3']
	# find the measure of the set S
	print M.measure(S)

Integrals
=======================
Suppose that a measure space :math:`(X, \mu)` and a measurable function `f` on `X` are given. The method ``integral`` computes 
:math:`\int_X fd\mu`. 
If :math:`\mu` is discrete, then `f` can be a dictionary with keys as points of domain and values as evaluation at each point.
Otherwise, `f` is simply a numerical function::

	from pyProximation import Measure
	from numpy import sqrt
	# define the density function
	w = lambda x:1./sqrt(1.-x**2)
	# define the support
	D = [(-1, 1)]
	# initiate the measure space
	M = Measure(D, w)
	# set f(x) = x^2
	f = lambda x: x**2
	# integrate f(x) w.r.t. w(x)
	print M.integral(f)

Or in two dimensions::

	from pyProximation import Measure
	from numpy import sqrt
	# define the density function
	w = lambda x, y:y**2/sqrt(1.-x**2)
	# define the support
	D = [(-1, 1), (-1, 1)]
	# initiate the measure space
	M = Measure(D, w)
	# set f(x, y) = x^2 + y
	f = lambda x, y: x**2 + y
	# integrate f(x, y) w.r.t. w(x, y)
	print M.integral(f)

`p`-norms
=========================
Given a measure space :math:`(X, \mu)` and a measurable function `f`, the `p`-norm of `f`, for a positive `p` is defined as:

.. math::
	\| f \|_p =\left(\int_{X} |f|^p d\mu\right)^{1/p}.

The method ``norm(p, f)`` calculates the above quantity.

Drawing samples
=========================
Suppose that :math:`(X, \mu)` is a measure space and one wishes to draw a sample of size `n` from `X` according to the distribution 
:math:`\mu`. This can be done by the method ``sample(n)`` which returns a list of `n` random points from the support, according to :math:`\mu`.
