=============================
Interpolation
=============================

Lagrange interpolation
=============================

Suppose that a list of :math:`m+1` points :math:`\{(x_0, y_0),\dots,(x_{m}, y_{m})\}` in :math:`\mathbb{R}^2` is given, such that
:math:`x_i\neq x_j`, if :math:`i\neq j`. Then

.. math::
	p({\bf x}) = \sum_{i=0}^m y_i\ell_i({\bf x}),

is a polynomial of degree `m` which passes through all the given points. Here 

.. math::
	\ell_i({\bf x}) = \prod_{j\neq i}\frac{{\bf x}-x_j}{x_i-x_j},

are *Lagrange Basis Polynomials*.

This procedure can be extended to multivariate case as well.

Suppose that a list of points :math:`\{{x}_1,\dots,{x}_{\rho}\}` in :math:`\mathbb{R}^n` and a list of corresponding 
values :math:`\{y_1,\dots,y_\rho\}` are given. Let's denote by :math:`{\bf X}` the tuple :math:`({\bf X}_1,\dots,{\bf X}_n)` of variables
and :math:`{\bf X}_1^{e_1}\cdots{\bf X}_n^{e_n}` by :math:`{\bf X}^{\bf e}`, where :math:`{\bf e}=(e_1,\dots,e_n)`.

If for some :math:`m>0`, we have :math:`\rho={{m+n}\choose{m}}`, then the number of given points matches the number of monomials of
degree at most `m` in the polynomial basis. Denote the exponents of these monomials by :math:`{\bf e}_i`, :math:`i=1,\dots,\rho` and let

.. math::
	\begin{aligned}
		D=(
		x_1^{{\bf e}_1} & \dots & x_1^{{\bf e}_{\rho}}\
		\vdots & & \vdots \
		x_{\rho}^{{\bf e}_1} & \dots & x_{\rho}^{{\bf e}_{\rho}}\
		)
	\end{aligned}

and for :math:`1\leq j\leq\rho`:

.. math::
	\begin{aligned}
		D_j=(
		x_1^{{\bf e}_1} & \dots & x_1^{{\bf e}_{\rho}}\
		\vdots & \vdots & \vdots \
		x_{j-1}^{{\bf e}_1} & \dots & x_{j-1}^{{\bf e}_{\rho}}\
		{\bf X}^{{\bf e}_1} & \dots & {\bf X}^{{\bf e}_{\rho}}\
		x_{j+1}^{{\bf e}_1} & \dots & x_{j+1}^{{\bf e}_{\rho}}\
		\vdots & \vdots & \vdots \
		x_{\rho}^{{\bf e}_1} & \dots & x_{\rho}^{{\bf e}_{\rho}}\
		).
	\end{aligned}

Then the polynomial

.. math::
	p({\bf X}) = \sum_{i=1}^{\rho} y_i\frac{|D_i|}{|D|},

interpolates the given list of points and their corresponding values. Here :math:`|M|` denotes the determinant of `M`.

The above procedure is implemented in the ``Interpolation`` module. The following code provides an example in 2 dimensional case::

	from sympy import *
	from pyProximation import Interpolation
	# define symbolic variables
	x = Symbol('x')
	y = Symbol('y')
	# initiate the interpolation instance
	Inter = Interpolation([x, y], 'sympy')
	# list of points
	points = [(-1, 1), (-1, 0), (0, 0), (0, 1), (1, 0), (1, -1)]
	# corresponding values
	values = [-1, 2, 0, 2, 1, 0]
	# interpolate
	p = Inter.Interpolate(points, values)
	# print the result
	print(p)

Lagrange interpolation as an :math:`L^2`-orthogonal projection
==============================================================
The ``Interpolation`` and ``OrthSystem`` modules are two views of the same
operator. Let :math:`X=\{x_0,\dots,x_m\}` be :math:`m+1` distinct nodes and let

.. math::
	\mu = \sum_{k=0}^{m} w_k\,\delta_{x_k}, \qquad w_k > 0,

be a discrete measure (the example below uses :math:`w_k=1`). On the polynomials,
define the associated inner product

.. math::
	(f,g)_\mu = \int f\,g\,d\mu = \sum_{k=0}^{m} w_k\,f(x_k)\,g(x_k),

and let :math:`V=\Pi_m` be the space of polynomials of degree at most :math:`m`,
so that :math:`\dim V = m+1` equals the number of atoms of :math:`\mu`.

Lagrange interpolation is the :math:`L^2(\mu)`-orthogonal projection
--------------------------------------------------------------------
The :math:`L^2(\mu)`-orthogonal projection :math:`P_V f` of :math:`f` onto
:math:`V` is the unique :math:`p\in V` satisfying :math:`p(x_k)=f(x_k)` for
:math:`k=0,\dots,m` --- that is, the Lagrange interpolant.

*Proof sketch.* The form :math:`(\cdot,\cdot)_\mu` is a genuine inner product on
:math:`V`: if :math:`v\in V` and :math:`(v,v)_\mu=0`, then :math:`v(x_k)=0` for
every :math:`k`, so :math:`v` has :math:`m+1` roots and hence :math:`v=0`. Thus
:math:`P_V` is well-defined. By the projection characterization,
:math:`p=P_V f` iff :math:`(f-p,v)_\mu=0` for all :math:`v\in V`, i.e.

.. math::
	\sum_{k=0}^{m} w_k\big(f(x_k)-p(x_k)\big)\,v(x_k)=0 \qquad \forall\, v\in V.

Because the evaluation functionals :math:`v\mapsto v(x_k)` span :math:`V^*`
(the Lagrange basis :math:`\{\ell_k\}_{k=0}^m` is the dual basis to the point
evaluations), this orthogonality is equivalent to
:math:`f(x_k)-p(x_k)=0` for each :math:`k`, i.e. :math:`p(x_k)=f(x_k)`.
Uniqueness follows from :math:`\dim V=m+1`.

Equivalently, in the orthonormal basis :math:`\{u_0,\dots,u_m\}` obtained from
:math:`\{1,x,\dots,x^m\}` by Gram--Schmidt with respect to
:math:`(\cdot,\cdot)_\mu`, the interpolant is the truncated Hilbert series

.. math::
	p = \sum_{i=0}^{m}\langle f,u_i\rangle_\mu\,u_i
	  = \sum_{i=0}^{m}\Big(\sum_{k=0}^{m} w_k\,f(x_k)\,u_i(x_k)\Big)\,u_i,

which is exactly what ``OrthSystem.FormBasis`` / ``OrthSystem.Series`` compute.
The following example builds the same interpolant two ways and shows they
coincide::

	# symbolic variable
	x = Symbol('x')
	# function to be approximated
	g = sin(x)*exp(sin(x)*x)#x*sin(x)
	# its numerical equivalent
	g_ = lambdify(x, g, 'numpy')
	# number of approximation terms
	n = 6
	# half interval length
	l = 3.1
	# interpolation points and values
	Xs = [[-3], [-2], [-1], [0], [1], [2], [3]]
	Ys = [g_(Xs[i][0]) for i in range(7)]
	# a discrete measure
	supp = {-3:1, -2:1, -1:1, 0:1, 1:1, 2:1, 3:1}
	M = Measure(supp)
	# orthogonal system
	S = OrthSystem([x], [(-l, l)])
	# link the measure
	S.SetMeasure(M)
	# polynomial basis
	B = S.PolyBasis(n)
	# link the basis to the orthogonal system
	S.Basis(B)
	# form the orthonormal basis
	S.FormBasis()
	# calculate coefficients
	cfs = S.Series(g)
	# orthogonal approximation
	aprx = sum([S.OrthBase[i]*cfs[i] for i in range(len(B))])
	# interpolate
	Intrp = Interpolation([x])
	intr = Intrp.Interpolate(Xs, Ys)
	print(intr)

The two constructions agree to machine precision: the ``OrthSystem`` truncated
series and the ``Interpolation`` Lagrange form both return the same degree-6
polynomial passing through the seven nodes (verified: max node difference
:math:`\le 10^{-15}`).

Interpolation and least-squares are one projection in two regimes
-----------------------------------------------------------------
The same operator :math:`P_V f` changes character with the relation between
:math:`\dim V` and the number :math:`m+1` of atoms of :math:`\mu`. The form
:math:`(\cdot,\cdot)_\mu` is a genuine (non-degenerate) inner product on
:math:`V=\Pi_n` exactly when :math:`n+1\le m+1`, i.e. :math:`\dim V\le` the
number of atoms.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - :math:`\dim V` vs atoms
     - What :math:`P_V f` is
     - Characterizing property
   * - :math:`\dim V < m+1` (underdetermined)
     - weighted least-squares fit
     - minimizes :math:`\sum w_k(f-p)^2`; normal equations; residual :math:`\neq 0`
   * - :math:`\dim V = m+1` (exact fit)
     - Lagrange interpolant
     - interpolates exactly; residual :math:`= 0`
   * - :math:`\dim V > m+1` (overdetermined)
     - degenerate (no unique projection)
     - :math:`m+1` interpolants; the minimum-norm (Moore--Penrose) interpolant is canonical


In the underdetermined regime the projection is the weighted least-squares fit:
it is the unique :math:`p\in V` minimizing :math:`\sum_k w_k\big(f(x_k)-p(x_k)\big)^2`,
equivalently the solution of the normal equations
:math:`(E^	op W E)\,c = E^	op W y` with :math:`E` the Vandermonde matrix.
The transition is sharp: the monomial Gram matrix :math:`E^	op W E` is
nonsingular for :math:`\dim V\le m+1` and singular for :math:`\dim V>m+1`.

Taylor / Maclaurin series lie outside the ordinary :math:`L^2` framework
------------------------------------------------------------------------
Unlike interpolation and least-squares, the Taylor expansion is **not** an
:math:`L^2(\mu)`-orthogonal projection for any ordinary positive measure
:math:`\mu` on an interval. Its coefficient functionals are the point
derivatives :math:`f\mapsto f^{(k)}(a)`, which are *unbounded* on
:math:`L^2`: the sequence
:math:`f_n(x)=n\,e^{-n^2(x-a)^2}(x-a)^2` satisfies
:math:`\|f_n\|_{L^2}	o 0` while :math:`f_n''(a)	o\infty`, so no
:math:`L^2`-inner product can represent them. Taylor series are recovered only
after extending "measure" to *distributions*: with
:math:`\langle f,\delta_a^{(k)}\rangle=(-1)^k f^{(k)}(a)`, the degree-`n`
Taylor polynomial is the truncated expansion

.. math::
	T_n(f)(x) = \sum_{k=0}^{n}\langle f,\,(-1)^k\delta_a^{(k)}/k!\rangle\,(x-a)^k,

which requires :math:`f` to be :math:`C^k` (a distributional pairing, not an
:math:`L^2` inner product). This is why ``OrthSystem`` naturally produces
interpolants and least-squares fits, while Taylor expansions are built
separately in ``approx.rst``.

Verification
------------
The identities above are checked numerically in ``verify_unified_approx.py``
(Lagrange = :math:`L^2` projection, 1D and 2D; Taylor :math:`\neq`
:math:`L^2` projection; distributional Taylor) and ``verify_least_squares.py``
(least-squares = :math:`L^2` projection in the underdetermined regime, and the
:math:`\dim V`-vs-atoms boundary). Both run in the ``IreneRewrite`` virtual
environment.
