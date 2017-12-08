=============================
Semidefinite Programming
=============================

A *positive semidefinite* matrix is a symmetric real matrix whose eigenvalues are all nonnegative.
A semidefinite programming problem is simply a linear program where the solutions are positive
semidefinite matrices instead of points in Euclidean space.

Primal and Dual formulations
=============================

A typical semidefinite program (SDP for short) in the primal form is the following optimization problem:

.. math::
	\left\lbrace
	\begin{array}{lll}
		\min & \sum_{i=1}^m b_i x_i & \\
		\textrm{subject to} & & \\
			& \sum_{i=1}^m A_{ij}x_i - C_j \succeq 0 & j=1,\dots,k.
	\end{array}\right.

The dual program associated to the above SDP will be the following:

.. math::
	\left\lbrace
	\begin{array}{lll}
		\max & \sum_{j=1}^k tr(C_j\times Z_j) & \\
		\textrm{subject to} & & \\
			& \sum_{j=1}^k tr(A_{ij}\times Z_j) = b_i & i=1,\dots,m,\\
			& Z_j \succeq 0 & j=1,\dots,k.
	\end{array}\right.

For convenience, we use a block representation for the matrices as follows:

.. math::
	C = \left(
	\begin{array}{cccc}
		C_1 & 0 & 0 & \dots \\
		0 & C_2 & 0 & \dots \\
		\vdots & \dots & \ddots & \vdots \\
		0 & \dots & 0 & C_k
	\end{array}
	\right),

and 

.. math::
	A_i = \left(
	\begin{array}{cccc}
		A_{i1} & 0 & 0 & \dots \\
		0 & A_{i2} & 0 & \dots \\
		\vdots & \dots & \ddots & \vdots \\
		0 & \dots & 0 & A_{ik}
	\end{array}
	\right).

This simplifies the :math:`k` constraints of the primal form in to one constraint 
:math:`\sum_{i=1}^m A_i x_i - C \succeq 0` and the objective and constraints of the 
dual form as :math:`tr(C\times Y)` and :math:`tr(A_i\times Z_i) = b_i` for :math:`i=1,\dots,m`.


The ``sdp`` class
=============================

The ``sdp`` class provides an interface to solve semidefinite programs using various range of
well-known SDP solvers. Currently, the following solvers are supported:

``CVXOPT``
----------------------------

This is a python native convex optimization solver which can be obtained from `CVXOPT <http://cvxopt.org/>`_.
Beside semidefinite programs, it has various other solvers to handle convex optimization problems.
In order to use this solver, the python package ``CVXOPT`` must be installed.

``DSDP``
----------------------------

If `DSDP <http://www.mcs.anl.gov/hs/software/DSDP/>`_ and ``CVXOPT`` are installed and ``DSDP`` is callable from command line, 
then it can be used as a SDP solver. Note that the current implementation uses ``CVXOPT`` to call ``DSDP``, so ``CVXOPT`` is a
requirement too.

``SDPA``
----------------------------

In case one manages to install `SDPA <http://sdpa.sourceforge.net/>`_ and it can be called from command line, one can use
``SDPA`` as a SDP solver.

``CSDP``
----------------------------

Also, if `csdp <https://projects.coin-or.org/Csdp/>`_ is installed and can be reached from command, then it can be used to solve
SDP problems through ``sdp`` class.

To initialize and set the solver to one of the above simply use::

	SDP = sdp('cvxopt') # initializes and uses `cvxopt` as solver.

.. note::
	In windows, one can provide the path to each of the above solvers as the second parameter of the constructor::

		SDP = sdp('csdp', {'csdp':"Path to executable csdp"}) # initializes and uses `csdp` as solver existing at the given path.

Set the :math:`b` vector:
----------------------------

To set the vector :math:`b=(b_1,\dots,b_m)` one should use the method ``sdp.SetObjective`` which takes a list or a numpy array of
numbers as :math:`b`.

Set a block constraint:
----------------------------

To introduce the block of matrices :math:`A_{i1},\dots, A_{ik}` associated with :math:`x_i`, one should use the method
``sdp.AddConstraintBlock`` that takes a list of matrices as blocks.

Set the constant block `C`:
----------------------------

The method ``sdp.AddConstantBlock`` takes a list of square matrices and use them to construct :math:`C`.

Solve the input SDP:
----------------------------

To solve the input SDP simply call the method ``sdp.solve()``. This will call the selected solver on the entered SDP and
the output of the solver will be set as dictionary in ``sdp.Info`` with the following keys:

	+ ``PObj``: The value of the primal objective.
	+ ``DObj``: The value of the dual objective.
	+ ``X``: The final :math:`X` matrix.
	+ ``Z``: The final :math:`Z` matrix.
	+ ``Status``: The final status of the solver.
	+ ``CPU``: Total run time of the solver.

Example:
----------------------------
Consider the following SDP:

.. math::
	\left\lbrace
	\begin{array}{lll}
		\min & x_1 - x_2 + x_3 \\
		\textrm{subject to} & \\
			& \left(\begin{array}{cc}7 & 11\\ 11 & -3 \end{array}\right)x_1 + 
			\left(\begin{array}{cc}-7 & 18\\ 18 & -8 \end{array}\right)x_2 +
			\left(\begin{array}{cc} 2 & 8\\ 8 & -1 \end{array}\right)x_3
			\succeq\left(\begin{array}{cc} -33 & 9\\ 9 & -26 \end{array}\right) \\
			& \left(\begin{array}{ccc}21 & 11 & 0\\ 11 & -10 & -8\\ 0 & -8 & -5\end{array}\right)x_1 + 
			\left(\begin{array}{ccc}0 & -10 & -16\\ -10 & 10 & 10\\ -16 & 10 & -3\end{array}\right)x_2 +
			\left(\begin{array}{ccc} 5 & -2 & 17\\ -2 & 6 & -8\\ 17 & -8 & -6\end{array}\right)x_3
			\succeq\left(\begin{array}{ccc} -14 & -9 & -40\\ -9 & -91 & -10\\ -40 & -10 & -15\end{array}\right) \\
	\end{array}
	\right.

The following code solves the above program::

	from numpy import matrix
	from Irene import sdp
	b = [1, -1, 1]
	C = [matrix([[-33, 9], [9, -26]]),
	     matrix([[-14, -9, -40], [-9, -91, -10], [-40, -10, -15]])]
	A1 = [matrix([[7, 11], [11, -3]]),
	      matrix([[21, 11, 0], [11, -10, -8], [0, -8, -5]])]
	A2 = [matrix([[-7, 18], [18, -8]]),
	      matrix([[0, -10, -16], [-10, 10, 10], [-16, 10, -3]])]
	A3 = [matrix([[2, 8], [8, -1]]),
	      matrix([[5, -2, 17], [-2, 6, -8], [17, -8, -6]])]
	SDP = sdp('cvxopt')
	SDP.SetObjective(b)
	SDP.AddConstantBlock(C)
	SDP.AddConstraintBlock(A1)
	SDP.AddConstraintBlock(A2)
	SDP.AddConstraintBlock(A3)
	SDP.solve()
	print SDP.Info
