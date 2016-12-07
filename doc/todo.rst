=============================
To Do
=============================

Based on the current implementation, the followings seems to be implemented/modified:

	+ Reduce dependency on SymPy.
	+ Write a ``__str__`` method for ``SDPRelaxations`` printing.
	+ Keep track of original expressions before reduction.
	+ Write a LaTeX method.
	+ Include sdp solvers installation (subject to copyright limitations).
	+ Error handling for CSDP and SDPA failure.
	+ Extract solutions (at least for polynomials).
	+ SOS decomposition.