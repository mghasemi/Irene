=============================
To Do
=============================

Based on the current implementation, the followings seems to be implemented/modified:

	+ Reduce dependency on SymPy.
	+ Include sdp solvers installation (subject to copyright limitations).
	+ Error handling for CSDP and SDPA failure.

Done
==================

The following to-dos were implemented:

	+ Extract solutions (at least for polynomials)- in v.1.1.0.
	+ SOS decomposition- in v.1.1.0.
	+ Write a ``__str__`` method for ``SDPRelaxations`` printing- in v.1.1.0.
	+ Write a LaTeX method- in v.1.2.0.
	+ Keep track of original expressions before reduction- in v.1.2.0.
	+ Removed dependency on ``joblib``- in v.1.2.1.
	+ Save the current status on break and resume later- in v.1.2.2.
	+ Windows support- in v.1.2.3.