=============================
Revision History
=============================

**Version 1.2.5 (Mar 12, 2026)**

	- Expanded documentation from SDP-only emphasis to a unified POP guide covering SDP, geometric programming, and SONC relaxations.
	- Added new chapters for architecture, group-ring foundations, optimization problem representation, geometric relaxations, SONC relaxations, and runnable examples.
	- Added method-selection guidance, dependency matrix, and solver troubleshooting notes.
	- Added explicit theory-to-code mapping for constrained SONC families (Section 3 style equations) and geometric lower-bound construction (Section 4 equation mapping).
	- Expanded API documentation coverage in Sphinx for ``program``, ``grouprings``, ``geometric``, and ``sonc`` modules.
	- Updated Sphinx configuration for modern toolchain compatibility and warning-free documentation builds.

**Version 1.2.2 (Mar 15, 2017)**

	- Serialization and pickling: Saving the latest state of the program on break which can be retrieved later and resume.

**Version 1.2.1 (Mar 2, 2017)**

	- Removed dependency on ``joblib`` for multiprocessing.

**Version 1.2.0 (Jan 5, 2017)**

	- LaTeX representation of Irene's objects.
	- ``SDRelaxSol`` can be called as an iterable.
	- Default objective function is set to a ``sympy`` object for truncated moment problems.

**Version 1.1.0 (Dec 25, 2016 - Merry Christmas)**

	- Extracting minimizers via ``SDRelaxSol.ExtractSolution()`` and help of ``scipy``,
	- Extracting minimizers implementing Lasserre-Henrion algorithm,
	- Adding ``SDPRelaxations.Probability`` and ``SDPRelaxations.PSDMoment`` to give more flexibility over moments and enables rational minimization.
	- SOS decomposition implemented.
	- ``__str__`` method for ``SDPRelaxations``.
	- Using `pyOpt` as the external optimizer.
	- More benchmark examples.

**Version 1.0.0 (Dec 07, 2016)**
	
	- Initial release (Birth of Irene)