=============================
Revision History
=============================

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