"""Unified Relaxation API
========================

A single entry point for all relaxation methods (SOS, SONC, SOS+SONC).

Design goals
------------
1. **One constructor** — ``RelaxationEngine(prog)`` wraps any ``OptimizationProblem``.
2. **Consistent return type** — every solve call returns a ``RelaxResult`` with the
   same attributes (value, status, timing, solver metadata).
3. **Method dispatch** — the user picks ``'sos'``, ``'sonc'``, or ``'sosonc'``;
   the engine routes to the correct backend class internally.
4. **Backward compatibility** — the old classes (``SDPRelaxations``, etc.) still work;
   this module is additive, not destructive.

Usage
-----
>>> from Irene.program import OptimizationProblem
>>> from Irene.relaxation_api import RelaxationEngine, RelaxResult
>>> engine = RelaxationEngine(prog)
>>> result: RelaxResult = engine.solve(method='sos', order=2, solver='cvxopt')
>>> print(result.value)          # lower bound
>>> print(result.status)         # 'optimal' | 'infeasible' | 'error'

For a full comparison across all methods in one call:

>>> results = engine.compare(order=1)
>>> for m, r in results.items():
...     print(f"{m}: {r.value:.6f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from .program import OptimizationProblem


# ──────────────────────────────────────────────────────────────
# Method enumeration and result type
# ──────────────────────────────────────────────────────────────


class RelaxMethod(str, Enum):
    """Supported relaxation backends."""

    SOS = "sos"
    SONC = "sonc"
    SOSPSONC_SOS_FIRST = "sosonc_sos_first"
    SOSPSONC_SONC_FIRST = "sosonc_sonc_first"


# Alias for convenience — users can pass strings directly
RelaxMethodStr = Literal["sos", "sonc", "sosonc_sos_first", "sosonc_sonc_first"]


@dataclass
class RelaxResult:
    """Uniform result container for all relaxation methods.

    Attributes
    ----------
    value : float
        Lower bound on the global minimum (``-inf`` on failure).
    method : str
        Which relaxation was used.
    status : str
        ``'optimal'``, ``'infeasible'``, or ``'error'``.
    error_code : int
        0 = success, 1 = infeasible, 2 = solver/computation error.
    runtime : float
        Wall-clock seconds for the solve phase (excludes matrix setup).
    init_time : Optional[float]
        Time spent building moment/localizing matrices (if available).
    message : str
        Human-readable status or error description.
    certificate : Optional[Any]
        SOS/SONC polynomial certificate if the backend exposes one.
    solver_info : dict
        Backend-specific metadata (solver name, iterations, etc.).
    """

    value: float = -float("inf")
    method: str = ""
    status: str = "error"
    error_code: int = 2
    runtime: float = 0.0
    init_time: Optional[float] = None
    message: str = ""
    certificate: Any = None
    solver_info: dict = field(default_factory=dict)

    # ── convenience ────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"RelaxResult(value={self.value:.6f}, method='{self.method}', "
            f"status='{self.status}', runtime={self.runtime:.3f}s)"
        )

    @property
    def success(self) -> bool:
        """True if the relaxation returned a finite bound without error."""
        return self.error_code == 0 and self.value > -float("inf")


# ──────────────────────────────────────────────────────────────
# Unified engine
# ──────────────────────────────────────────────────────────────


class RelaxationEngine:
    """Single entry point for SOS, SONC, and combined relaxations.

    Parameters
    ----------
    prog : OptimizationProblem
        An Irene optimization problem with an objective set via
        ``prog.Minimize(f)`` or ``prog.set_objective(f)``.
    order : int
        Relaxation/hierarchy order (default 1).
    solver : str
        SDP solver name forwarded to the SOS backend
        (``'cvxopt'``, ``'csdp'``, ``'sdpa'``, ``'dsdp'``).
    error_bound : float
        Numerical zero tolerance for SONC GP.
    verbosity : int
        Log level: 0 = silent, 1 = normal, 2+ = verbose.
    use_local_solve : bool
        Use signomial GP local solver for the SONC portion.

    Examples
    --------
    >>> engine = RelaxationEngine(prog, order=2, solver='cvxopt')
    >>> res = engine.solve('sos')
    >>> print(res.value)

    Compare all methods at once:

    >>> results = engine.compare()
    >>> best_method = max(results, key=lambda m: results[m].value)
    """

    def __init__(
        self,
        prog: OptimizationProblem,
        order: int = 1,
        solver: str = "cvxopt",
        error_bound: float = 1e-10,
        verbosity: int = 1,
        use_local_solve: bool = True,
    ) -> None:
        self.prog = prog
        self.order = order
        self.solver = solver
        self.error_bound = error_bound
        self.verbosity = verbosity
        self.use_local_solve = use_local_solve

    # ── public API ────────────────────────────────────────

    def solve(
        self,
        method: RelaxMethodStr | RelaxMethod = "sos",
        order: Optional[int] = None,
        solver: Optional[str] = None,
    ) -> RelaxResult:
        """Run a single relaxation and return a ``RelaxResult``.

        Parameters
        ----------
        method : str or RelaxMethod
            Which relaxation to use.  Accepted values::

                'sos'                  — pure SOS (Gram-matrix SDP)
                'sonc'                 — pure SONC (signomial GP)
                'sosonc_sos_first'     — two-step: SOS → SONC residual
                'sosonc_sonc_first'    — two-step: SONC → SOS residual

        order : int, optional
            Override the hierarchy order for this call.
        solver : str, optional
            Override the SDP solver name for this call.

        Returns
        -------
        RelaxResult
        """
        method_str = method.value if isinstance(method, RelaxMethod) else method

        dispatch = {
            "sos": self._solve_sos,
            "sonc": self._solve_sonc,
            "sosonc_sos_first": self._solve_sosonc_sos_first,
            "sosonc_sonc_first": self._solve_sosonc_sonc_first,
        }

        fn = dispatch.get(method_str)
        if fn is None:
            raise ValueError(
                f"Unknown method '{method_str}'. "
                f"Choose from {{'sos', 'sonc', 'sosonc_sos_first', 'sosonc_sonc_first'}}."
            )

        return fn(order=order, solver=solver)

    def compare(
        self,
        order: Optional[int] = None,
        solver: Optional[str] = None,
    ) -> dict[str, RelaxResult]:
        """Run all four relaxation variants and return results keyed by method.

        Parameters are the same as ``solve()``.  Returns a dict mapping
        method name → ``RelaxResult``.
        """
        methods: list[RelaxMethodStr] = [
            "sos",
            "sonc",
            "sosonc_sos_first",
            "sosonc_sonc_first",
        ]
        return {m: self.solve(m, order=order, solver=solver) for m in methods}

    # ── internal dispatchers ──────────────────────────────

    def _solve_sos(
        self, *, order: Optional[int] = None, solver: Optional[str] = None
    ) -> RelaxResult:
        """Route to SDPRelaxations backend."""
        from .relaxations import SDPRelaxations

        result = RelaxResult(method="sos")
        t0 = time.time()

        try:
            sdp_relax = SDPRelaxations.from_problem(self.prog)
            sdp_relax.MomentsOrd(order if order is not None else self.order)
            sdp_relax.SetSDPSolver(solver or self.solver)
            sdp_relax.InitSDP()
            result.init_time = time.time() - t0

            sdp_relax.Minimize()
            sol = sdp_relax.Solution

            if sol is None:
                raise RuntimeError("SDPRelaxations returned no solution object")

            primal_val = getattr(sol, "Primal", None)
            if primal_val is None:
                raise RuntimeError("SDP solution has no Primal value")
            result.value = float(primal_val)
            result.status = "optimal"
            result.error_code = 0
            result.message = f"SOS relaxation solved (solver={solver or self.solver})"
            result.certificate = getattr(sol, "f_sos", None)
            result.solver_info = {
                "solver": solver or self.solver,
                "order": order or self.order,
                "status_str": str(getattr(sol, "Status", "")),
            }

            # Check for infeasibility keywords
            status_msg = str(getattr(sol, "Message", "")) + str(
                getattr(sol, "Status", "")
            )
            if any(kw in status_msg.lower() for kw in ("infeasib", "-inf")):
                result.status = "infeasible"
                result.error_code = 1
                result.value = -float("inf")
                result.message = "SOS relaxation declared infeasible"

        except Exception as exc:
            result.status = "error"
            result.error_code = 2
            result.message = str(exc)[:300]

        result.runtime = time.time() - t0
        return result

    def _solve_sonc(
        self, *, order: Optional[int] = None, solver: Optional[str] = None
    ) -> RelaxResult:
        """Route to SONCRelaxations backend."""
        from .sonc import SONCRelaxations

        result = RelaxResult(method="sonc")
        t0 = time.time()

        try:
            sonc_relax = SONCRelaxations(
                self.prog,
                error_bound=self.error_bound,
                verbosity=max(0, self.verbosity - 1),
                use_local_solve=self.use_local_solve,
            )
            val = sonc_relax.solve(verbosity=self.verbosity)

            import math

            if math.isinf(val):
                result.status = "infeasible"
                result.error_code = 1
                result.value = -float("inf")
                result.message = "SONC relaxation returned infinite bound"
            else:
                result.value = float(val)
                result.status = "optimal"
                result.error_code = 0
                result.message = "SONC relaxation solved"

        except Exception as exc:
            result.status = "error"
            result.error_code = 2
            result.message = str(exc)[:300]

        result.runtime = time.time() - t0
        return result

    def _solve_sosonc_sos_first(
        self, *, order: Optional[int] = None, solver: Optional[str] = None
    ) -> RelaxResult:
        """Two-step: SOS preprocess → SONC on residual."""
        from .sosonc import SOSONCRelaxations

        result = RelaxResult(method="sosonc_sos_first")
        t0 = time.time()

        try:
            engine = SOSONCRelaxations(
                self.prog,
                error_bound=self.error_bound,
                verbosity=self.verbosity,
                solver=solver or self.solver,
                use_local_solve=self.use_local_solve,
                relaxation_order=order if order is not None else self.order,
            )
            sol = engine.globalMinSOSPSONC(first="sos")

            result.value = float(sol.val)
            result.status = "optimal" if sol.error_code == 0 else "infeasible"
            result.error_code = sol.error_code
            result.message = sol.message
            result.certificate = {
                "f_sos": sol.f_sos,
                "f_sonc": sol.f_sonc,
            }

        except Exception as exc:
            result.status = "error"
            result.error_code = 2
            result.message = str(exc)[:300]

        result.runtime = time.time() - t0
        return result

    def _solve_sosonc_sonc_first(
        self, *, order: Optional[int] = None, solver: Optional[str] = None
    ) -> RelaxResult:
        """Two-step: SONC preprocess → SOS on residual."""
        from .sosonc import SOSONCRelaxations

        result = RelaxResult(method="sosonc_sonc_first")
        t0 = time.time()

        try:
            engine = SOSONCRelaxations(
                self.prog,
                error_bound=self.error_bound,
                verbosity=self.verbosity,
                solver=solver or self.solver,
                use_local_solve=self.use_local_solve,
                relaxation_order=order if order is not None else self.order,
            )
            sol = engine.globalMinSOSPSONC(first="sonc")

            result.value = float(sol.val)
            result.status = "optimal" if sol.error_code == 0 else "infeasible"
            result.error_code = sol.error_code
            result.message = sol.message
            result.certificate = {
                "f_sos": sol.f_sos,
                "f_sonc": sol.f_sonc,
            }

        except Exception as exc:
            result.status = "error"
            result.error_code = 2
            result.message = str(exc)[:300]

        result.runtime = time.time() - t0
        return result


# ──────────────────────────────────────────────────────────────
# Module-level convenience function (mirrors sosonc.sosonc_bounds)
# ──────────────────────────────────────────────────────────────


def relax(
    prog: OptimizationProblem,
    method: RelaxMethodStr | RelaxMethod = "sos",
    **kwargs: Any,
) -> RelaxResult:
    """Quick one-liner for solving a relaxation.

    Parameters
    ----------
    prog : OptimizationProblem
        The optimization problem to relax.
    method : str or RelaxMethod
        Which relaxation backend to use.
    **kwargs
        Passed through to ``RelaxationEngine`` constructor
        (``order``, ``solver``, ``error_bound``, ``verbosity``, etc.).

    Returns
    -------
    RelaxResult

    Examples
    --------
    >>> from Irene.relaxation_api import relax
    >>> res = relax(prog, method='sos', order=2)
    >>> print(res.value)
    """
    engine = RelaxationEngine(prog, **kwargs)
    return engine.solve(method)


def compare_all(
    prog: OptimizationProblem,
    **kwargs: Any,
) -> dict[str, RelaxResult]:
    """Run all four relaxation variants and return results keyed by method.

    Convenience wrapper around ``RelaxationEngine.compare()``.
    """
    engine = RelaxationEngine(prog, **kwargs)
    return engine.compare()
