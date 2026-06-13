r"""
SOS+SONC Relaxation Framework
==============================
Implements the SOS+SONC two-step optimization algorithms from
Moritz Schick's PhD thesis (*Sums of squares plus sums of
nonnegative circuit polynomials*, Universität Konstanz),
translated from MATLAB to Python and integrated into the Irene
framework.

Classes
-------
SOSONCRelaxations
    Combined SOS+SONC lower-bound computation for unconstrained
    polynomial optimization.
SOSONCRelaxSol
    Container for relaxation results (optimal value, certificates,
    timing, status).
"""

import math
import time
from typing import Any, Optional

from .program import OptimizationProblem


# ──────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────


class SOSONCRelaxSol(object):
    """Carries optimisation results for SOS+SONC relaxations.

    Attributes
    ----------
    val : float
        Optimal :math:`\\lambda^*`, a lower bound on :math:`f^*`.
    method : str
        ``'sos'``, ``'sonc'``, ``'sos-first'``, or ``'sonc-first'``.
    f_sos : optional
        SOS summand of ``f - val`` (if applicable).
    f_sonc : optional
        SONC summand of ``f - val`` (if applicable).
    status : str
        ``'optimal'``, ``'infeasible'``, or ``'error'``.
    error_code : int
        0 = success, 1 = infeasible, 2 = solver error.
    runtime : float
        Wall-clock time in seconds.
    message : str
        Human-readable status message.
    """

    __slots__ = (
        "val",
        "method",
        "f_sos",
        "f_sonc",
        "status",
        "error_code",
        "runtime",
        "message",
    )

    def __init__(self) -> None:
        self.val: float = -float("inf")
        self.method: str = ""
        self.f_sos: Any = None
        self.f_sonc: Any = None
        self.status: str = "error"
        self.error_code: int = 2
        self.runtime: float = 0.0
        self.message: str = ""

    def __repr__(self) -> str:
        return (
            f"SOSONCRelaxSol(val={self.val}, method='{self.method}', "
            f"status='{self.status}', runtime={self.runtime:.4f}s)"
        )


# ──────────────────────────────────────────────────────────────
# SOS+SONC relaxation engine
# ──────────────────────────────────────────────────────────────


class SOSONCRelaxations(object):
    r"""Combined SOS+SONC relaxation framework.

    Implements the algorithms from Schick's SOS+SONC toolbox:

    - ``globalMinSOS``   — pure SOS relaxation (SDP via Gram matrix)
    - ``globalMinSONC``  — pure SONC relaxation (signomial GP)
    - ``globalMinSOSPSONC`` — two-step SOS+SONC (Algorithms 4 & 5)

    Parameters
    ----------
    prog : OptimizationProblem
        An Irene ``OptimizationProblem`` with an objective set via
        ``prog.Minimize(f)``.  Constraints are optional.
    error_bound : float
        Numerical zero tolerance (default ``1e-10``).
    verbosity : int
        Log level (0 = silent, 1 = verbose; default 1).
    solver : str
        SDP solver name forwarded to ``SDPRelaxations``
        (``'cvxopt'``, ``'csdp'``, ``'sdpa'``, ``'dsdp'``).
    use_local_solve : bool
        Use signomial GP local solve for the SONC portion
        (default ``True``).
    relaxation_order : int
        Relaxation order for SDP (default ``1``, i.e., the first
        Lasserre moment relaxation).
    """

    _SDP_ERROR_KEYWORDS = (
        "infeasib",
        "feasibility",
    )

    def __init__(
        self,
        prog: OptimizationProblem,
        **kwargs,
    ) -> None:
        self.prog = prog
        self.error_bound = kwargs.get("error_bound", 1e-10)
        self.verbosity = kwargs.get("verbosity", 1)
        self.solver = kwargs.get("solver", "cvxopt")
        self.use_local_solve = kwargs.get("use_local_solve", True)
        self.relaxation_order = kwargs.get("relaxation_order", 1)

    # ── Utility ──────────────────────────────────────────

    def _is_sdp_infeasible(self, status: Optional[str], message: str) -> bool:
        """Heuristic check for SDP infeasibility."""
        if status is not None and "infeas" in str(status).lower():
            return True
        for kw in self._SDP_ERROR_KEYWORDS:
            if kw in str(message).lower():
                return True
        if "-inf" in str(message) or "Infeasible" in str(message):
            return True
        return False

    def _wrap_sos_result(self, sos_sol, runtime: float) -> SOSONCRelaxSol:
        """Build an SOSONCRelaxSol from an SDPRelaxations result."""
        out = SOSONCRelaxSol()
        out.method = "sos"
        out.runtime = runtime

        # Attempt to read the primal value
        try:
            out.val = float(sos_sol.Primal)
        except (TypeError, ValueError, AttributeError):
            out.val = -float("inf")

        # Check for infeasibility
        if self._is_sdp_infeasible(
            getattr(sos_sol, "Status", ""),
            getattr(sos_sol, "Message", ""),
        ):
            out.status = "infeasible"
            out.error_code = 1
            out.val = -float("inf")
            out.message = "SOS relaxation infeasible"
            return out

        out.status = "optimal"
        out.error_code = 0
        out.message = "SOS relaxation solved"

        # Store the SOS certificate polynomial if available
        try:
            out.f_sos = getattr(sos_sol, "f_sos", None)
        except Exception:
            out.f_sos = None

        return out

    def _solve_sdp(self, problem: OptimizationProblem):
        """Solve Irene SDP relaxation for an ``OptimizationProblem``.

        The canonical pipeline in Irene is:
        ``from_problem`` -> ``MomentsOrd`` -> ``SetSDPSolver`` ->
        ``InitSDP`` -> ``Minimize``.
        """
        from .relaxations import SDPRelaxations

        sdp_relax = SDPRelaxations.from_problem(problem)
        sdp_relax.MomentsOrd(int(self.relaxation_order))
        sdp_relax.SetSDPSolver(str(self.solver))
        sdp_relax.InitSDP()
        sdp_relax.Minimize()
        return sdp_relax.Solution

    def _wrap_sonc_result(self, sonc_val: float, runtime: float) -> SOSONCRelaxSol:
        """Build an SOSONCRelaxSol from a SONC relaxation value."""
        out = SOSONCRelaxSol()
        out.method = "sonc"
        out.runtime = runtime
        out.val = float(sonc_val)
        out.status = "optimal" if not math.isinf(sonc_val) else "infeasible"
        out.error_code = 0 if not math.isinf(sonc_val) else 1
        out.message = "SONC relaxation solved"
        out.f_sonc = None  # could extract from SONCRelaxations.solution
        return out

    # ── Algorithm 1: Pure SOS ─────────────────────────────

    def globalMinSOS(self) -> SOSONCRelaxSol:
        """Compute a lower bound using the SOS relaxation.

        Solves :math:`\\sup\\{\\lambda : f - \\lambda \\in \\Sigma\\}`
        via a Gram-matrix SDP using Irene's ``SDPRelaxations``.

        Returns
        -------
        SOSONCRelaxSol
        """
        t0 = time.time()
        try:
            sos_sol = self._solve_sdp(self.prog)
        except Exception as exc:
            out = SOSONCRelaxSol()
            out.method = "sos"
            out.runtime = time.time() - t0
            out.status = "error"
            out.error_code = 2
            out.message = str(exc)[:200]
            return out

        runtime = time.time() - t0
        if sos_sol is None:
            out = SOSONCRelaxSol()
            out.method = "sos"
            out.runtime = runtime
            out.status = "error"
            out.error_code = 2
            out.message = "SOS relaxation returned no solution object"
            return out

        return self._wrap_sos_result(sos_sol, runtime)

    # ── Algorithm 2: Pure SONC ────────────────────────────

    def globalMinSONC(self) -> SOSONCRelaxSol:
        """Compute a lower bound using the SONC relaxation.

        Solves :math:`\\sup\\{\\lambda : f - \\lambda \\in C\\}`
        via a signomial geometric program using Irene's
        ``SONCRelaxations``.

        Returns
        -------
        SOSONCRelaxSol
        """
        from .sonc import SONCRelaxations

        t0 = time.time()
        try:
            sonc_relax = SONCRelaxations(
                self.prog,
                error_bound=self.error_bound,
                verbosity=max(0, self.verbosity - 1),
                use_local_solve=self.use_local_solve,
            )
            sonc_val = sonc_relax.solve(verbosity=self.verbosity)
        except Exception as exc:
            out = SOSONCRelaxSol()
            out.method = "sonc"
            out.runtime = time.time() - t0
            out.status = "error"
            out.error_code = 2
            out.message = str(exc)[:200]
            return out

        runtime = time.time() - t0
        return self._wrap_sonc_result(sonc_val, runtime)

    # ── Preprocessing helpers ─────────────────────────────

    @staticmethod
    def _coefficient_distance(
        coeffs_a: dict,
        coeffs_b: dict,
    ) -> float:
        """L2 distance between coefficient dictionaries."""
        all_keys = set(coeffs_a) | set(coeffs_b)
        squared_sum = 0.0
        for k in all_keys:
            diff = coeffs_a.get(k, 0.0) - coeffs_b.get(k, 0.0)
            squared_sum += diff * diff
        return math.sqrt(squared_sum)

    def _extract_coefficients(self, element) -> dict:
        """Extract coefficient dict from a SemigroupAlgebraElement.

        Keys are exponent tuples, values are float coefficients.
        """
        coeffs: dict = {}
        gen_names = [g.name for g in self.prog.semigroup.generators]
        for coeff, mono in element.content:
            key = self.prog.mono2ord_tuple(mono)
            coeffs[key] = float(coeff)
        return coeffs

    # ── Algorithm 3: Two-step SOS+SONC ────────────────────

    def globalMinSOSPSONC(
        self,
        first: str = "sos",
    ) -> SOSONCRelaxSol:
        r"""Two-step SOS+SONC lower bound (Algorithms 4 & 5).

        Parameters
        ----------
        first : str
            ``'sos'`` (Algorithm 4: SOS preprocess, then SONC)
            or ``'sonc'`` (Algorithm 5: SONC preprocess, then SOS).

        Returns
        -------
        SOSONCRelaxSol
            Lower bound :math:`f_{\\Sigma + C}^*` together with
            decomposed certificate.
        """
        if first not in ("sos", "sonc"):
            raise ValueError("first must be 'sos' or 'sonc'")

        t0 = time.time()

        if first == "sos":
            out = self._two_step_sos_first()
        else:
            out = self._two_step_sonc_first()

        out.runtime = time.time() - t0
        return out

    def _two_step_sos_first(self) -> SOSONCRelaxSol:
        """Algorithm 4: SOS preprocessing → SONC relaxation.

        1. Solve SOS → λ_sos with certificate g* = f - λ_sos ∈ Σ.
        2. Build residual h = f - λ_sos (constant shift of the
           objective) and solve SONC on h → μ*.
        3. Combined bound: λ_sos + μ*.
           Certificate: f - (λ_sos + μ*) = (h - μ*) + (λ_sos + μ*).
        """
        out = SOSONCRelaxSol()
        out.method = "sos-first"

        # Step 1: SOS
        sos_result = self.globalMinSOS()
        if sos_result.error_code != 0:
            sonc_result = self.globalMinSONC()
            sonc_result.method = "sos-first"
            return sonc_result

        lambda_sos = sos_result.val
        if self.verbosity > 0:
            print(f"[SOS+SONC] SOS preprocess: λ_sos = {lambda_sos}")

        # Step 2: Build residual h = f - λ_sos
        try:
            f_obj = self.prog.objective
            # Copy the SGA element and subtract λ_sos from the constant term
            h_coeffs = [(c, m) for c, m in f_obj.content]
            identity = self.prog.semigroup.identity()
            found = False
            new_content = []
            for c, m in h_coeffs:
                if m == identity:
                    new_content.append((float(c) - lambda_sos, m))
                    found = True
                else:
                    new_content.append((float(c), m))
            if not found:
                new_content.append((-lambda_sos, identity))

            # Build residual OptimizationProblem
            from .grouprings import SemigroupAlgebraElement
            h_element = SemigroupAlgebraElement(self.prog.sga, new_content)
            prog_residual = OptimizationProblem(self.prog.sga)
            prog_residual.set_objective(h_element)
            # Carry over any constraints
            if self.prog.constraints:
                prog_residual.add_constraints(self.prog.constraints)

            # Step 3: SONC on residual
            from .sonc import SONCRelaxations
            sonc_relax = SONCRelaxations(
                prog_residual,
                error_bound=self.error_bound,
                verbosity=max(0, self.verbosity - 1),
                use_local_solve=self.use_local_solve,
            )
            mu_star = sonc_relax.solve(verbosity=self.verbosity)

            out.val = lambda_sos + mu_star
            out.status = "optimal"
            out.error_code = 0
            out.message = (
                f"SOS-first: λ_sos={lambda_sos:.6f}, μ*={mu_star:.6f}, "
                f"combined={out.val:.6f}"
            )
        except Exception:
            # Fallback: max of individual bounds
            sonc_result = self.globalMinSONC()
            if sonc_result.error_code == 0:
                out.val = max(lambda_sos, sonc_result.val)
                out.status = "optimal"
                out.error_code = 0
                out.message = (
                    f"SOS-first (fallback — max): λ_sos={lambda_sos:.6f}, "
                    f"λ_sonc={sonc_result.val:.6f}"
                )
            else:
                out.val = lambda_sos
                out.status = "optimal"
                out.error_code = 0
                out.message = (
                    "SOS-first residual SONC failed; returning SOS bound "
                    f"λ_sos={lambda_sos:.6f}"
                )

        out.f_sos = sos_result.f_sos
        return out

    def _two_step_sonc_first(self) -> SOSONCRelaxSol:
        """Algorithm 5: SONC preprocessing → SOS relaxation.

        1. Solve SONC → λ_sonc with certificate g* = f - λ_sonc ∈ C.
        2. Build residual h = f - λ_sonc and solve SOS on h → μ*.
        3. Combined bound: λ_sonc + μ*.
        """
        out = SOSONCRelaxSol()
        out.method = "sonc-first"

        # Step 1: SONC
        sonc_result = self.globalMinSONC()
        if sonc_result.error_code != 0:
            sos_result = self.globalMinSOS()
            sos_result.method = "sonc-first"
            return sos_result

        lambda_sonc = sonc_result.val
        if self.verbosity > 0:
            print(f"[SOS+SONC] SONC preprocess: λ_sonc = {lambda_sonc}")

        # Step 2: Build residual h = f - λ_sonc
        try:
            f_obj = self.prog.objective
            identity = self.prog.semigroup.identity()
            new_content = []
            found = False
            for c, m in f_obj.content:
                if m == identity:
                    new_content.append((float(c) - lambda_sonc, m))
                    found = True
                else:
                    new_content.append((float(c), m))
            if not found:
                new_content.append((-lambda_sonc, identity))

            from .grouprings import SemigroupAlgebraElement
            h_element = SemigroupAlgebraElement(self.prog.sga, new_content)
            prog_residual = OptimizationProblem(self.prog.sga)
            prog_residual.set_objective(h_element)
            if self.prog.constraints:
                prog_residual.add_constraints(self.prog.constraints)

            # Step 3: SOS on residual
            sos_residual = self._solve_sdp(prog_residual)
            try:
                mu_star = float(sos_residual.Primal)
            except (TypeError, ValueError, AttributeError):
                mu_star = -float("inf")

            if not math.isinf(mu_star) and not self._is_sdp_infeasible(
                getattr(sos_residual, "Status", ""),
                getattr(sos_residual, "Message", ""),
            ):
                out.val = lambda_sonc + mu_star
                out.status = "optimal"
                out.error_code = 0
                out.message = (
                    f"SONC-first: λ_sonc={lambda_sonc:.6f}, μ*={mu_star:.6f}, "
                    f"combined={out.val:.6f}"
                )
            else:
                out.val = lambda_sonc
                out.status = "optimal"
                out.error_code = 0
                out.message = (
                    "SONC-first residual SOS infeasible; returning SONC bound "
                    f"λ_sonc={lambda_sonc:.6f}"
                )
        except Exception:
            sos_result = self.globalMinSOS()
            if sos_result.error_code == 0:
                out.val = max(lambda_sonc, sos_result.val)
                out.status = "optimal"
                out.error_code = 0
                out.message = (
                    f"SONC-first (fallback — max): λ_sonc={lambda_sonc:.6f}, "
                    f"λ_sos={sos_result.val:.6f}"
                )
            else:
                out.val = lambda_sonc
                out.status = "optimal"
                out.error_code = 0
                out.message = (
                    "SONC-first residual SOS failed; returning SONC bound "
                    f"λ_sonc={lambda_sonc:.6f}"
                )

        out.f_sonc = sonc_result.f_sonc
        return out


# ──────────────────────────────────────────────────────────────
# Module-level convenience
# ──────────────────────────────────────────────────────────────


def sosonc_bounds(
    prog: OptimizationProblem,
    **kwargs,
) -> dict[str, float]:
    """Compute SOS, SONC, and SOS+SONC lower bounds.

    Returns a dict with keys ``'sos'``, ``'sonc'``,
    ``'sos_first'``, ``'sonc_first'``.
    """
    engine = SOSONCRelaxations(prog, **kwargs)
    results: dict[str, float] = {}

    for method, func in [
        ("sos", engine.globalMinSOS),
        ("sonc", engine.globalMinSONC),
        ("sos_first", lambda: engine.globalMinSOSPSONC("sos")),
        ("sonc_first", lambda: engine.globalMinSOSPSONC("sonc")),
    ]:
        try:
            sol = func()
            results[method] = sol.val
        except Exception:
            results[method] = -float("inf")

    return results
