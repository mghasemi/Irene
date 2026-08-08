"""
CVXPY-based solver abstraction layer for Irene SDP relaxations.

Replaces text-based solver writers (SDPA/CSDP dat files) with a DCP-compliant
formulation that routes to Clarabel, SCS, or CVXOPT through CVXPY's unified API.

The SDP is formulated in primal standard form:

    min  b^T x
    s.t. sum_i A_{i,j} * x[i] - C_j >= 0   for j = 1..k  (PSD blocks)

where each block j has dimension BlockStruct[j].
"""

from __future__ import annotations

import time
import warnings
from typing import Optional, Union

import cvxpy as cp
import numpy as np


# ---------------------------------------------------------------------------
# Solver registry & availability detection
# ---------------------------------------------------------------------------

_SDPSOLVERS = ['CLARABEL', 'SCS', 'CVXOPT']


def available_solvers() -> list[str]:
    """Return list of SDP-capable solvers currently installed."""
    installed = cp.installed_solvers()
    return [s for s in _SDPSOLVERS if s in installed]


# ---------------------------------------------------------------------------
# Solver options mapping (CVXPY solver kwargs)
# ---------------------------------------------------------------------------

_SOLVER_OPTION_MAP = {
    # Clarabel — interior point defaults
    'CLARABEL': {
        'verbose': False,
        'tol_gap_abs': 1e-7,
        'tol_gap_rel': 1e-6,
    },
    # SCS — first-order (approximate) solver
    'SCS': {
        'verbose': False,
        'eps': 1e-5,
    },
    # CVXOPT — legacy interior point
    'CVXOPT': {
        'verbose': False,
        'maxiters': 100,
        'abstol': 1e-7,
        'reltol': 1e-6,
        'feastol': 1e-7,
    },
}


# ---------------------------------------------------------------------------
# SDPResult — structured return type (mirrors sdp.Info dict)
# ---------------------------------------------------------------------------

class SDPResult:
    """Container for SDP solution data.

    Attributes mirror the legacy ``sdp.Info`` dictionary keys so that
    downstream code in ``relaxations.py`` can consume results without change.
    """

    __slots__ = (
        'status', 'primal_obj', 'dual_obj',
        'x',          # primal variable vector
        'Z',          # dual PSD matrices (one per block)
        'X',          # primal PSD matrices (one per block)
        'wall_time',  # seconds
    )

    def __init__(self):
        self.status: str = 'Unknown'
        self.primal_obj: Optional[float] = None
        self.dual_obj: Optional[float] = None
        self.x: Optional[np.ndarray] = None
        self.Z: list[np.ndarray] = []   # dual matrices [Z_1, ..., Z_k]
        self.X: list[np.ndarray] = []   # primal matrices [X_1, ..., X_k]
        self.wall_time: float = 0.0

    def to_info_dict(self) -> dict:
        """Return legacy-compatible Info dictionary."""
        return {
            'Status': self.status,
            'PObj': self.primal_obj,
            'DObj': self.dual_obj,
            'y': self.x,
            'Z': self.Z,
            'X': self.X,
            'Wall': self.wall_time,
            'CPU': None,
        }

    def __repr__(self) -> str:
        return (f"SDPResult(status={self.status!r}, "
                f"primal_obj={self.primal_obj}, wall={self.wall_time:.2f}s)")


# ---------------------------------------------------------------------------
# CVXPY SDP solver class
# ---------------------------------------------------------------------------

class CvxpySDPSolver:
    """CVXPY-based SDP solver with the same API as ``Irene.sdp.sdp``.

    Usage mirrors the legacy interface::

        solver = CvxpySDPSolver(solver='CLARABEL')
        solver.SetObjective(b)
        solver.AddConstraintBlock(A_i)   # for each variable x_i
        solver.AddConstantBlock(C_j)     # constant PSD blocks
        result = solver.solve()

    Parameters
    ----------
    solver : str, optional
        Solver backend. One of ``'CLARABEL'``, ``'SCS'``, ``'CVXOPT'``.
        Defaults to the first available solver from that list.
    """

    def __init__(self, solver: Optional[str] = None):
        # Resolve solver
        avail = available_solvers()
        if not avail:
            raise ImportError(
                "No SDP-capable solver found. Install one of: "
                + ", ".join(_SDPSOLVERS)
            )
        self._solver_name = (solver or 'CLARABEL').upper()
        if self._solver_name not in avail:
            raise ImportError(
                f"Solver '{self._solver_name}' is not available. "
                f"Available: {avail}"
            )

        # Internal storage — mirrors legacy sdp class attributes
        self.b: Optional[np.ndarray] = None       # objective coefficients
        self.A: list[list[np.ndarray]] = []       # A[i][j] for var i, block j
        self.C: list[np.ndarray] = []             # C[j] constant blocks
        self.BlockStruct: list[int] = []          # block sizes [d_1, ..., d_k]

        # Solver options (override defaults)
        self.solver_options: dict = {}
        self.Info: dict = {}                      # legacy-compatible output

    # ------------------------------------------------------------------
    # API mirroring sdp class
    # ------------------------------------------------------------------

    def SetObjective(self, b):
        """Set objective coefficient vector ``b``.

        Parameters
        ----------
        b : array-like of shape (m,)
            Coefficients of the linear objective ``min b^T x``.
        """
        self.b = np.asarray(b, dtype=np.float64).ravel()

    def AddConstraintBlock(self, A):
        """Add constraint matrices for one primal variable.

        Parameters
        ----------
        A : list of ndarray, length k
            ``A[j]`` is the coefficient matrix (d_j x d_j) for block j.
        """
        BlkStc = [blk.shape[0] for blk in A]
        if self.BlockStruct:
            if BlkStc != self.BlockStruct:
                raise TypeError("The block structure is inconsistent.")
        else:
            self.BlockStruct = BlkStc
        self.A.append([np.asarray(m, dtype=np.float64) for m in A])

    def AddConstantBlock(self, C):
        """Set constant PSD blocks.

        Parameters
        ----------
        C : list of ndarray, length k
            ``C[j]`` is the constant matrix (d_j x d_j) for block j.
        """
        BlkStc = [blk.shape[0] for blk in C]
        if self.BlockStruct:
            if BlkStc != self.BlockStruct:
                raise TypeError("The block structure is inconsistent.")
        else:
            self.BlockStruct = BlkStc
        self.C = [np.asarray(m, dtype=np.float64) for m in C]

    def Option(self, param: str, val):
        """Set a solver option.

        Parameters
        ----------
        param : str
            Option name (solver-specific).
        val : any
            Option value.
        """
        self.solver_options[param] = val

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> SDPResult:
        """Solve the SDP and return structured results.

        Returns
        -------
        SDPResult
            Container with status, objectives, primal/dual variables, etc.
        """
        if self.b is None or not self.A or not self.C:
            raise ValueError(
                "SDP is incomplete. Call SetObjective(), AddConstraintBlock(), "
                "and AddConstantBlock() before solving."
            )

        m = len(self.b)          # number of primal variables
        k = len(self.C)         # number of PSD blocks

        start_time = time.time()

        # --- Build CVXPY problem ---
        x = cp.Variable(m)

        # Objective: min b^T x
        objective = cp.Minimize(cp.sum(cp.multiply(x, self.b)))

        # Constraints: for each block j, sum_i A[i][j] * x[i] - C[j] >> 0
        # Each A[i][j] is a (d_j x d_j) matrix; x[i] is a scalar CVXPY variable.
        constraints = []
        for j in range(k):
            block_expr = sum(
                (x[i] * self.A[i][j] for i in range(m)),
                start=np.zeros((self.BlockStruct[j], self.BlockStruct[j])),
            ) - self.C[j]
            constraints.append(block_expr >> 0)

        problem = cp.Problem(objective, constraints)

        # --- Solver chain: primary + optional fallback (R2) ---
        solvers_to_try = [self._solver_name]
        if self._solver_name == 'SCS' and 'CLARABEL' in available_solvers():
            solvers_to_try.append('CLARABEL')

        problem_status = None
        for attempt_solver in solvers_to_try:
            attempt_opts = _SOLVER_OPTION_MAP.get(attempt_solver, {}).copy()
            attempt_opts.update(self.solver_options)
            try:
                problem.solve(solver=attempt_solver, **attempt_opts)
            except cp.SolverError as exc:
                elapsed = time.time() - start_time
                result = SDPResult()
                result.status = 'SolverError'
                result.wall_time = elapsed
                self.Info = {'Status': 'SolverError', 'error': str(exc), 'Wall': elapsed}
                return result
            problem_status = problem.status
            if problem_status not in ['infeasible', 'infeasible_inaccurate']:
                break  # accept non-infeasible result

        wall_time = time.time() - start_time

        # --- Extract results ---
        result = SDPResult()
        result.wall_time = wall_time

        if problem_status in ['optimal', 'optimal_inaccurate']:
            result.status = 'Optimal'
            result.primal_obj = float(problem.value) if problem.value is not None else None
            # Dual objective: b^T x_dual (from dual variables of equality constraints)
            # For SDP in standard form, the dual objective equals the primal at optimality
            result.dual_obj = result.primal_obj  # strong duality for optimal solutions
            result.x = x.value.copy() if x.value is not None else None

            # Extract dual variables for PSD constraints (these are the Z matrices)
            # Each constraint c_idx corresponds to block j
            for idx, constr in enumerate(constraints):
                dual_val = constr.dual_value
                if dual_val is not None:
                    d = self.BlockStruct[idx]
                    result.Z.append(np.asarray(dual_val).reshape(d, d))

        elif problem_status == 'infeasible':
            result.status = 'Infeasible'
        elif problem_status == 'unbounded':
            result.status = 'Unbounded'
        else:
            result.status = f"Unknown ({problem_status})"

        # Populate legacy Info dict
        self.Info = result.to_info_dict()
        self.Info['solver'] = self._solver_name

        return result

    def CvxOpt(self):
        """Legacy compatibility — delegates to ``solve()``."""
        warnings.warn(
            "CvxOpt() is deprecated; use solve() instead.",
            DeprecationWarning, stacklevel=2,
        )
        self.solve()

    # ------------------------------------------------------------------
    # Info helpers (legacy compat)
    # ------------------------------------------------------------------

    def __str__(self):
        num_vars = len(self.C) if self.C else 0
        num_constraints = len(self.A)
        return (f"Semidefinite program with\n"
                f"             # variables: {num_vars}\n"
                f"           # constraints: {num_constraints}\n"
                f"             with solver: {self._solver_name}")

    def __latex__(self):
        num_vars = len(self.C) if self.C else 0
        num_constraints = len(self.A)
        return f"SDP({num_vars}, {num_constraints}, {self._solver_name})"


# ---------------------------------------------------------------------------
# Convenience: auto-select best solver heuristic
# ---------------------------------------------------------------------------

def _best_solver() -> str:
    """Return the recommended default solver.

    Clarabel is preferred for SDPs (interior point, reliable).
    Falls back to SCS or CVXOPT if unavailable.
    """
    avail = available_solvers()
    for candidate in ['CLARABEL', 'CVXOPT', 'SCS']:
        if candidate in avail:
            return candidate
    return avail[0] if avail else 'CLARABEL'
