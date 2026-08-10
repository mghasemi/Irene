"""Verification tests for phase-2 risk mitigations (R1-R3)."""

import inspect

import numpy as np
import pytest
from sympy import Symbol

from Irene.cvxpy_solver import CvxpySDPSolver, available_solvers
from Irene.relaxations import SDPRelaxations


x = Symbol("x")
y = Symbol("y")


def _base_relaxation():
    rlx = SDPRelaxations([x, y])
    rlx.SetObjective(x**2 + y**2)
    rlx.AddConstraint(1 - x**2 - y**2 >= 0)
    return rlx


def test_moment_stability_flags_well_and_ill_conditioned_cases():
    rlx = _base_relaxation()

    stable_mat = np.eye(5) * 10.0 + np.ones((5, 5))
    stable = rlx._check_moment_stability(stable_mat)
    assert stable["warning"] is False

    ill_cond = np.diag([1.0, 1e-14, 1e-15])
    ill = rlx._check_moment_stability(ill_cond)
    assert ill["warning"] is True


def test_safe_cholesky_handles_near_psd_input():
    rlx = _base_relaxation()
    near_psd = np.array([[1.0, 0.9], [0.9, 0.8]])

    chol = rlx._safe_cholesky(near_psd)
    assert chol.shape == near_psd.shape


def test_safe_cholesky_succeeds_on_psd_matrix():
    rlx = _base_relaxation()
    psd = np.array([[2.0, 1.0], [1.0, 2.0]])

    chol = rlx._safe_cholesky(psd)
    assert chol.shape == psd.shape


def test_scs_solver_fallback_codepath_mentions_clarabel():
    cvx = CvxpySDPSolver(solver="SCS")
    src = inspect.getsource(cvx.solve)

    assert "CLARABEL" in src
    assert "solvers_to_try" in src


def test_cvxpy_and_native_cvxopt_give_close_bounds_when_available():
    solvers = set(available_solvers())
    if "CLARABEL" not in solvers or "CVXOPT" not in solvers:
        pytest.skip("Requires both CLARABEL and CVXOPT to compare parity")

    from Irene.sdp import sdp

    rlx_cvxpy = _base_relaxation()
    rlx_cvxpy.SDP = sdp(solver="clarabel")
    rlx_cvxpy.MomentsOrd(2)
    rlx_cvxpy.InitSDP()
    lb_cvxpy = float(rlx_cvxpy.Minimize())

    rlx_native = _base_relaxation()
    rlx_native.SetSDPSolver("cvxopt")
    rlx_native.MomentsOrd(2)
    rlx_native.InitSDP()
    lb_native = float(rlx_native.Minimize())

    denom = max(abs(lb_native), 1.0)
    rel_diff = abs(lb_cvxpy - lb_native) / denom
    assert rel_diff < 1e-3
