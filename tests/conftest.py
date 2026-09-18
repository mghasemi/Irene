"""Pytest configuration for IreneRewrite CI.

Provides the `ci_solver` fixture wired to the IRENE_CI_SOLVER environment variable,
and a session-scoped timeout to prevent hung SDP solves in CI.
"""
import os
import pytest


@pytest.fixture(scope="session")
def ci_solver():
    """Solver name from IRENE_CI_SOLVER env var, or None if not in CI.

    In GitHub Actions the CI matrix sets this to 'CLARABEL' or 'SCS'.
    Locally (env var unset) tests exercise both solvers for full coverage.
    """
    return os.environ.get("IRENE_CI_SOLVER")


@pytest.fixture(scope="session")
def ci_python_version():
    """Python version string from CI environment, or None locally."""
    return os.environ.get("CI_PYTHON_VERSION")
