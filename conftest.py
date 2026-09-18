"""Pytest configuration for IreneRewrite."""
import sys
import os
import pytest
from pathlib import Path

# Ensure the project root is on sys.path so `Irene` is importable
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def ci_solver():
    """Solver name from IRENE_CI_SOLVER env var, or None if not in CI.
    
    GitHub Actions CI matrix sets IRENE_CI_SOLVER=CLARABEL or =SCS.
    When set, tests that exercise solver routing should prefer this solver
    so the matrix actually validates different backends.
    """
    return os.environ.get("IRENE_CI_SOLVER")
