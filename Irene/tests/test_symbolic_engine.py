"""Tests for the user-selectable symbolic backend (SymEngine vs SymPy).

Covers:
  - default backend resolution (env var / auto)
  - set_backend() / get_backend() / available_backends()
  - symbol, matrix, expand object types follow the selected backend
  - SymPy-only operations (groebner, Poly, lambdify) work in both backends
  - invalid backend names raise
"""
import os
import subprocess
import sys

import pytest

from Irene.symbolic_engine import (
    SymbolicEngine,
    engine,
    get_symbolic_backend,
    set_symbolic_backend,
)

BACKENDS = engine.available_backends()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_engine():
    eng = SymbolicEngine()
    yield eng


# ---------------------------------------------------------------------------
# Backend selection API
# ---------------------------------------------------------------------------

class TestBackendSelection:
    def test_get_backend_returns_valid_name(self, fresh_engine):
        assert fresh_engine.get_backend() in ("symengine", "sympy")

    def test_available_backends_always_has_sympy(self):
        backends = SymbolicEngine.available_backends()
        assert "sympy" in backends
        assert isinstance(backends, list)

    def test_set_backend_sympy(self, fresh_engine):
        fresh_engine.set_backend("sympy")
        assert fresh_engine.get_backend() == "sympy"

    def test_set_backend_symengine(self, fresh_engine):
        if "symengine" not in fresh_engine.available_backends():
            pytest.skip("symengine not installed")
        fresh_engine.set_backend("symengine")
        assert fresh_engine.get_backend() == "symengine"

    def test_set_backend_auto(self, fresh_engine):
        fresh_engine.set_backend("auto")
        assert fresh_engine.get_backend() in ("symengine", "sympy")

    def test_set_backend_case_insensitive(self, fresh_engine):
        fresh_engine.set_backend("SYMPY")
        assert fresh_engine.get_backend() == "sympy"

    def test_set_backend_invalid_raises(self, fresh_engine):
        with pytest.raises(ValueError):
            fresh_engine.set_backend("magic")

    def test_set_backend_symengine_missing_raises(self, fresh_engine):
        if "symengine" in fresh_engine.available_backends():
            pytest.skip("symengine installed — cannot test missing case")
        with pytest.raises(ImportError):
            fresh_engine.set_backend("symengine")


# ---------------------------------------------------------------------------
# Object types follow the selected backend
# ---------------------------------------------------------------------------

class TestBackendObjectTypes:
    def test_symbol_type_sympy(self):
        eng = SymbolicEngine(use_symengine=False)
        import sympy
        assert isinstance(eng.Symbol("x"), sympy.Symbol)

    def test_symbol_type_symengine(self):
        if "symengine" not in BACKENDS:
            pytest.skip("symengine not installed")
        eng = SymbolicEngine(use_symengine=True)
        import symengine
        assert isinstance(eng.Symbol("x"), symengine.Symbol)

    def test_symbols_list(self):
        eng = SymbolicEngine(use_symengine=False)
        syms = eng.symbols("x y")
        assert isinstance(syms, list) and len(syms) == 2

    def test_matrix_type_sympy(self):
        eng = SymbolicEngine(use_symengine=False)
        M = eng.Matrix([[1, 2], [3, 4]])
        import sympy
        assert isinstance(M, sympy.Matrix)

    def test_matrix_type_symengine(self):
        if "symengine" not in BACKENDS:
            pytest.skip("symengine not installed")
        eng = SymbolicEngine(use_symengine=True)
        M = eng.Matrix([[1, 2], [3, 4]])
        import symengine
        assert isinstance(M, symengine.DenseMatrix)

    def test_expand_type_sympy(self):
        eng = SymbolicEngine(use_symengine=False)
        x = eng.Symbol("x")
        e = eng.expand((x + 1) ** 3)
        import sympy
        assert isinstance(e, sympy.Basic)

    def test_expand_type_symengine(self):
        if "symengine" not in BACKENDS:
            pytest.skip("symengine not installed")
        eng = SymbolicEngine(use_symengine=True)
        x = eng.Symbol("x")
        e = eng.expand((x + 1) ** 3)
        import symengine
        assert isinstance(e, symengine.Basic)

    def test_expand_equivalence(self):
        """Both backends produce the same expanded polynomial."""
        eng_sp = SymbolicEngine(use_symengine=False)
        eng_se = SymbolicEngine(use_symengine=True) if "symengine" in BACKENDS else eng_sp
        x_sp = eng_sp.Symbol("x")
        y_sp = eng_sp.Symbol("y")
        x_se = eng_se.Symbol("x")
        y_se = eng_se.Symbol("y")
        e_sp = str(eng_sp.expand((x_sp + y_sp) ** 4))
        e_se = str(eng_se.expand((x_se + y_se) ** 4))
        # Sort terms: SymEngine and SymPy may order terms differently
        assert sorted(e_sp.split(" + ")) == sorted(e_se.split(" + "))


# ---------------------------------------------------------------------------
# SymPy-only operations work in both backends
# ---------------------------------------------------------------------------

class TestSymPyOnlyOps:
    def test_groebner_in_sympy_backend(self):
        eng = SymbolicEngine(use_symengine=False)
        x, y = eng.symbols("x y")
        gb = eng.groebner([x**2 + y**2 - 1, x - y], x, y)
        assert gb is not None

    def test_groebner_in_symengine_backend(self):
        if "symengine" not in BACKENDS:
            pytest.skip("symengine not installed")
        eng = SymbolicEngine(use_symengine=True)
        x, y = eng.symbols("x y")
        gb = eng.groebner([x**2 + y**2 - 1, x - y], x, y)
        assert gb is not None

    def test_poly_in_both_backends(self):
        import sympy
        for use_se in (False, True):
            if use_se and "symengine" not in BACKENDS:
                continue
            eng = SymbolicEngine(use_symengine=use_se)
            x = eng.Symbol("x")
            p = eng.Poly(x**2 + 1, x)
            assert isinstance(p, sympy.Poly)

    def test_lambdify_in_both_backends(self):
        import numpy as np
        for use_se in (False, True):
            if use_se and "symengine" not in BACKENDS:
                continue
            eng = SymbolicEngine(use_symengine=use_se)
            x = eng.Symbol("x")
            f = eng.lambdify(x, x**2 + 1, "numpy")
            assert abs(f(np.array([2.0])) - 5.0) < 1e-12


# ---------------------------------------------------------------------------
# Env var integration
# ---------------------------------------------------------------------------

class TestEnvVar:
    def test_env_var_sympy(self):
        code = (
            "from Irene.symbolic_engine import engine; "
            "print(engine.get_backend())"
        )
        env = dict(os.environ, IRENE_SYMBOLIC_BACKEND="sympy")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        assert out.returncode == 0, out.stderr
        assert "sympy" in out.stdout.strip()

    def test_env_var_symengine_or_auto(self):
        code = (
            "from Irene.symbolic_engine import engine; "
            "print(engine.get_backend())"
        )
        env = dict(os.environ, IRENE_SYMBOLIC_BACKEND="auto")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() in ("symengine", "sympy")

    def test_module_level_helpers(self):
        set_symbolic_backend("sympy")
        assert get_symbolic_backend() == "sympy"
        if "symengine" in BACKENDS:
            set_symbolic_backend("symengine")
            assert get_symbolic_backend() == "symengine"
        # restore default-ish state
        set_symbolic_backend("auto")
