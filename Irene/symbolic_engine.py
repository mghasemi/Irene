"""
symbolic_engine.py -- user-selectable symbolic backend (SymEngine or SymPy).

Design:
  - The default backend is SymEngine (C++ backend) for expansion, matrix and
    symbol creation; operations SymEngine does not implement (Groebner basis,
    Poly.as_dict(), reduced(), lambdify, DomainMatrix/PolyMatrix, ...) always
    run on SymPy, with transparent to_sympy() casting.
  - Users can select the backend at runtime:

        from Irene.symbolic_engine import engine, set_symbolic_backend
        set_symbolic_backend('sympy')      # or 'symengine' / 'auto'

    or via the environment variable IRENE_SYMBOLIC_BACKEND:

        IRENE_SYMBOLIC_BACKEND=sympy python3 my_script.py
        IRENE_SYMBOLIC_BACKEND=symengine python3 my_script.py

    Accepted values: 'symengine' (default when installed), 'sympy', 'auto'
    (prefer SymEngine when available, else SymPy).
  - If SymEngine is not installed, the engine automatically falls back to
    SymPy and `engine.backend` reports 'sympy'.

Usage in existing modules (e.g., relaxations.py):
    # OLD:
    from sympy import groebner, Poly, Matrix, expand, lambdify, Symbol, QQ, zeros, reduced, sympify
    # NEW:
    from Irene.symbolic_engine import engine
    x = engine.Symbol('x')
    g = engine.groebner([f1, f2], x)       # always SymPy (SymEngine lacks Groebner)
    p = engine.Poly(expr, x)               # always SymPy (SymEngine lacks full Poly API)
    M = engine.Matrix([[x**2, 1], [0, x]]) # DenseMatrix (SymEngine) or sp.Matrix (SymPy)
    result = engine.expand(p * q)          # backend-native expand
"""

import os
from functools import wraps
from typing import Any

try:
    import symengine as _se_mod  # type: ignore[import-not-found]
    _HAS_SYMENGINE = True
except ImportError:  # pragma: no cover - exercised only without symengine installed
    _se_mod = None  # type: ignore[assignment]
    _HAS_SYMENGINE = False

se: Any = _se_mod

import sympy as sp


# =============================================================================
# Backend resolution helpers
# =============================================================================

_VALID_BACKENDS = ("auto", "symengine", "sympy")


def _env_default_backend() -> bool:
    """Resolve the default `use_symengine` flag from IRENE_SYMBOLIC_BACKEND."""
    raw = os.environ.get("IRENE_SYMBOLIC_BACKEND", "auto").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        raw = "symengine"
    elif raw in ("0", "false", "no", "off"):
        raw = "sympy"
    if raw == "symengine":
        return _HAS_SYMENGINE  # requested but unavailable -> silently SymPy
    if raw == "sympy":
        return False
    # 'auto' or unknown value: prefer SymEngine when installed
    return _HAS_SYMENGINE


# =============================================================================
# Cast utilities -- the bridge between backends
# =============================================================================

def to_sympy(obj):
    """Cast a SymEngine object (or list/matrix of them) to SymPy."""
    if obj is None:
        return None
    # Short-circuit: already a SymPy object -- skip expensive _sympy_() tree conversion
    if isinstance(obj, sp.Basic) and not (_HAS_SYMENGINE and isinstance(obj, se.Basic)):
        return obj
    if _HAS_SYMENGINE:
        if isinstance(obj, se.Basic):
            try:
                return obj._sympy_()
            except AttributeError:
                # Fallback: stringify and re-parse (lossy but safe)
                return sp.sympify(str(obj))
        if isinstance(obj, se.DenseMatrix):
            return sp.Matrix(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_sympy(item) for item in obj)
    # Already SymPy or plain Python
    return obj


def to_symengine(obj):
    """Cast a SymPy object (or list/matrix of them) to SymEngine.

    Returns the object unchanged when SymEngine is unavailable or the object
    cannot be converted.
    """
    if obj is None or not _HAS_SYMENGINE:
        return obj
    if isinstance(obj, sp.Basic):
        try:
            return se.sympy2symengine(obj)
        except (AttributeError, TypeError, NotImplementedError):
            # Can't convert -- return as-is and let caller handle
            return obj
    if isinstance(obj, sp.Matrix):
        entries = [to_symengine(obj[i, j]) for i in range(obj.rows) for j in range(obj.cols)]
        return se.DenseMatrix(obj.rows, obj.cols, entries)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_symengine(item) for item in obj)
    return obj


# =============================================================================
# Fallback decorator -- try SymEngine first, fall back to SymPy transparently
# =============================================================================

def fallback_to_sympy(func):
    """Decorator: run func with SymEngine; on failure, cast inputs->SymPy->run native->cast result."""
    _symengine_errors = (
        (AttributeError, NotImplementedError, TypeError)
        + ((se.LibExpressionException,) if _HAS_SYMENGINE else ())
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except _symengine_errors:
            # Cast all Basic inputs to SymPy, call the SymPy equivalent, cast result back flag
            sympy_args = [to_sympy(a) if _HAS_SYMENGINE and isinstance(a, (se.Basic, se.DenseMatrix)) else a
                          for a in args]
            sympy_kwargs = {k: (to_sympy(v) if _HAS_SYMENGINE and isinstance(v, (se.Basic, se.DenseMatrix)) else v)
                            for k, v in kwargs.items()}
            # Dispatch to SymPy equivalent
            sympy_func_name = func.__name__.replace('se_', '')
            sympy_result = getattr(sp, sympy_func_name)(*sympy_args, **sympy_kwargs)
            return sympy_result  # Return as SymPy object; caller decides whether to cast back
    return wrapper


# =============================================================================
# SymbolicEngine -- the unified interface with selectable backend
# =============================================================================

class SymbolicEngine:
    """Unified symbolic computation engine.

    Primary path: SymEngine (C++ backend) when `use_symengine` is True.
    Fallback: SymPy for operations SymEngine doesn't support.

    The backend can be selected at construction, at runtime via
    :meth:`set_backend`, or globally via the ``IRENE_SYMBOLIC_BACKEND``
    environment variable (read once at import time for the default engine).

    Attributes:
        use_symengine  : bool -- True uses SymEngine primary path (default True)
        fallback_log   : list -- records which calls fell back to SymPy
    """

    def __init__(self, use_symengine=None):
        if use_symengine is None:
            use_symengine = _env_default_backend()
        self.use_symengine = bool(use_symengine)
        self.fallback_log = []

    # ------------------------------------------------------------------
    # Backend selection API
    # ------------------------------------------------------------------

    def set_backend(self, backend):
        """Select the symbolic backend.

        Args:
            backend: 'symengine', 'sympy', or 'auto' (prefer installed).

        Returns:
            self (chainable).

        Raises:
            ValueError: unknown backend name.
            ImportError: 'symengine' requested but not installed.
        """
        val = str(backend).strip().lower()
        if val not in _VALID_BACKENDS:
            raise ValueError(
                f"Unknown symbolic backend {backend!r}. Choose from {_VALID_BACKENDS}.")
        if val == "sympy":
            self.use_symengine = False
        elif val == "symengine":
            if not _HAS_SYMENGINE:
                raise ImportError(
                    "SymEngine backend requested but 'symengine' is not installed. "
                    "Install it with `pip install symengine` or select 'sympy'.")
            self.use_symengine = True
        else:  # auto
            self.use_symengine = _HAS_SYMENGINE
        return self

    def get_backend(self):
        """Current backend name: 'symengine' or 'sympy'."""
        return "symengine" if self.use_symengine else "sympy"

    @staticmethod
    def available_backends():
        """Backends that can be selected on this installation."""
        backends = ["sympy"]
        if _HAS_SYMENGINE:
            backends.append("symengine")
        return backends

    # ------------------------------------------------------------------
    # Symbol creation
    # ------------------------------------------------------------------

    def Symbol(self, name, **kwargs):
        """Create a symbolic variable in the selected backend."""
        if self.use_symengine:
            return se.symbols(name)
        return sp.Symbol(name, **kwargs)

    def symbols(self, names, **kwargs):
        """Create multiple symbolic variables in the selected backend."""
        if isinstance(names, str) and ' ' in names:
            if self.use_symengine:
                return list(se.symbols(names))
            return list(sp.symbols(names))
        if isinstance(names, str):
            # comma-separated
            parts = [n.strip() for n in names.split(',')]
            if self.use_symengine:
                return [se.sympy2symengine(sp.Symbol(n)) for n in parts]
            return [sp.Symbol(n) for n in parts]
        return [self.Symbol(str(n)) for n in names]

    # ------------------------------------------------------------------
    # Polynomial operations -- backend-native where possible
    # ------------------------------------------------------------------

    def expand(self, expr):
        """Expand a polynomial expression using the selected backend."""
        try:
            if self.use_symengine and _HAS_SYMENGINE:
                if isinstance(expr, se.Basic):
                    return se.expand(expr)
                if isinstance(expr, sp.Basic):
                    se_expr = to_symengine(expr)
                    if isinstance(se_expr, se.Basic):
                        return se.expand(se_expr)
                    # Couldn't convert -- use SymPy directly
                    return sp.expand(expr)
                return expr
            # SymPy backend (or SymEngine unavailable)
            if isinstance(expr, se.Basic) if _HAS_SYMENGINE else False:
                expr = to_sympy(expr)
            return sp.expand(expr)
        except (AttributeError, NotImplementedError, TypeError):
            self.fallback_log.append(('expand', type(expr).__name__))
            if _HAS_SYMENGINE and isinstance(expr, se.Basic):
                expr_sp = to_sympy(expr)
                return sp.expand(expr_sp)
            elif isinstance(expr, sp.Basic):
                return sp.expand(expr)
            return expr

    def groebner(self, polys, *gens, order='lex'):
        """Groebner basis -- always SymPy (SymEngine doesn't support this)."""
        # Convert all inputs to SymPy
        sp_polys = [to_sympy(p) if _HAS_SYMENGINE and isinstance(p, se.Basic) else p for p in polys]
        sp_gens = [to_sympy(g) if _HAS_SYMENGINE and isinstance(g, se.Basic) else g for g in gens]
        return sp.groebner(sp_polys, *sp_gens, order=order)

    def reduced(self, expr, groebner_basis):
        """Reduce expression modulo Groebner basis -- always SymPy."""
        sp_expr = to_sympy(expr) if _HAS_SYMENGINE and isinstance(expr, se.Basic) else expr
        sp_gb = [to_sympy(g) for g in groebner_basis]
        return sp.reduced(sp_expr, sp_gb)

    def Poly(self, expr, *gens):
        """Construct polynomial. SymPy fallback (SymEngine lacks full Poly API)."""
        sp_expr = to_sympy(expr) if _HAS_SYMENGINE and isinstance(expr, se.Basic) else expr
        sp_gens = [to_sympy(g) if _HAS_SYMENGINE and isinstance(g, se.Basic) else g for g in gens]
        return sp.Poly(sp_expr, *sp_gens)

    def lambdify(self, symbols, expressions, modules='numpy'):
        """Compile to numerical function -- SymPy lambdify (SymEngine lacks this)."""
        if isinstance(expressions, (list, tuple)):
            sp_exprs = [to_sympy(e) if _HAS_SYMENGINE and isinstance(e, se.Basic) else e
                        for e in expressions]
        else:
            sp_exprs = to_sympy(expressions) if _HAS_SYMENGINE and isinstance(expressions, se.Basic) else expressions

        # Handle both single symbol and list of symbols (matches SymPy API)
        if isinstance(symbols, (sp.Basic,)) or (_HAS_SYMENGINE and isinstance(symbols, se.Basic)):
            sp_syms = to_sympy(symbols) if _HAS_SYMENGINE and isinstance(symbols, se.Basic) else symbols
        elif hasattr(symbols, '__iter__'):
            sp_syms = [to_sympy(s) if _HAS_SYMENGINE and isinstance(s, se.Basic) else s for s in symbols]
        else:
            sp_syms = symbols
        return sp.lambdify(sp_syms, sp_exprs, modules)

    # ------------------------------------------------------------------
    # Matrix operations -- backend-native
    # ------------------------------------------------------------------

    def Matrix(self, *args, **kwargs):
        """Construct a matrix in the selected backend.

        SymEngine DenseMatrix when the backend is SymEngine and entries are
        compatible; SymPy Matrix otherwise.
        """
        if self.use_symengine and _HAS_SYMENGINE:
            try:
                if len(args) == 1 and isinstance(args[0], list):
                    data = args[0]
                    # Check if entries are SymEngine-compatible
                    flat = [item for row in data for item in (row if isinstance(row, list) else [row])]
                    if all(isinstance(e, (int, float, se.Basic)) for e in flat):
                        rows = len(data)
                        cols = len(data[0]) if data else 0
                        entries_flat = []
                        for row in data:
                            for entry in row:
                                if isinstance(entry, sp.Basic) and not isinstance(entry, se.Basic):
                                    entries_flat.append(to_symengine(entry))
                                else:
                                    entries_flat.append(entry)
                        return se.DenseMatrix(rows, cols, entries_flat)
                # Fallback to SymPy Matrix for complex constructions
                sp_args = []
                for a in args:
                    if isinstance(a, list):
                        sp_rows = []
                        for row in a:
                            if isinstance(row, list):
                                sp_rows.append([to_sympy(e) if _HAS_SYMENGINE and isinstance(e, se.Basic) else e
                                                for e in row])
                            else:
                                sp_rows.append([to_sympy(row) if _HAS_SYMENGINE and isinstance(row, se.Basic) else row])
                        sp_args.append(sp_rows)
                    elif isinstance(a, se.DenseMatrix):
                        sp_args.append(to_sympy(a))
                    else:
                        sp_args.append(a)
                return sp.Matrix(*sp_args, **kwargs)
            except Exception:
                # Ultimate fallback
                sp_args = [to_sympy(a) if _HAS_SYMENGINE and isinstance(a, (se.Basic, se.DenseMatrix)) else a
                           for a in args]
                return sp.Matrix(*sp_args, **kwargs)
        # SymPy backend
        sp_args = []
        for a in args:
            if isinstance(a, list):
                sp_rows = []
                for row in a:
                    if isinstance(row, list):
                        sp_rows.append([to_sympy(e) if _HAS_SYMENGINE and isinstance(e, se.Basic) else e
                                        for e in row])
                    else:
                        sp_rows.append([to_sympy(row) if _HAS_SYMENGINE and isinstance(row, se.Basic) else row])
                sp_args.append(sp_rows)
            elif _HAS_SYMENGINE and isinstance(a, se.DenseMatrix):
                sp_args.append(to_sympy(a))
            else:
                sp_args.append(a)
        return sp.Matrix(*sp_args, **kwargs)

    def zeros(self, rows, cols):
        """Zero matrix in the selected backend."""
        if self.use_symengine and _HAS_SYMENGINE:
            try:
                return se.zeros(rows, cols)
            except Exception:
                return sp.zeros(rows, cols)
        return sp.zeros(rows, cols)

    # ------------------------------------------------------------------
    # Field and number types
    # ------------------------------------------------------------------

    @property
    def QQ(self):
        """Rational field -- SymPy only."""
        return sp.QQ

    def sympify(self, obj):
        """Convert Python/SymEngine object to symbolic expression."""
        if _HAS_SYMENGINE and isinstance(obj, se.Basic):
            return to_sympy(obj)
        return sp.sympify(obj)

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    def latex(self, expr):
        """LaTeX string representation (SymPy printer for both backends)."""
        if _HAS_SYMENGINE and isinstance(expr, se.Basic):
            # SymEngine has no native latex printer; convert to SymPy for LaTeX
            sp_expr = to_sympy(expr)
            return sp.latex(sp_expr)
        elif isinstance(expr, sp.Basic):
            return sp.latex(expr)
        return str(expr)

    def sqrt(self, expr):
        """Square root in the selected backend."""
        if self.use_symengine and _HAS_SYMENGINE and isinstance(expr, se.Basic):
            return se.sqrt(expr)
        elif isinstance(expr, sp.Basic):
            return sp.sqrt(expr)
        return float(expr) ** 0.5

    def Abs(self, expr):
        """Absolute value in the selected backend."""
        if self.use_symengine and _HAS_SYMENGINE and isinstance(expr, se.Basic):
            return se.Abs(expr)
        return sp.Abs(expr)

    def Function(self, name):
        """Symbolic function -- SymPy only (SymEngine lacks this)."""
        return sp.Function(name)

    # ------------------------------------------------------------------
    # PolyMatrix / DomainMatrix -- SymPy only
    # ------------------------------------------------------------------

    def PolyMatrix(self, matrix, *gens):
        """Polynomial matrix -- SymPy only."""
        from sympy.polys.polymatrix import PolyMatrix as SP_PolyMatrix
        if _HAS_SYMENGINE and isinstance(matrix, se.DenseMatrix):
            matrix = to_sympy(matrix)
        sp_gens = [to_sympy(g) if _HAS_SYMENGINE and isinstance(g, se.Basic) else g for g in gens]
        return SP_PolyMatrix(matrix, *sp_gens)

    def DomainMatrix(self, matrix, domain):
        """Domain matrix -- SymPy only."""
        from sympy.polys.matrices import DomainMatrix as SP_DomainMatrix
        if _HAS_SYMENGINE and isinstance(matrix, se.DenseMatrix):
            matrix = to_sympy(matrix)
        # Ensure we have a SymPy Matrix before passing to from_Matrix
        if not isinstance(matrix, sp.Matrix):
            matrix = sp.Matrix(matrix)
        return SP_DomainMatrix.from_Matrix(matrix, domain=domain)

    # ------------------------------------------------------------------
    # Relational types (from sympy.core.relational)
    # ------------------------------------------------------------------

    @property
    def Equality(self):
        return sp.Equality

    @property
    def GreaterThan(self):
        return sp.GreaterThan

    @property
    def LessThan(self):
        return sp.LessThan

    @property
    def StrictGreaterThan(self):
        return sp.StrictGreaterThan

    @property
    def StrictLessThan(self):
        return sp.StrictLessThan

    # ------------------------------------------------------------------
    # Error types
    # ------------------------------------------------------------------

    @property
    def PolynomialError(self):
        from sympy.polys.polyerrors import PolynomialError
        return PolynomialError

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def fallback_stats(self):
        """Return dict of how many times each operation fell back to SymPy."""
        from collections import Counter
        if not self.fallback_log:
            return {}
        counter = Counter(op for op, _ in self.fallback_log)
        return dict(counter)

    def clear_fallback_log(self):
        """Clear the fallback log."""
        self.fallback_log.clear()

    def __repr__(self):
        return f"<SymbolicEngine backend={self.get_backend()} symengine_installed={_HAS_SYMENGINE}>"


# =============================================================================
# Default engine instance -- import and use directly
# =============================================================================
# Backend selection order:
#   1. IRENE_SYMBOLIC_BACKEND env var (read once at import time)
#   2. 'auto' default: SymEngine when installed, otherwise SymPy

engine = SymbolicEngine(use_symengine=None)


def set_symbolic_backend(backend):
    """Module-level helper: select the backend of the default engine."""
    return engine.set_backend(backend)


def get_symbolic_backend():
    """Module-level helper: name of the default engine's backend."""
    return engine.get_backend()
