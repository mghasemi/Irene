"""
symbolic_engine.py — SymEngine primary path with SymPy fallback router.

Design:
  - All polynomial expansion, symbol creation, and numeric evaluation route through
    SymEngine's C++ backend for speed.
  - Groebner basis, Poly.as_dict(), reduced(), DomainMatrix/PolyMatrix, and other
    advanced SymPy-only APIs fall back transparently via to_sympy() cast.
  - The router is a context: import `engine` and call engine.expand(), engine.groebner(), etc.

Usage in existing modules (e.g., relaxations.py):
    # OLD:
    from sympy import groebner, Poly, Matrix, expand, lambdify, Symbol, QQ, zeros, reduced, sympify
    # NEW:
    from Irene.symbolic_engine import engine
    x = engine.Symbol('x')
    g = engine.groebner([f1, f2], x)       # auto-fallback to SymPy
    p = engine.Poly(expr, x)               # SymEngine when possible, SymPy fallback
    M = engine.Matrix([[x**2, 1], [0, x]]) # DenseMatrix primary
    result = engine.expand(p * q)          # SymEngine C++ expand
"""

import symengine as se
import sympy as sp
from functools import wraps


# =============================================================================
# Cast utilities — the bridge between backends
# =============================================================================

def to_sympy(obj):
    """Cast a SymEngine object (or list/matrix of them) to SymPy."""
    if obj is None:
        return None
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
    """Cast a SymPy object (or list/matrix of them) to SymEngine."""
    if obj is None:
        return None
    if isinstance(obj, sp.Basic):
        try:
            return se.sympy2symengine(obj)
        except (AttributeError, TypeError, NotImplementedError):
            # Can't convert — return as-is and let caller handle
            return obj
    if isinstance(obj, sp.Matrix):
        entries = [to_symengine(obj[i, j]) for i in range(obj.rows) for j in range(obj.cols)]
        return se.DenseMatrix(obj.rows, obj.cols, entries)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_symengine(item) for item in obj)
    return obj


# =============================================================================
# Fallback decorator — try SymEngine first, fall back to SymPy transparently
# =============================================================================

def fallback_to_sympy(func):
    """Decorator: run func with SymEngine; on failure, cast inputs→SymPy→run native→cast result."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (AttributeError, NotImplementedError, TypeError, se.LibExpressionException) as exc:
            # Cast all Basic inputs to SymPy, call the SymPy equivalent, cast result back flag
            sympy_args = [to_sympy(a) if isinstance(a, (se.Basic, se.DenseMatrix)) else a for a in args]
            sympy_kwargs = {k: (to_sympy(v) if isinstance(v, (se.Basic, se.DenseMatrix)) else v)
                           for k, v in kwargs.items()}
            # Dispatch to SymPy equivalent
            sympy_func_name = func.__name__.replace('se_', '')
            sympy_result = getattr(sp, sympy_func_name)(*sympy_args, **sympy_kwargs)
            return sympy_result  # Return as SymPy object; caller decides whether to cast back
    return wrapper


# =============================================================================
# SymbolicEngine — the unified interface
# =============================================================================

class SymbolicEngine:
    """Unified symbolic computation engine.

    Primary path: SymEngine (C++ backend).
    Fallback: SymPy for operations SymEngine doesn't support.

    Attributes:
        use_symengine  : bool — force primary path (default True)
        fallback_log   : list — records which calls fell back to SymPy
    """

    def __init__(self, use_symengine=True):
        self.use_symengine = use_symengine
        self.fallback_log = []

    # ------------------------------------------------------------------
    # Symbol creation
    # ------------------------------------------------------------------

    def Symbol(self, name, **kwargs):
        """Create a symbolic variable. Returns SymEngine symbol."""
        return se.symbols(name)

    def symbols(self, names, **kwargs):
        """Create multiple symbolic variables."""
        if isinstance(names, str) and ' ' in names:
            return list(se.symbols(names))
        if isinstance(names, str):
            # comma-separated
            return [se.sympy2symengine(sp.Symbol(n.strip())) for n in names.split(',')]
        return [self.Symbol(str(n)) for n in names]

    # ------------------------------------------------------------------
    # Polynomial operations — SymEngine primary, SymPy fallback
    # ------------------------------------------------------------------

    def expand(self, expr):
        """Expand a polynomial expression. SymEngine C++ backend."""
        try:
            if isinstance(expr, se.Basic):
                return se.expand(expr)
            elif isinstance(expr, sp.Basic):
                se_expr = to_symengine(expr)
                if isinstance(se_expr, se.Basic):
                    return se.expand(se_expr)
                # Couldn't convert — use SymPy directly
                return sp.expand(expr)
            else:
                return expr
        except (AttributeError, NotImplementedError, TypeError):
            self.fallback_log.append(('expand', type(expr).__name__))
            if isinstance(expr, se.Basic):
                expr_sp = to_sympy(expr)
                return sp.expand(expr_sp)
            elif isinstance(expr, sp.Basic):
                return sp.expand(expr)
            return expr

    def groebner(self, polys, *gens, order='lex'):
        """Groebner basis — always SymPy (SymEngine doesn't support this)."""
        # Convert all inputs to SymPy
        sp_polys = [to_sympy(p) if isinstance(p, se.Basic) else p for p in polys]
        sp_gens = [to_sympy(g) if isinstance(g, se.Basic) else g for g in gens]
        return sp.groebner(sp_polys, *sp_gens, order=order)

    def reduced(self, expr, groebner_basis):
        """Reduce expression modulo Groebner basis — always SymPy."""
        sp_expr = to_sympy(expr) if isinstance(expr, se.Basic) else expr
        sp_gb = [to_sympy(g) for g in groebner_basis]
        return sp.reduced(sp_expr, sp_gb)

    def Poly(self, expr, *gens):
        """Construct polynomial. SymPy fallback (SymEngine lacks full Poly API)."""
        sp_expr = to_sympy(expr) if isinstance(expr, se.Basic) else expr
        sp_gens = [to_sympy(g) if isinstance(g, se.Basic) else g for g in gens]
        return sp.Poly(sp_expr, *sp_gens)

    def lambdify(self, symbols, expressions, modules='numpy'):
        """Compile to numerical function — SymPy lambdify (SymEngine lacks this)."""
        if isinstance(expressions, (list, tuple)):
            sp_exprs = [to_sympy(e) if isinstance(e, se.Basic) else e for e in expressions]
        else:
            sp_exprs = to_sympy(expressions) if isinstance(expressions, se.Basic) else expressions

        # Handle both single symbol and list of symbols (matches SymPy API)
        if isinstance(symbols, (se.Basic, sp.Basic)):
            sp_syms = to_sympy(symbols) if isinstance(symbols, se.Basic) else symbols
        elif hasattr(symbols, '__iter__'):
            sp_syms = [to_sympy(s) if isinstance(s, se.Basic) else s for s in symbols]
        else:
            sp_syms = symbols
        return sp.lambdify(sp_syms, sp_exprs, modules)

    # ------------------------------------------------------------------
    # Matrix operations — SymEngine DenseMatrix primary
    # ------------------------------------------------------------------

    def Matrix(self, *args, **kwargs):
        """Construct a matrix. Uses SymEngine DenseMatrix for numeric/symbolic entries."""
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
                            sp_rows.append([to_sympy(e) if isinstance(e, se.Basic) else e for e in row])
                        else:
                            sp_rows.append([to_sympy(row) if isinstance(row, se.Basic) else row])
                    sp_args.append(sp_rows)
                elif isinstance(a, se.DenseMatrix):
                    sp_args.append(to_sympy(a))
                else:
                    sp_args.append(a)
            return sp.Matrix(*sp_args, **kwargs)
        except Exception:
            # Ultimate fallback
            sp_args = [to_sympy(a) if isinstance(a, (se.Basic, se.DenseMatrix)) else a for a in args]
            return sp.Matrix(*sp_args, **kwargs)

    def zeros(self, rows, cols):
        """Zero matrix."""
        try:
            return se.zeros(rows, cols)
        except Exception:
            return sp.zeros(rows, cols)

    # ------------------------------------------------------------------
    # Field and number types
    # ------------------------------------------------------------------

    @property
    def QQ(self):
        """Rational field — SymPy only."""
        return sp.QQ

    def sympify(self, obj):
        """Convert Python/SymEngine object to symbolic expression."""
        if isinstance(obj, se.Basic):
            return to_sympy(obj)
        return sp.sympify(obj)

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    def latex(self, expr):
        """LaTeX string representation."""
        if isinstance(expr, se.Basic):
            # SymEngine has no native latex printer; convert to SymPy for LaTeX
            sp_expr = to_sympy(expr)
            return sp.latex(sp_expr)
        elif isinstance(expr, sp.Basic):
            return sp.latex(expr)
        return str(expr)

    def sqrt(self, expr):
        """Square root."""
        if isinstance(expr, se.Basic):
            return se.sqrt(expr)
        elif isinstance(expr, sp.Basic):
            return sp.sqrt(expr)
        return float(expr)**0.5

    def Abs(self, expr):
        """Absolute value."""
        if isinstance(expr, se.Basic):
            return se.Abs(expr)
        return sp.Abs(expr)

    def Function(self, name):
        """Symbolic function — SymPy only (SymEngine lacks this)."""
        return sp.Function(name)

    # ------------------------------------------------------------------
    # PolyMatrix / DomainMatrix — SymPy only
    # ------------------------------------------------------------------

    def PolyMatrix(self, matrix, *gens):
        """Polynomial matrix — SymPy only."""
        from sympy.polys.polymatrix import PolyMatrix as SP_PolyMatrix
        if isinstance(matrix, se.DenseMatrix):
            matrix = to_sympy(matrix)
        sp_gens = [to_sympy(g) if isinstance(g, se.Basic) else g for g in gens]
        return SP_PolyMatrix(matrix, *sp_gens)

    def DomainMatrix(self, matrix, domain):
        """Domain matrix — SymPy only."""
        from sympy.polys.matrices import DomainMatrix as SP_DomainMatrix
        if isinstance(matrix, se.DenseMatrix):
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


# =============================================================================
# Default engine instance — import and use directly
# =============================================================================

engine = SymbolicEngine(use_symengine=True)
