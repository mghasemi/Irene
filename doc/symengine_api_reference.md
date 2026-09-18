# SymEngine Python API Reference — installed version 0.14.1

Extracted at runtime from the `symengine` package in the IreneRewrite venv
(`/home/mehdi/Code/Python/IreneRewrite/.venv`, Python 3.11). Every symbol
listed below was verified present via `hasattr` on the live module.
This is the authoritative API surface for the Phase 1 SymPy->SymEngine
refactor of IreneRewrite (grouprings.py, relaxations.py, matrices.py,
sdp.py, sonc.py).

## Build capabilities (probed)

- `have_flint = True`
- `have_llvm = True`
- `have_mpfr = True`
- `have_mpc = True`
- `have_numpy = True`
- `have_piranha = False`

## CRITICAL: API differences vs SymPy (0.14.1 does NOT have these)

The following SymPy names are **absent** at the top level of `symengine`
0.14.1 — code that assumes them will raise `AttributeError`:

| Missing name | SymPy equivalent | Workaround in 0.14.1 |
|---|---|---|
| `Poly` | `sympy.Poly` | Use SymPy `Poly` (Irene `engine.Poly` routes to SymPy) |
| `free_symbols` (module fn) | `sympy.free_symbols` | `expr.free_symbols` (method on Basic) |
| `degree` | `sympy.degree` | Not available; use SymPy or `Poly` |
| `factor` | `sympy.factor` | Not available; use SymPy |
| `integrate` | `sympy.integrate` | Not available; use SymPy |
| `solve` | `sympy.solve` | `linsolve` (linear only) is available |
| `sympy2symengine` (top-level) | — | Import from `symengine.lib.symengine_wrapper` (deprecated path) or string round-trip `se.sympify(str(expr))` |
| `symengine2sympy` (top-level) | — | `expr._sympy_()` (method on every Basic) |
| `simplify`, `trigsimp`, `cancel`, `together`, `collect`, `horner` | SymPy | Not available; use SymPy |
| `gcd`, `lcm`, `rem`, `quo`, `roots`, `factor_list`, `nsimplify` | SymPy | Not available; use SymPy |
| `RealFloat` | — | Use `se.Float(v)` or `se.RealDouble(v)` |
| `Sum`, `Product`, `Integral` | SymPy | Not available; use SymPy |
| `Complexes` | SymPy | Not available |
| `LibExpressionException` | — | Not importable in 0.14.1 (guard with try/except) |

## Verified conversion paths (SymPy <-> SymEngine)

```python
import symengine as se, sympy as sp
x = se.symbols('x')
f = se.expand((x + 1)**2)

# SymEngine -> SymPy: method on every Basic
sp_expr = f._sympy_()                    # (x + 1)**2 -> sympy Add
sp_args = f.args_as_sympy()             # [x + 1, 3] as SymPy objects

# SymPy -> SymEngine: NO top-level se.sympy2symengine in 0.14.1!
# Option A (deprecated but works):
from symengine.lib.symengine_wrapper import sympy2symengine
se_expr = sympy2symengine(sp_expr)
# Option B (string round-trip, lossy for non-polynomials):
se_expr = se.sympify(str(sp_expr))
```

NOTE: `Irene/symbolic_engine.py::to_symengine` calls `se.sympy2symengine`
inside a try/except AttributeError and silently returns the SymPy object
unchanged on this installation — i.e., the SymEngine fast path in
`Irene/relaxations.py` (P5.6 localized moment matrix) is silently inert.
Also `se.RealFloat` is absent, so the float branch of that path raises
TypeError and falls through to SymPy as well.

## Module-level functions (verified present)

- `symengine.expand((x, deep=True))` — 
- `symengine.diff((expr, *args))` — 
- `symengine.series((ex, x=None, x0=0, n=6, as_deg_coef_pair=False))` — 
- `symengine.lambdify((args, exprs, **kwargs))` — 
- `symengine.sympify((a))` — Converts an expression 'a' into a SymEngine type.
- `symengine.symbols((names, **args))` — Transform strings into instances of :class:`Symbol` class.
- `symengine.var((names, **args))` — Create symbols and inject them into the global namespace.
- `symengine.latex((expr))` — 
- `symengine.ccode((expr))` — 
- `symengine.unicode((expr))` — 
- `symengine.init_printing((pretty_print=True, use_latex=True))` — 
- `symengine.count_ops((*exprs))` — 
- `symengine.has_symbol((obj, symbol=None))` — 
- `symengine.cse((exprs))` — 
- `symengine.linsolve((eqs, syms))` — Solve a set of linear equations given as an iterable `eqs`
- `symengine.sqrt((x))` — 
- `symengine.Abs((x))` — 
- `symengine.zeros((r, c=None))` — 
- `symengine.ones((r, c=None))` — 
- `symengine.eye((n))` — 
- `symengine.diag((*values))` — 
- `symengine.symarray((prefix, shape, **kwargs))` — Creates an nd-array of symbols
- `symengine.sin((x))` — 
- `symengine.cos((x))` — 
- `symengine.tan((x))` — 
- `symengine.cot((x))` — 
- `symengine.sec((x))` — 
- `symengine.csc((x))` — 
- `symengine.acot((x))` — 
- `symengine.asec((x))` — 
- `symengine.asin((x))` — 
- `symengine.acos((x))` — 
- `symengine.atan((x))` — 
- `symengine.atan2((x, y))` — 
- `symengine.atanh((x))` — 
- `symengine.asinh((x))` — 
- `symengine.acosh((x))` — 
- `symengine.sinh((x))` — 
- `symengine.cosh((x))` — 
- `symengine.tanh((x))` — 
- `symengine.coth((x))` — 
- `symengine.csch((x))` — 
- `symengine.sech((x))` — 
- `symengine.acoth((x))` — 
- `symengine.asech((x))` — 
- `symengine.exp((x))` — 
- `symengine.log((x, y=None))` — 
- `symengine.gamma((x))` — 
- `symengine.digamma((x))` — 
- `symengine.trigamma((x))` — 
- `symengine.polygamma((x, y))` — 
- `symengine.loggamma((x))` — 
- `symengine.beta((x, y))` — 
- `symengine.uppergamma((x, y))` — 
- `symengine.lowergamma((x, y))` — 
- `symengine.zeta((s, a=None))` — 
- `symengine.erf((x))` — 
- `symengine.erfc((x))` — 
- `symengine.floor((x))` — 
- `symengine.ceiling((x))` — 
- `symengine.sign((x))` — 
- `symengine.conjugate((x))` — 
- `symengine.sqrt_mod((a, p, all_roots=False))` — 
- `symengine.integer_nthroot((a, n))` — 
- `symengine.isprime((n, reps=25))` — 
- `symengine.perfect_power((n))` — 
- `symengine.add((*args, **kwargs))` — 

## Class hierarchy (verified)

```
Basic
├── Expr
│   ├── Number
│   │   ├── Integer
│   │   ├── Rational
│   │   ├── Float (RealDouble, RealMPFR)
│   │   └── ComplexDouble / ComplexMPC
│   ├── Symbol / Dummy
│   ├── Add / Mul / Pow
│   ├── FunctionSymbol / Function / UndefFunction / AppliedUndef
│   ├── Derivative
│   └── Piecewise / Subs / UnevaluatedExpr
├── MatrixBase -> DenseMatrix / ImmutableMatrix / MutableDenseMatrix
├── Boolean: Eq, Ne, Lt, Le, Gt, Ge, And, Or, Not, Nand, Nor, Xnor, Xor, Contains
└── Set: Interval, FiniteSet, EmptySet, UniversalSet, Integers, Rationals, Reals
```

## Expr methods (verified on a live expression)

On `f = x**2*y + 3*x*y**3` (type `Add`):

```
args
args_as_sage
args_as_sympy
as_coefficients_dict
as_numer_denom
as_powers_dict
as_real_imag
atoms
coeff
copy
diff
evalf
expand
free_symbols
func
has
identity
is_Add
is_AlgebraicNumber
is_Atom
is_Boolean
is_Derivative
is_Dummy
is_Equality
is_Float
is_Function
is_Integer
is_Matrix
is_Mul
is_Not
is_Number
is_Pow
is_Rational
is_Relational
is_Symbol
is_finite
is_integer
is_negative
is_nonnegative
is_nonpositive
is_number
is_positive
is_real
is_symbol
is_zero
make_args
msubs
n
replace
simplify
subs
subs_dict
subs_oldnew
xreplace
```

Key patterns for IreneRewrite:

```python
f.args                    # (3*x*y**3, x**2*y) — term children
f.free_symbols            # {x, y} — a set of Symbols (METHOD, not module fn)
f.has(x)                  # True if x appears
f.func                    # class of the top node (Add, Mul, Pow, Symbol, ...)
f.msubs({x: 1, y: 2})     # simultaneous substitution (-> 26 for x**2*y + 3*x*y**3)
f.subs(x, 1)              # single substitution
f.evalf(50)               # MPFR evaluation at 50-bit precision
f.diff(x)                 # partial derivative (also se.diff(f, x))
f.simplify()              # method exists on Expr (heuristic)
f.replace(x, x**2)        # structural replace
f.xreplace({x: x**2})     # structural replace (dict form)
f.as_powers_dict()        # {base: exponent} — for Pow-typed nodes
f.as_coefficients_dict()  # {term: coefficient} — for Add-typed nodes
f.as_numer_denom()        # (numerator, denominator)
f.as_real_imag()          # (real part, imag part)
f.count_ops()             # operation count (also se.count_ops(f))
f._sympy_()               # -> SymPy object
f.args_as_sympy()         # args as SymPy objects
```

## Getting monomial exponents (the key Phase-1 question)

`symengine` 0.14.1 has **no `Poly` class**. To get monomial exponent
tuples, either (a) route through SymPy `Poly` (what `engine.Poly` does),
or (b) use Expr methods on expanded expressions:

```python
import symengine as se, sympy as sp
x, y = se.symbols('x y')
f = se.expand(x**2*y + 3*x*y**3)

# Path A: SymPy Poly (robust, canonical generator order)
p = sp.Poly(f._sympy_(), x._sympy_(), y._sympy_())
p.as_dict()      # {(2,1): 1, (1,3): 3}
p.monoms()       # [(2, 1), (1, 3)] — list of exponent tuples (SymPy 1.14: monoms, not monomials)
p.degree()       # 2 — max degree over all variables; p.degree(x)=2, p.degree(y)=3, p.total_degree()=4

# Path B: Expr.as_coefficients_dict on an Add (no Poly needed)
d = f.as_coefficients_dict()   # {x**2*y: 1, 3*x*y**3: 3}
# each key is a Mul; exponents via key.as_powers_dict() on Pow nodes
```

## Numeric evaluation

```python
f = se.sin(x) * se.exp(-x**2)
g = se.lambdify([x], [f])      # args AND exprs must be ITERABLES in 0.14.1
r = g([0.5])                   # -> 0-d numpy array; use float(r) or r.item()
f.subs(x, se.Float(0.5)).evalf(50)   # MPFR high-precision
```

Pitfalls (verified 0.14.1):
- `se.lambdify(x, f)` with a bare symbol FAILS: `TypeError: Value after *
  must be an iterable, not Pow` — the wrapper does `Lambdify(args, *exprs)`.
- `se.lambdify([x], [f])` returns a function taking a list and returning a
  **0-d numpy ndarray** for a single expression (not a list). Use
  `float(g([0.5]))`.
- No `mode=` kwarg exists in 0.14.1 (`TypeError: __cinit__() got an
  unexpected keyword argument 'mode'`).
- Irene's `engine.lambdify` deliberately always uses **SymPy** lambdify
  (see `symbolic_engine.py` line 293-308) — do not switch it to
  `se.lambdify` without handling the ndarray return type.

## Matrix API

```python
M = se.DenseMatrix(2, 2, [1, x, 0, 1])   # flat row-major entries
M.T          # transpose (DenseMatrix)
M[0, 1]      # entry access -> x
M.tolist()   # [[1, x], [0, 1]]
M.rows, M.cols
v = se.DenseMatrix(2, 1, [x, 1])
P = se.DenseMatrix(1, 2, [1, x])
(P * v)      # DenseMatrix(1,1) — matrix product in C++
se.zeros(2, 3), se.ones(2, 2), se.eye(3), se.diag(x, 1)
A = se.symarray('a', (3, 3))   # returns a NUMPY array of Symbols (a_0_0, ...)
```

## Printing

```python
str(f)                  # canonical string: '3*x*y**3 + x**2*y'
se.latex(f)             # LaTeX string (uses ^ not ^{ }: 'x^2')
se.ccode(f)             # C code
se.unicode(f)           # unicode pretty-print
se.init_printing()      # enable rich repr in REPL
```

## Solving (limited)

```python
a = se.symbols('a')
se.linsolve([se.Eq(x + a, 1), se.Eq(x - a, 3)], [x, a])   # linear systems only
se.perfect_power(81)           # True — INTEGER-only, single arg (not symbolic)
se.integer_nthroot(27, 3)     # (3, True)
se.isprime(97)                # True
```

## Type predicates (methods on Expr)

`is_Add, is_Mul, is_Pow, is_Symbol, is_Number, is_Integer, is_Rational,
is_Float, is_Boolean, is_Relational, is_Equality, is_Function,
is_Derivative, is_Dummy, is_Atom, is_commutative, is_integer, is_real,
is_positive, is_negative, is_zero, is_finite, is_AlgebraicNumber,
is_Matrix, is_Not` — all verified present on Expr in 0.14.1.

## Singleton `S`

`se.S` provides: `One, Zero, Half, NegativeOne, E, I (ImaginaryUnit),
Pi, Catalan, EulerGamma, GoldenRatio, Infinity, NegativeInfinity,
ComplexInfinity, NaN, true, false, Integers, Rationals, Reals,
UniversalSet, EmptySet`.

## IreneRewrite integration notes

- `Irene/symbolic_engine.py` wraps all of this: `engine.expand` uses
  `se.expand` (C++) for SymEngine objects; `engine.groebner`,
  `engine.reduced`, `engine.Poly`, `engine.lambdify`, `engine.Function`,
  `engine.PolyMatrix`, `engine.DomainMatrix` are ALWAYS SymPy (SymEngine
  lacks them).
- `to_sympy(obj)` uses `obj._sympy_()` with string round-trip fallback.
- `to_symengine(obj)` calls `se.sympy2symengine` which is ABSENT in
  0.14.1 -> returns the SymPy object unchanged (silent no-op). Fix:
  import `sympy2symengine` from `symengine.lib.symengine_wrapper`
  (deprecated but functional) or use `se.sympify(str(obj))`.
- `fallback_to_sympy` (decorator, symbolic_engine.py:126) builds its
  exception tuple with `se.LibExpressionException` at DECORATION time,
  but that name is ABSENT in 0.14.1 — so decorating ANY function with it
  would raise AttributeError. LATENT BUG: the decorator is currently
  never applied to any function (verified: zero usages), so the import
  succeeds and nothing breaks. Fix before first use: guard with
  `getattr(se, 'LibExpressionException', None)` and filter out None.
- P5.6 fast path in `Irene/relaxations.py` (lines ~830-861): inert on
  this installation for the reasons above; the SymPy fallback path
  executes. No correctness impact, but the intended C++ speedup is not
  active.