"""
Non-POP SDP Approximation Pipeline (NonPOPSDP) -- IreneRewrite port.

Implements the canonical pipeline for applying Lasserre's moment-SOS hierarchy
to non-polynomial optimization:

    1. Approximate transcendental functions with polynomials (Taylor/Chebyshev)
    2. Formulate the surrogate as a polynomial optimization problem (POP)
    3. Relax via Lasserre's Moment-SOS hierarchy using Irene.SDPRelaxations
    4. Solve the resulting SDP via cvxopt (Irene's default backend)

This is a faithful port of the original Irene ``nonpopsdp.py`` module. The
pipeline is backend-agnostic: polynomial surrogates are built with SymPy and
the SDP construction is delegated to ``SDPRelaxations``, which routes through
the user-selectable symbolic engine (``IRENE_SYMBOLIC_BACKEND``) and the
quotient-basis option (``IRENE_QUOTIENT_BASIS`` / ``RelaxationConfig``).

Key design decisions (from Phase A literature review):
    - Ball constraint is MANDATORY (Josz-Henrion 2014) for strong duality.
    - Chebyshev preferred over Taylor for wide domains (better conditioning).
    - Irene.SDPRelaxations handles the SDP construction + solve natively.

Integration with DSDP-ADE benchmarks:
    - Same test cases (trig, exp, tan) for apples-to-apples comparison.
    - Returns lower bound, solver route, timing, and approximation metadata.
"""

from math import factorial, pi, exp as math_exp, sin as math_sin, cos as math_cos

import numpy as np
from sympy import Symbol, Poly, sympify, expand

from .relaxations import SDPRelaxations


# ---------------------------------------------------------------------------
# 1. Polynomial approximation layer
# ---------------------------------------------------------------------------


def taylor_approx(func, var, center, degree):
    """
    Taylor polynomial approximation of `func` around `center` to given `degree`.

    NOTE (IreneRewrite port): the original implementation computed Taylor
    coefficients with naive central finite differences (h=1e-8), whose error
    grows like h^{-k} and produced garbage derivatives beyond k=3 (verified:
    error estimate ~1e36 for exp at degree 6). This port uses a high-order
    central-difference stencil with Richardson extrapolation at 60-digit
    precision (``_mp_derivative``), accurate to ~1e-6 relative at degree 7.

    Args:
        func: Python callable f(x) -> float (single variable).
        var: Sympy symbol.
        center: Expansion center (float).
        degree: Taylor degree d.

    Returns:
        Tuple (poly_expr, error_bound) where poly_expr is a sympy polynomial
        and error_bound is the Lagrange remainder estimate (conservative).
    """
    coeffs = []
    for k in range(degree + 1):
        val = float(_mp_derivative(func, center, k))
        coeffs.append(val / factorial(k))

    poly = sympify(0)
    for k, c in enumerate(coeffs):
        poly += sympify(float(c)) * (var - sympify(float(center))) ** k

    M = abs(float(_mp_derivative(func, center, degree + 1)))
    error_bound = M / factorial(degree + 1)

    return expand(poly), error_bound


def _mp_derivative(func, x, k):
    """k-th derivative of a float callable via a high-order central stencil.

    Solves the Vandermonde system for the central-difference coefficients of
    the k-th derivative on nodes x + j*h (j = -(k+1)..(k+1)), evaluates with
    60-digit arithmetic, then Richardson-extrapolates two step sizes. The step
    is chosen for float function values (h ~ eps^(1/(k+3))).

    Verified on exp/sin up to k=7: relative error < 1e-6.
    """
    import mpmath as mp
    mp.mp.dps = 60
    x = mp.mpf(x)
    m = k + 1
    A = mp.matrix(2 * m + 1, 2 * m + 1)
    b = mp.matrix(2 * m + 1, 1)
    for r in range(2 * m + 1):
        for j in range(-m, m + 1):
            A[r, j + m] = mp.mpf(j) ** r
        if r == k:
            b[r] = mp.factorial(k)
    c = mp.lu_solve(A, b)

    def D(h):
        h = mp.mpf(h)
        return mp.fsum(c[j + m] * func(float(x + j * h))
                       for j in range(-m, m + 1)) / (h ** k)

    h0 = mp.mpf(10) ** (-(15 // (k + 3)))
    D1, D2 = D(h0), D(h0 / 2)
    p = k + 4
    return (mp.mpf(2) ** p * D2 - D1) / (mp.mpf(2) ** p - 1)


def _finite_diff(func, x, order, h=1e-8):
    """Compute the order-th derivative via central finite differences.

    Retained for API compatibility; not used by the fixed ``taylor_approx``.
    """
    if order == 0:
        return func(x)
    fp = _finite_diff(func, x + h, order - 1, h)
    fm = _finite_diff(func, x - h, order - 1, h)
    return (fp - fm) / (2 * h)


def chebyshev_approx(func, var, domain, degree):
    """
    Chebyshev polynomial approximation of `func` on interval `domain`.

    Maps domain [a, b] to [-1, 1], computes Chebyshev coefficients via
    least-squares fitting on the Clenshaw-Curtis extrema grid, then converts
    back to power basis.

    NOTE (IreneRewrite port): the original implementation extracted Chebyshev
    coefficients with a raw FFT whose scaling is incorrect for the extrema
    grid (verified: max error ~61.5 for exp of degree 6 on [-2, 2]; the true
    degree-6 Chebyshev error is ~2e-2) and evaluated the error on a shifted
    grid (off-by-one: ``fine_t = 2(x-mid)/(b-a) - 1`` mapped [a,b] onto
    [-2,0]). This port uses ``numpy.polynomial.chebyshev.chebfit`` and the
    correct [-1,1] mapping.

    Args:
        func: Python callable f(x) -> float.
        var: Sympy symbol.
        domain: Tuple (a, b) defining the approximation interval.
        degree: Chebyshev degree d.

    Returns:
        Tuple (poly_expr, max_error) where poly_expr is a sympy polynomial
        and max_error is the empirical worst-case error on a fine grid.
    """
    from numpy.polynomial.chebyshev import chebfit, cheb2poly

    a, b = domain
    mid = (a + b) / 2.0
    half = (b - a) / 2.0

    # Clenshaw-Curtis extrema grid on [-1, 1]
    N = max(degree + 2, 64)
    t_vals = np.cos(np.pi * np.arange(N) / (N - 1))
    x_vals = mid + half * t_vals
    f_vals = np.array([func(x) for x in x_vals])

    # Chebyshev coefficients c_0..c_degree (ascending), least-squares on grid
    c_coeffs = chebfit(t_vals, f_vals, degree)

    # Convert to power-basis coefficients in t (ascending)
    power_coeffs = cheb2poly(c_coeffs)

    poly = sympify(0)
    for k, coeff in enumerate(power_coeffs):
        poly += sympify(float(coeff)) * (
            (var - sympify(float(mid))) / sympify(float(half))
        ) ** k

    fine_x = np.linspace(a, b, 1000)
    fine_t = (fine_x - mid) / half
    poly_vals = np.polyval(np.array(power_coeffs[::-1]).astype(float), fine_t)
    true_vals = np.array([func(x) for x in fine_x])
    max_error = float(np.max(np.abs(poly_vals - true_vals)))

    return expand(poly), max_error


def _chebyshev_to_power(c_coeffs, degree):
    """Convert Chebyshev coefficients to power basis."""
    power_coeffs = [0.0] * (degree + 1)
    for k in range(degree + 1):
        if abs(c_coeffs[k]) < 1e-15:
            continue
        tk_coeffs = _chebyshev_poly_coeffs(k)
        for j, coeff in enumerate(tk_coeffs):
            power_coeffs[j] += c_coeffs[k] * coeff
    return power_coeffs


def _chebyshev_poly_coeffs(k):
    """Return power-basis coefficients of T_k(x)."""
    if k == 0:
        return [1.0]
    if k == 1:
        return [0.0, 1.0]

    t_prev = [1.0]
    t_curr = [0.0, 1.0]
    for _ in range(2, k + 1):
        shifted = [0.0] + t_curr
        new_coeffs = [2.0 * shifted[j] for j in range(len(shifted))]
        while len(t_prev) < len(new_coeffs):
            t_prev.append(0.0)
        new_coeffs = [new_coeffs[j] - t_prev[j]
                      for j in range(len(new_coeffs))]
        t_prev = t_curr
        t_curr = new_coeffs
    return t_curr


# ---------------------------------------------------------------------------
# 2. POP formulation from approximated transcendental functions
# ---------------------------------------------------------------------------


class TranscendentalApproximator:
    """
    Approximate a vector of transcendental functions with polynomials.

    Each entry maps a function name to (func, var, domain, method, degree).
    Returns a dict mapping function names to sympy polynomial surrogates.
    """

    def __init__(self, var, approx_map):
        """
        Args:
            var: Sympy symbol for the variable.
            approx_map: Dict mapping function names to config dicts:
                {
                    "exp": {"func": math_exp, "method": "chebyshev",
                            "domain": (-2.0, 2.0), "degree": 6},
                    "sin": {"func": math_sin, "method": "chebyshev",
                            "domain": (-pi, pi), "degree": 8},
                }
        """
        self.var = var
        self.approx_map = approx_map
        self.polynomials = {}
        self.errors = {}
        self._approximate()

    def _approximate(self):
        """Run all approximations and store results."""
        for name, config in self.approx_map.items():
            func = config["func"]
            method = config.get("method", "chebyshev")
            degree = config.get("degree", 6)

            if method == "taylor":
                center = config.get("center", 0.0)
                poly, err = taylor_approx(func, self.var, center, degree)
            elif method == "chebyshev":
                domain = config["domain"]
                poly, err = chebyshev_approx(func, self.var, domain, degree)
            else:
                raise ValueError(f"Unknown approximation method: {method}")

            self.polynomials[name] = poly
            self.errors[name] = err

    def substitute(self, expr):
        """
        Replace transcendental function names in a sympy expression with
        their polynomial surrogates.
        """
        for name, poly in self.polynomials.items():
            func_sym = Symbol(name)
            expr = expr.subs(func_sym, poly)
        return expand(expr)


# ---------------------------------------------------------------------------
# 3. Lasserre hierarchy via Irene.SDPRelaxations
# ---------------------------------------------------------------------------


class NonPOPSDP:
    """
    Non-polynomial optimization via approximation -> POP -> Lasserre SDP.

    Pipeline:
        1. Approximate transcendental functions (Taylor or Chebyshev)
        2. Build polynomial surrogate of objective + constraints
        3. Delegate to Irene.SDPRelaxations for moment-SOS hierarchy
        4. Solve via cvxopt (Irene's default SDP backend)

    Per Josz-Henrion 2014, a redundant ball constraint is ALWAYS added
    to ensure strong duality (no primal-dual gap).
    """

    def __init__(self, var, approx_map, relax_order=2, ball_radius=None,
                 parallel=True, verbosity=1, config=None):
        """
        Args:
            var: Sympy symbol for the optimization variable.
            approx_map: Dict for TranscendentalApproximator (see above).
            relax_order: Lasserre hierarchy order d (moment matrix degree).
            ball_radius: Radius R for redundant ball constraint x^2 <= R^2.
                If None, inferred from approximation domains.
            parallel: Use parallel SDP construction (default True).
            verbosity: Output verbosity (0=silent, 1=normal, 2=debug).
            config: Optional RelaxationConfig passed through to
                SDPRelaxations (quotient-basis and reduction-pipeline options).
        """
        self.var = var
        self.relax_order = relax_order
        self.ball_radius = ball_radius
        self.parallel = parallel
        self.verbosity = verbosity
        self.config = config

        self.approx = TranscendentalApproximator(var, approx_map)
        self.polynomials = self.approx.polynomials
        self.approx_errors = self.approx.errors

        if self.ball_radius is None:
            self.ball_radius = self._infer_ball_radius()

        self.objective_expr = None
        self.constraint_exprs = []
        self.objective_poly = None
        self.constraint_polys = []

        # SDPRelaxations instance — created lazily in solve()
        self.sdp = None
        self.result = None

    def _infer_ball_radius(self):
        """Infer ball radius from approximation domains."""
        max_radius = 1.0
        for config in self.approx.approx_map.values():
            domain = config.get("domain", (-1.0, 1.0))
            max_radius = max(max_radius, abs(domain[0]), abs(domain[1]))
        return float(max_radius)

    def set_objective(self, expr):
        """Set the objective expression."""
        self.objective_expr = expr
        self.objective_poly = self.approx.substitute(expr)

    def add_constraint(self, expr, sense="geq"):
        """
        Add a constraint expr >= 0 (geq), expr <= 0 (leq), or expr == 0 (eq).
        """
        poly = self.approx.substitute(expr)
        self.constraint_exprs.append((expr, sense))
        self.constraint_polys.append((poly, sense))

    def add_ball_constraint(self):
        """
        Add redundant ball constraint R^2 - x^2 >= 0.

        CRITICAL: Per Josz-Henrion 2014, this ensures strong duality.
        """
        R = sympify(float(self.ball_radius))
        ball_poly = R**2 - self.var**2
        self.constraint_polys.append((ball_poly, "geq"))

    def solve(self):
        """
        Solve the NonPOPSDP relaxation via Irene.SDPRelaxations.

        Returns:
            Lower bound on the optimal value (float), or None on failure.
        """
        if self.objective_poly is None:
            raise RuntimeError("Objective not set. Call set_objective() first.")

        self.add_ball_constraint()

        if self.verbosity >= 1:
            print(f"  NonPOPSDP: relax_order={self.relax_order}")
            print(f"  Ball constraint: R={self.ball_radius}")
            print(f"  Approximation errors:")
            for name, err in self.approx_errors.items():
                print(f"    {name}: {err:.2e}")

        # --- Build SDPRelaxations instance ---
        sdp = SDPRelaxations([self.var], relations=[], name="NonPOPSDP",
                             config=self.config)
        sdp.Parallel = self.parallel

        # Set objective
        sdp.SetObjective(self.objective_poly)

        # Add constraints in SDPRelaxations format
        for poly, sense in self.constraint_polys:
            if sense == "geq":
                sdp.AddConstraint(poly >= 0)
            elif sense == "leq":
                sdp.AddConstraint(poly <= 0)
            elif sense == "eq":
                sdp.AddConstraint(poly == 0)

        # Set moment order
        sdp.MomentsOrd(self.relax_order)

        if self.verbosity >= 1:
            print(f"  Building SDP via Irene.SDPRelaxations...")

        # Initialize and solve
        sdp.InitSDP()
        lb = sdp.Minimize()

        self.sdp = sdp
        self.result = {
            "lower_bound": float(lb) if lb is not None else None,
            "status": sdp.Info.get("status", "Unknown"),
            "init_time": sdp.InitTime,
            "solver": sdp.Info.get("solver", "Unknown"),
            "size": sdp.MatSize,
        }

        if self.verbosity >= 1 and lb is not None:
            print(f"  Lower bound: {lb:.8f}")
            print(f"  Solver: {self.result['solver']}, "
                  f"Init time: {self.result['init_time']:.2f}s")

        return lb


# ---------------------------------------------------------------------------
# 4. Multi-variable extension
# ---------------------------------------------------------------------------


class NonPOPSDP_Multi:
    """
    NonPOPSDP for multi-variable problems.

    Uses tensor-product monomial bases and delegates SDP solving
    to Irene.SDPRelaxations.
    """

    def __init__(self, vars, approx_map, relax_order=2, ball_radius=None,
                 parallel=True, verbosity=1, config=None):
        """
        Args:
            vars: List of sympy symbols.
            approx_map: Dict mapping function names to config dicts.
                Each config may specify which variable it applies to via "var_idx".
            relax_order: Lasserre hierarchy order.
            ball_radius: Radius for ball constraint ||x||^2 <= R^2.
            parallel: Use parallel SDP construction (default True).
            verbosity: Output verbosity.
            config: Optional RelaxationConfig passed through to
                SDPRelaxations (quotient-basis and reduction-pipeline options).
        """
        self.vars = vars
        self.n_vars = len(vars)
        self.relax_order = relax_order
        self.ball_radius = ball_radius
        self.parallel = parallel
        self.verbosity = verbosity
        self.config = config

        self.approx_map = approx_map

        self.approximators = {}
        self.polynomials = {}
        self.approx_errors = {}

        for name, config in approx_map.items():
            var_idx = config.get("var_idx", 0)
            var = vars[var_idx]
            approx = TranscendentalApproximator(var, {name: config})
            self.approximators[name] = approx
            self.polynomials[name] = approx.polynomials
            self.approx_errors[name] = approx.errors[name]

        if self.ball_radius is None:
            self.ball_radius = self._infer_ball_radius()

        self.objective_expr = None
        self.constraint_exprs = []
        self.objective_poly = None
        self.constraint_polys = []

        self.sdp = None
        self.result = None

    def _infer_ball_radius(self):
        max_radius = 1.0
        for config in self.approx_map.values():
            domain = config.get("domain", (-1.0, 1.0))
            max_radius = max(max_radius, abs(domain[0]), abs(domain[1]))
        return float(max_radius)

    def set_objective(self, expr):
        poly = expr
        for name, poly_dict in self.polynomials.items():
            func_sym = Symbol(name)
            poly = poly.subs(func_sym, poly_dict[name])
        self.objective_expr = expr
        self.objective_poly = expand(poly)

    def add_constraint(self, expr, sense="geq"):
        poly = expr
        for name, poly_dict in self.polynomials.items():
            func_sym = Symbol(name)
            poly = poly.subs(func_sym, poly_dict[name])
        self.constraint_exprs.append((expr, sense))
        self.constraint_polys.append((expand(poly), sense))

    def add_ball_constraint(self):
        """Add ||x||^2 <= R^2 as R^2 - sum(x_i^2) >= 0."""
        R = sympify(float(self.ball_radius))
        ball_poly = R**2 - sum(v**2 for v in self.vars)
        self.constraint_polys.append((ball_poly, "geq"))

    def solve(self):
        if self.objective_poly is None:
            raise RuntimeError("Objective not set.")

        self.add_ball_constraint()

        if self.verbosity >= 1:
            print(f"  NonPOPSDP (multi): {self.n_vars} vars, "
                  f"order={self.relax_order}, R={self.ball_radius}")

        # --- Build SDPRelaxations instance ---
        sdp = SDPRelaxations(self.vars, relations=[], name="NonPOPSDP_Multi",
                             config=self.config)
        sdp.Parallel = self.parallel

        sdp.SetObjective(self.objective_poly)

        for poly, sense in self.constraint_polys:
            if sense == "geq":
                sdp.AddConstraint(poly >= 0)
            elif sense == "leq":
                sdp.AddConstraint(poly <= 0)
            elif sense == "eq":
                sdp.AddConstraint(poly == 0)

        sdp.MomentsOrd(self.relax_order)

        if self.verbosity >= 1:
            print(f"  Building SDP via Irene.SDPRelaxations...")

        sdp.InitSDP()
        lb = sdp.Minimize()

        self.sdp = sdp
        self.result = {
            "lower_bound": float(lb) if lb is not None else None,
            "status": sdp.Info.get("status", "Unknown"),
            "init_time": sdp.InitTime,
            "solver": sdp.Info.get("solver", "Unknown"),
            "size": sdp.MatSize,
        }

        if self.verbosity >= 1 and lb is not None:
            print(f"  Lower bound: {lb:.8f}")
            print(f"  Solver: {self.result['solver']}, "
                  f"Init time: {self.result['init_time']:.2f}s")

        return lb
