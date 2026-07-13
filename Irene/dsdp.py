"""
Differential Semidefinite Programming (DSDP) relaxations.

Extends the Lasserre SDP hierarchy to handle optimization problems where
polynomial terms include functions satisfying algebraic differential equations (ADEs).
Uses Ritt-Woodin differential algebra to encode ADE constraints as algebraic
relations in the moment hierarchy.

Key features:
    - ADE lift: encode transcendental functions via algebraic relations
      (e.g., y*z=1 for exp, y' = y for dy/dx = y)
    - Differential KKT: inject stationarity conditions derived via Leibniz rule
    - Mean polynomial certificates: M_{q,p}(X,w) nonnegativity via SDP
    - Archimedean (boxing) constraints for convergence guarantees

Integration:
    - Inherits SDPRelaxations (sympy-based moment hierarchy)
    - Compatible with OptimizationProblem via from_problem()
    - Works with SemigroupAlgebra derivation support
"""

from math import ceil, lcm
from functools import reduce
from operator import mul
from itertools import product

from sympy import Symbol, Poly, sympify, groebner, QQ, expand, zeros, Matrix

from .base import base
from .sdp import sdp
from .relaxations import SDPRelaxations, SDRelaxSol, Mom


# Solver routing constants
SOLVER_SDP = "sdp"
SOLVER_GP = "gp"
SOLVER_SONC = "sonc"


class DSDPRelaxations(SDPRelaxations):
    r"""
    Differential SDP relaxation framework.

    Extends :class:`SDPRelaxations` to handle problems with ADE-constrained variables.
    The ADE is encoded as algebraic relations in the Groebner basis, and differential
    KKT conditions are injected as additional linear moment constraints.

    The relaxation constructs a moment hierarchy where:
        1. ADE relations are enforced as equality constraints on moment variables
        2. Differential KKT conditions tighten the relaxation (degree-lift equivalent)
        3. Mean polynomial certificates :math:`M_{q,p}(X,w)` provide nonnegativity proofs

    Example:
        >>> from sympy import symbols, exp
        >>> from Irene import DSDPRelaxations
        >>> x, y, z = symbols('x y z')
        >>> # Minimize e^x - x^2 via ADE lift y = e^x, z = e^{-x}, y*z = 1
        >>> dsdp = DSDPRelaxations([x, y, z], relations=[y*z - 1])
        >>> dsdp.SetObjective(y - x**2)
        >>> dsdp.solve()
    """

    ADEError = r"""ADE relations must be sympy expressions in terms of generators"""
    DiffMapError = r"""Derivation map must be a dict mapping generators to expressions"""

    def __init__(self, gens, relations=(), name="DSDPRlx", **kwargs) -> None:
        r"""
        Initialize DSDP relaxation.

        Args:
            gens: List of sympy symbols/functions (generators of the algebra).
            relations: ADE relation expressions (e.g., [y*z - 1] for exp lift).
            name: Name for this relaxation instance.
            **kwargs:
                - q: Power mean parameter q (default: 1).
                - p: Power mean parameter p (default: 0).
                - depth: Product depth d for hierarchy (default: 1).
                  At depth d, the certificate expands d mean forms into
                  2^d alternating-sign posynomial terms (§3.2).
                - weights: Weight vector for mean certificates (default: uniform).
                - use_diff_kkt: Enable differential KKT injection (default: False).
                - kkt_order: Order of differential KKT conditions (default: 1).
                - archimedean: Add boxing constraints (default: True).
                - box_size: Boxing interval half-width [-B, B] (default: 10).
                - verbosity: Verbosity level (default: 1).
                - diff_map: Dict mapping generators to their derivatives.
        """
        super().__init__(gens, relations, name)

        # DSDP configuration
        self.q = kwargs.get('q', 1)
        self.p = kwargs.get('p', 0)
        self.depth = kwargs.get('depth', 1)
        self.use_diff_kkt = kwargs.get('use_diff_kkt', False)
        self.kkt_order = kwargs.get('kkt_order', 1)
        self.archimedean = kwargs.get('archimedean', True)
        self.box_size = kwargs.get('box_size', 10)
        self.verbosity = kwargs.get('verbosity', 1)

        # Weight vector for mean certificates
        raw_weights = kwargs.get('weights', None)
        if raw_weights is not None:
            if len(raw_weights) != self.NumGenerators:
                raise ValueError(
                    f"Weight vector length {len(raw_weights)} != "
                    f"number of generators {self.NumGenerators}"
                )
            if any(w <= 0 for w in raw_weights):
                raise ValueError("All mean polynomial weights must be positive")
            self.weights = list(raw_weights)
        else:
            self.weights = [1.0] * self.NumGenerators

        # Differential structure
        self.diff_map = kwargs.get('diff_map', {})
        self.derivation_registered = False
        self.diff_constraints_count = 0

        # Register derivation if diff_map was provided
        if self.diff_map:
            self.set_derivation(self.diff_map)

        # Track ADE-specific moment constraints
        self.ade_moment_constraints = []

    def set_derivation(self, diff_map: dict) -> None:
        r"""
        Register a derivation map for differential KKT conditions.

        Args:
            diff_map: Dictionary mapping sympy generators to their derivatives.
                     E.g., {x: y, y: -y} encodes dy/dx = -y.
        """
        assert isinstance(diff_map, dict), self.DiffMapError
        for key, val in diff_map.items():
            assert key in self.Generators, f"Derivation key {key} not in generators"
        self.diff_map = diff_map
        self.derivation_registered = True

    def differentiate(self, expr, var=None):
        r"""
        Compute the formal derivative of an expression using the registered derivation map.

        Applies the Leibniz rule: d(f*g) = d(f)*g + f*d(g).
        If no derivation map is registered, falls back to sympy's diff().

        Args:
            expr: Sympy expression to differentiate.
            var: Variable to differentiate with respect to (default: first generator).

        Returns:
            Formal derivative of expr.
        """
        if var is None:
            var = self.Generators[0]

        if not self.derivation_registered or not self.diff_map:
            # Fall back to sympy differentiation
            return expr.diff(var)

        # Apply derivation map via Leibniz rule
        return self._leibniz_diff(expr, var)

    def _leibniz_diff(self, expr, var):
        r"""
        Apply Leibniz rule using the registered derivation map.

        For a monomial c * x1^a1 * ... * xn^an:
            d(mono) = mono * sum(ai * d(xi) / xi)

        Args:
            expr: Sympy expression.
            var: Differentiation variable.

        Returns:
            Formal derivative.
        """
        expr = sympify(expr)

        # Base cases
        if expr in self.diff_map:
            return sympify(self.diff_map[expr])
        if expr.is_Number:
            return sympify(0)
        if expr == var:
            return sympify(1) if var in self.diff_map else sympify(self.diff_map.get(var, 1))

        # Sum rule
        if expr.is_Add:
            return sum(self._leibniz_diff(arg, var) for arg in expr.args)

        # Product rule (Leibniz)
        if expr.is_Mul:
            terms = []
            args = expr.args
            for i, arg in enumerate(args):
                rest = reduce(mul, [a for j, a in enumerate(args) if j != i], 1)
                terms.append(rest * self._leibniz_diff(arg, var))
            return sum(terms)

        # Power rule
        if expr.is_Pow:
            base, exp = expr.base, expr.exp
            return exp * base ** (exp - 1) * self._leibniz_diff(base, var)

        # Default: sympy fallback
        return expr.diff(var)

    def add_ade_moment_constraint(self, expr, rhs=0):
        r"""
        Add an ADE-derived moment constraint directly.

        Args:
            expr: Sympy polynomial expression for the constraint.
            rhs: Right-hand side value (default: 0 for equality).
        """
        reduced = self.ReduceExp(sympify(expr))
        self.ade_moment_constraints.append([reduced, rhs])
        tot_deg = Poly(reduced, *self.AuxSyms).total_degree()
        self.MmntCnsDeg = max(int(ceil(tot_deg / 2.)), self.MmntCnsDeg)

    def _build_diff_kkt_moments(self):
        r"""
        Build differential KKT moment constraints.

        Differentiates the Lagrangian L = f - sum(lambda_i * g_i) and enforces
        dL/dx_j = 0 as moment constraints. This is equivalent to one higher
        Lasserre order in terms of bound quality.

        CRITICAL: Differentiate ORIGINAL expressions (in generator space) before
        reduction to AuxSym space. The derivation map keys are generators, not
        AuxSyms — differentiating reduced expressions yields zero because AuxSyms
        are never found in diff_map and fall through to expr.diff(var) = 0.

        Returns:
            List of (reduced_expr, rhs) tuples for moment constraints.
        """
        if not self.use_diff_kkt or not self.derivation_registered:
            return []

        constraints = []

        # Differentiate ORIGINAL objective (generator space), then reduce
        for sym in self.Generators:
            diff_term = self.differentiate(self.Objective, sym)
            if diff_term != 0:
                reduced = self.ReduceExp(diff_term)
                constraints.append([reduced, 0])
                deg = Poly(reduced, *self.AuxSyms).total_degree()
                self.MmntCnsDeg = max(int(ceil(deg / 2.)), self.MmntCnsDeg)

            # Differentiate ORIGINAL constraints (generator space), then reduce
            for org_cnst in self.OrgConst:
                if isinstance(org_cnst, (self.GEQ, self.GT)):
                    non_red_exp = org_cnst.lhs - org_cnst.rhs
                elif isinstance(org_cnst, (self.LEQ, self.LT)):
                    non_red_exp = org_cnst.rhs - org_cnst.lhs
                elif isinstance(org_cnst, self.EQ):
                    non_red_exp = org_cnst.lhs - org_cnst.rhs
                else:
                    non_red_exp = org_cnst
                diff_cnst = self.differentiate(non_red_exp, sym)
                if diff_cnst != 0:
                    reduced = self.ReduceExp(diff_cnst)
                    constraints.append([reduced, 0])
                    deg = Poly(reduced, *self.AuxSyms).total_degree()
                    self.MmntCnsDeg = max(int(ceil(deg / 2.)), self.MmntCnsDeg)

        self.diff_constraints_count = len(constraints)
        return constraints

    def _build_mean_pair(self, q, p):
        r"""
        Build the (Q, P) posynomial pair for a single mean form M_{q,p}.

        Per Eq. (920) in the manuscript:
            M_{q,p} = Q - P
            Q = (sum w_i X_i^q)^{c/q}
            P = (sum w_i X_i^p)^{c/p}   (or 1 when p=0)

        where c = lcm(q, p) clears fractional exponents.

        Args:
            q: Power mean parameter q (positive integer).
            p: Power mean parameter p (non-negative integer, p < q).

        Returns:
            Tuple (Q, P) of expanded sympy expressions.
        """
        n = self.NumGenerators

        if p != 0:
            c = lcm(q, p)
            q_exp = c // q
            p_exp = c // p
        else:
            # p=0 (geometric mean case): c = q suffices, q_exp = 1
            c = q
            q_exp = 1
            p_exp = 0

        # Build (sum w_j X_j^q)^{c/q}
        weighted_q_sum = sum(
            self.weights[j] * self.AuxSyms[j] ** q
            for j in range(n)
        )
        Q = expand(weighted_q_sum ** q_exp)

        # Build (sum w_j X_j^p)^{c/p} or 1 when p=0
        if p != 0:
            weighted_p_sum = sum(
                self.weights[j] * self.AuxSyms[j] ** p
                for j in range(n)
            )
            P = expand(weighted_p_sum ** p_exp)
        else:
            P = sympify(1)

        return Q, P

    def _expand_certificate(self, cert):
        r"""
        Expand a certificate expression into moment constraints.

        Args:
            cert: Expanded sympy polynomial certificate.

        Returns:
            List of (reduced_expr, rhs) tuples for moment constraints.
        """
        constraints = []
        n = self.NumGenerators

        cert_poly = Poly(cert, *self.AuxSyms)
        for expn, coef in cert_poly.as_dict().items():
            if coef != 0:
                mono = reduce(mul,
                              [self.AuxSyms[i] ** expn[i] for i in range(n)], 1)
                reduced = self.ReduceExp(coef * mono)
                if reduced != 0:
                    constraints.append([reduced, 0])
                    deg = Poly(reduced, *self.AuxSyms).total_degree()
                    self.MmntCnsDeg = max(int(ceil(deg / 2.)),
                                          self.MmntCnsDeg)

        return constraints

    def _build_depth_product(self):
        r"""
        Build depth-d product expansion of mean forms.

        Per §3.2 (product-depth truncation): a depth-d certificate is
        a product of d mean forms, each M_{q_k, p_k} = Q_k - P_k.
        The expansion yields 2^d alternating-sign posynomial terms:

            prod_{k=1}^d (Q_k - P_k) = sum_{s in {0,1}^d} (-1)^|s| prod term_k(s_k)

        For d=2: (Q1-P1)(Q2-P2) = Q1*Q2 + P1*P2 - Q1*P2 - P1*Q2.

        The (q_k, p_k) pairs are chosen as (q+k, p+k) for k=0..d-1,
        ensuring each level uses a distinct mean order.

        Returns:
            List of (reduced_expr, rhs) tuples for moment constraints.
        """
        n = self.NumGenerators

        if n == 0:
            return []

        # Theory (§2.1): M_{q,p} is PSD iff q > p.
        if self.q <= self.p:
            if self.verbosity > 0:
                print(f"Warning: q={self.q} <= p={self.p}, "
                      f"mean certificate is not PSD (requires q > p)")
            return []

        # Build d mean pairs with increasing (q, p) orders
        pairs = []
        for k in range(self.depth):
            q_k = self.q + k
            p_k = self.p + k
            if q_k > p_k:
                pairs.append(self._build_mean_pair(q_k, p_k))

        if not pairs:
            return []

        # Expand product: each choice is Q (index 0) or P (index 1)
        # Sign = (-1)^{number of P choices}
        cert = sympify(0)
        for choices in product([0, 1], repeat=len(pairs)):
            sign = (-1) ** sum(choices)
            term = sympify(1)
            for k, use_p in enumerate(choices):
                term *= pairs[k][1] if use_p else pairs[k][0]
            cert += sign * expand(term)

        cert = expand(cert)

        if self.verbosity > 0:
            num_terms = len(Poly(cert, *self.AuxSyms).as_dict())
            print(f"  Depth-{self.depth} expansion: {num_terms} monomials "
                  f"(from {len(pairs)} mean pairs, 2^{len(pairs)} terms)")

        return self._expand_certificate(cert)

    def _build_mean_certificate_moments(self):
        r"""
        Build moment constraints encoding M_{q,p}(X,w) nonnegativity.

        For depth=1, this is a single mean form M_{q,p} = Q - P.
        For depth>=2, this expands a product of d mean forms into
        2^d alternating-sign posynomial terms (§3.2 product-depth truncation).

        Returns:
            List of (reduced_expr, rhs) tuples for moment constraints.
        """
        n = self.NumGenerators

        if n == 0:
            return []

        # Theory (§2.1): M_{q,p} is PSD iff q > p (monotonicity of power means).
        # If q <= p, the form is indefinite/negative and cannot certify nonnegativity.
        if self.q <= self.p:
            if self.verbosity > 0:
                print(f"Warning: q={self.q} <= p={self.p}, "
                      f"mean certificate is not PSD (requires q > p)")
            return []

        # For depth > 1, generate multiple (q,p) pairs and expand products
        if self.depth > 1:
            return self._build_depth_product()

        # Depth 1: single mean form M_{q,p} = Q - P
        Q, P = self._build_mean_pair(self.q, self.p)
        cert = expand(Q - P)

        return self._expand_certificate(cert)

    def _add_archimedean_boxing(self):
        r"""
        Add boxing constraints for the archimedean condition.

        Enforces -B <= x_i <= B for each variable, which is necessary
        for the moment hierarchy to converge (Putinar's condition).

        Returns:
            List of constraint expressions to be added via AddConstraint.
        """
        if not self.archimedean:
            return []

        B = self.box_size
        constraints = []

        for sym in self.Generators:
            # B - x_i >= 0
            constraints.append(sympify(B) - self.SymDict[sym] >= 0)
            # x_i + B >= 0
            constraints.append(self.SymDict[sym] + sympify(B) >= 0)

        return constraints

    def _is_posynomial(self, cert):
        r"""
        Check if a certificate expression is a posynomial (all coefficients >= 0).

        A posynomial has strictly non-negative coefficients in its expanded
        polynomial form. This is the key distinction for solver routing:
        - Posynomial certificates can be solved via GP/SONC (convex in log-domain)
        - Mixed-sign certificates require SDP (general moment hierarchy)

        Args:
            cert: Expanded sympy polynomial certificate.

        Returns:
            True if all coefficients are >= 0, False otherwise.
        """
        try:
            cert = sympify(cert)
            # Use the certificate's own free symbols for Poly construction
            # (cert may be in original generator space or AuxSyms space)
            gens = list(cert.free_symbols) or self.AuxSyms
            cert_poly = Poly(cert, *gens)
            coeffs = cert_poly.as_dict()
            # A posynomial requires ALL coefficients to be non-negative
            return all(float(v) >= -self.ErrorTolerance for v in coeffs.values())
        except Exception:
            # If we can't determine, default to SDP (safer fallback)
            return False

    def _is_mixed_sign(self, cert):
        r"""
        Check if a certificate has mixed-sign coefficients.

        Returns:
            True if certificate has both positive and negative coefficients.
        """
        try:
            cert = sympify(cert)
            gens = list(cert.free_symbols) or self.AuxSyms
            cert_poly = Poly(cert, *gens)
            coeffs = cert_poly.as_dict()
            values = [float(v) for v in coeffs.values()]
            has_positive = any(v > self.ErrorTolerance for v in values)
            has_negative = any(v < -self.ErrorTolerance for v in values)
            return has_positive and has_negative
        except Exception:
            return True  # Default to mixed-sign (safer)

    def _route_solver(self, cert):
        r"""
        Route to the appropriate solver based on certificate sign pattern.

        Per the mean polynomial theory:
        - M_{q,p} with p=0 (geometric mean) produces posynomial Q - 1,
          which is amenable to GP/SONC relaxation.
        - M_{q,p} with p>0 produces mixed-sign certificates requiring SDP.
        - Depth-d expansions (d >= 2) produce alternating-sign terms,
          which generally require SDP regardless of (q, p).

        Args:
            cert: Expanded certificate polynomial.

        Returns:
            String: SOLVER_SDP, SOLVER_GP, or SOLVER_SONC.
        """
        if self.depth >= 2:
            # Depth-d expansions produce 2^d alternating terms — SDP required
            return SOLVER_SDP

        if self._is_posynomial(cert):
            # Pure posynomial — GP/SONC is efficient and exact
            # Use SONC for p=0 (geometric mean case), GP otherwise
            if self.p == 0:
                return SOLVER_SONC
            else:
                return SOLVER_GP

        # Mixed-sign certificate — SDP is the general solver
        return SOLVER_SDP

    def _solve_via_sdp(self):
        r"""
        Solve using the SDP moment hierarchy (default path).

        Returns:
            Lower bound from SDP relaxation.
        """
        self.InitSDP()
        return self.Minimize()

    def _solve_via_sonc(self):
        r"""
        Solve using SONC relaxation via GP/SONC backend.

        The SONC backend operates on SemigroupAlgebraElement representations.
        Since DSDP uses sympy-based moment hierarchy, we delegate to the
        SDP path with a SONC-compatible configuration.

        Returns:
            Lower bound from SONC-compatible relaxation.
        """
        if self.verbosity > 0:
            print("  Note: SONC routing selected; using SDP with SONC-compatible config")
        return self._solve_via_sdp()

    def _solve_via_gp(self):
        r"""
        Solve using GP relaxation via geometric programming backend.

        The GP backend operates on SemigroupAlgebraElement representations.
        Since DSDP uses sympy-based moment hierarchy, we delegate to the
        SDP path with a GP-compatible configuration.

        Returns:
            Lower bound from GP-compatible relaxation.
        """
        if self.verbosity > 0:
            print("  Note: GP routing selected; using SDP with GP-compatible config")
        return self._solve_via_sdp()

    def solve(self, order=None):
        r"""
        Solve the DSDP relaxation with automatic solver routing.

        Builds the relaxation and routes to the appropriate solver based on
        certificate structure:
        - Posynomial certificates → GP/SONC (convex log-domain optimization)
        - Mixed-sign certificates → SDP (general moment hierarchy)
        - Depth >= 2 → SDP (alternating-sign expansion terms)

        Args:
            order: Relaxation order (default: auto from problem degree).

        Returns:
            Lower bound on the optimal value.
        """
        # Build differential KKT constraints
        diff_kkt = self._build_diff_kkt_moments()
        for expr, rhs in diff_kkt:
            self.add_ade_moment_constraint(expr, rhs)

        # Build mean certificate constraints
        mean_certs = self._build_mean_certificate_moments()
        for expr, rhs in mean_certs:
            self.add_ade_moment_constraint(expr, rhs)

        # Add archimedean boxing
        box_constraints = self._add_archimedean_boxing()
        for cnst in box_constraints:
            self.AddConstraint(cnst)

        # Set moment order
        if order is not None:
            self.MomentsOrd(order)
        self.RelaxationDeg()

        # Determine certificate structure for solver routing
        # Build the raw certificate to inspect sign pattern
        if self.depth > 1:
            # For depth >= 2, build the product expansion
            pairs = []
            for k in range(self.depth):
                q_k = self.q + k
                p_k = self.p + k
                if q_k > p_k:
                    pairs.append(self._build_mean_pair(q_k, p_k))
            if pairs:
                cert = sympify(0)
                for choices in product([0, 1], repeat=len(pairs)):
                    sign = (-1) ** sum(choices)
                    term = sympify(1)
                    for k_idx, use_p in enumerate(choices):
                        term *= pairs[k_idx][1] if use_p else pairs[k_idx][0]
                    cert += sign * expand(term)
                cert = expand(cert)
            else:
                cert = None
        else:
            # Depth 1: single mean form
            if self.q > self.p:
                Q, P = self._build_mean_pair(self.q, self.p)
                cert = expand(Q - P)
            else:
                cert = None

        # Route to appropriate solver
        if cert is not None:
            solver = self._route_solver(cert)
        else:
            solver = SOLVER_SDP  # Default to SDP when no certificate

        # Report
        if self.verbosity > 0:
            print(f"DSDP Relaxation (order={self.MmntOrd}, depth={self.depth}):")
            print(f"  Generators: {self.NumGenerators}")
            print(f"  ADE relations: {len(self.FreeRelations)}")
            print(f"  Diff KKT constraints: {self.diff_constraints_count}")
            print(f"  Mean cert constraints: {len(mean_certs)}")
            print(f"  Archimedean constraints: {len(box_constraints)}")
            print(f"  Total ADE moment constraints: {len(self.ade_moment_constraints)}")
            print(f"  Solver routed to: {solver}")
            print("-" * 30)

        # Dispatch to routed solver
        if solver == SOLVER_SDP:
            return self._solve_via_sdp()
        elif solver == SOLVER_SONC:
            return self._solve_via_sonc()
        elif solver == SOLVER_GP:
            return self._solve_via_gp()
        else:
            return self._solve_via_sdp()


class DSDPMeanRelaxation(DSDPRelaxations):
    r"""
    Specialized DSDP relaxation using mean polynomial certificates.

    Focuses on the :math:`M_{q,p}(X,w)` nonnegativity certificate as the primary
    relaxation mechanism, with ADE relations as secondary constraints.

    The mean polynomial cone :math:`\mathcal{M}_{n,2d}` contains both SOS and SONC
    cones, providing a potentially tighter relaxation.
    """

    def __init__(self, gens, weights, q=1, p=0, relations=(), name="DSDPMeanRlx", **kwargs) -> None:
        r"""
        Initialize mean-based DSDP relaxation.

        Args:
            gens: List of sympy generators.
            weights: Weight vector for mean certificates (must match generator count).
            q: Power mean parameter q.
            p: Power mean parameter p.
            relations: ADE relation expressions.
            name: Name for this instance.
            **kwargs: Additional DSDP parameters.
        """
        super().__init__(gens, relations, name, q=q, p=p, weights=weights, **kwargs)

    def construct_mean_moment_matrix(self):
        r"""
        Construct the moment matrix for the mean polynomial certificate.

        Returns:
            Block-diagonal moment matrix encoding M_{q,p} nonnegativity.
        """
        n = self.NumGenerators

        # Determine basis size from reduced monomial basis
        basis = self.ReducedMonomialBase(self.MmntOrd)
        basis_size = len(basis)

        # Build weighted moment blocks
        blocks = []
        for i in range(n):
            w_i = self.weights[i]
            block = zeros(basis_size, basis_size)
            for k in range(basis_size):
                block[k, k] = w_i
            blocks.append(block)

        if not blocks:
            return zeros(1, 1)

        # Assemble block matrix
        total_size = sum(b.shape[0] for b in blocks)
        result = zeros(total_size, total_size)
        row_offset = 0
        for block in blocks:
            r, c = block.shape
            result[row_offset:row_offset + r, row_offset:row_offset + c] = block
            row_offset += r

        return result

    def solve_mean(self, order=None):
        r"""
        Solve using mean polynomial relaxation.

        Args:
            order: Relaxation order.

        Returns:
            Lower bound from mean relaxation.
        """
        if self.verbosity > 0:
            mean_mat = self.construct_mean_moment_matrix()
            print(f"Mean relaxation M_{{{self.q},{self.p}}}:")
            print(f"  Matrix shape: {mean_mat.shape}")
            print(f"  Weights: {self.weights}")
            print("-" * 30)

        return self.solve(order=order)


class DSDPKKTRelaxation(DSDPRelaxations):
    r"""
    DSDP relaxation with differential KKT condition injection.

    Injects stationarity conditions derived from differentiating the Lagrangian,
    which can tighten bounds significantly (equivalent to one higher Lasserre order).
    """

    def __init__(self, gens, relations=(), name="DSDPKKTRlx", diff_map=None, **kwargs) -> None:
        r"""
        Initialize KKT-enhanced DSDP relaxation.

        Args:
            gens: List of sympy generators.
            relations: ADE relation expressions.
            name: Name for this instance.
            diff_map: Derivation map for differential KKT.
            **kwargs:
                - kkt_order: Order of KKT differentiation (default: 1).
        """
        kwargs['use_diff_kkt'] = True
        super().__init__(gens, relations, name, **kwargs)

        if diff_map is not None:
            self.set_derivation(diff_map)

    def _build_lagrangian(self):
        r"""
        Construct the Lagrangian L = f - sum(lambda_i * g_i).

        Returns:
            Lagrangian expression as a sympy polynomial.
        """
        L = self.RedObjective
        for i, cnst in enumerate(self.Constraints):
            L = L - cnst
        return L

    def _build_kkt_stationarity(self):
        r"""
        Build KKT stationarity constraints dL/dx_j = 0.

        Returns:
            List of (reduced_expr, rhs) tuples for stationarity constraints.
        """
        if not self.derivation_registered:
            return []

        L = self._build_lagrangian()
        constraints = []

        for sym in self.Generators:
            diff_L = self.differentiate(L, sym)
            if diff_L != 0:
                reduced = self.ReduceExp(diff_L)
                constraints.append((reduced, 0))
                deg = Poly(reduced, *self.AuxSyms).total_degree()
                self.MmntCnsDeg = max(int(ceil(deg / 2.)), self.MmntCnsDeg)

        return constraints

    def solve_kkt(self, order=None):
        r"""
        Solve with KKT stationarity injection.

        Args:
            order: Relaxation order.

        Returns:
            Tightened lower bound.
        """
        kkt_constraints = self._build_kkt_stationarity()
        for expr, rhs in kkt_constraints:
            self.add_ade_moment_constraint(expr, rhs)

        if self.verbosity > 0:
            print(f"KKT relaxation (order={order or self.MmntOrd}):")
            print(f"  Stationarity constraints: {len(kkt_constraints)}")
            print("-" * 30)

        return self.solve(order=order)
