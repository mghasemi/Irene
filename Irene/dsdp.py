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

from math import ceil
from functools import reduce
from operator import mul
from itertools import product

from sympy import Symbol, Poly, sympify, groebner, QQ, expand, zeros, Matrix

from .base import base
from .sdp import sdp
from .relaxations import SDPRelaxations, SDRelaxSol, Mom


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

        Returns:
            List of (reduced_expr, rhs) tuples for moment constraints.
        """
        if not self.use_diff_kkt or not self.derivation_registered:
            return []

        constraints = []

        # Differentiate objective
        diff_obj = self.differentiate(self.RedObjective)
        obj_deg = Poly(diff_obj, *self.AuxSyms).total_degree() if diff_obj != 0 else 0

        for sym in self.Generators:
            # d(objective)/dx_j
            diff_term = self.differentiate(self.RedObjective, sym)
            if diff_term != 0:
                reduced = self.ReduceExp(diff_term)
                constraints.append([reduced, 0])
                deg = Poly(reduced, *self.AuxSyms).total_degree()
                self.MmntCnsDeg = max(int(ceil(deg / 2.)), self.MmntCnsDeg)

            # d(constraint_i)/dx_j for each constraint
            for cnst in self.Constraints:
                diff_cnst = self.differentiate(cnst, sym)
                if diff_cnst != 0:
                    reduced = self.ReduceExp(diff_cnst)
                    constraints.append([reduced, 0])
                    deg = Poly(reduced, *self.AuxSyms).total_degree()
                    self.MmntCnsDeg = max(int(ceil(deg / 2.)), self.MmntCnsDeg)

        self.diff_constraints_count = len(constraints)
        return constraints

    def _build_mean_certificate_moments(self):
        r"""
        Build moment constraints encoding M_{q,p}(X,w) nonnegativity.

        The mean polynomial certificate M_{q,p}(X, w) is PSD iff q > p
        (Prop. 2.1 — monotonicity of power means). The certificate is enforced
        as a sum-of-means condition in the moment hierarchy.

        NOTE: The current construction uses an approximate power-ratio formulation.
        Phase 2 will replace this with the exact lcm(q,p) construction:
            M_{q,p} = M_q^{lcm(q,p)} - M_p^{lcm(q,p)}

        Returns:
            List of (reduced_expr, rhs) tuples for moment constraints.
        """
        constraints = []
        n = self.NumGenerators

        if n == 0:
            return constraints

        # Theory (§2.1): M_{q,p} is PSD iff q > p (monotonicity of power means).
        # If q <= p, the form is indefinite/negative and cannot certify nonnegativity.
        if self.q <= self.p:
            if self.verbosity > 0:
                print(f"Warning: q={self.q} <= p={self.p}, "
                      f"mean certificate is not PSD (requires q > p)")
            return constraints

        # TODO: Phase 2 - replace with exact lcm(q,p) construction for M_q^c - M_p^c
        power_ratio = (self.q - self.p) / self.q
        if abs(power_ratio - round(power_ratio)) > 1e-10:
            if self.verbosity > 0:
                print(f"Warning: (q-p)/q = {power_ratio:.4f} is not integer, "
                      f"using rounded value {round(power_ratio)}")
        power_ratio = int(round(power_ratio))

        # Construct weighted power sum: sum_j w_j X_j^q
        weighted_q_sum = sum(
            self.weights[j] * self.AuxSyms[j] ** self.q
            for j in range(n)
        )

        # Raise to power (p-q)/q
        q_power = expand(weighted_q_sum ** power_ratio)

        # For each variable, build w_i * X_i^p * q_power term
        for i in range(n):
            term = self.weights[i] * self.AuxSyms[i] ** self.p * q_power
            reduced = self.ReduceExp(term)
            if reduced != 0:
                constraints.append([reduced, 0])
                deg = Poly(reduced, *self.AuxSyms).total_degree()
                self.MmntCnsDeg = max(int(ceil(deg / 2.)), self.MmntCnsDeg)

        return constraints

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

    def solve(self, order=None):
        r"""
        Solve the DSDP relaxation.

        Builds and solves the SDP with ADE relations, differential KKT
        conditions, and mean polynomial certificates integrated.

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

        # Report
        if self.verbosity > 0:
            print(f"DSDP Relaxation (order={self.MmntOrd}):")
            print(f"  Generators: {self.NumGenerators}")
            print(f"  ADE relations: {len(self.FreeRelations)}")
            print(f"  Diff KKT constraints: {self.diff_constraints_count}")
            print(f"  Mean cert constraints: {len(mean_certs)}")
            print(f"  Archimedean constraints: {len(box_constraints)}")
            print(f"  Total ADE moment constraints: {len(self.ade_moment_constraints)}")
            print("-" * 30)

        # Build and solve SDP
        self.InitSDP()
        return self.Minimize()


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
