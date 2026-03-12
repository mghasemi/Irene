from collections import OrderedDict
from math import ceil
from typing import Any, Optional, Sequence

import numpy as np
from scipy import optimize
from scipy.spatial import ConvexHull, Delaunay
from sympy import sympify, Symbol
 
from .grouprings import _degree, SemigroupAlgebraElement, SemigroupAlgebra, CommutativeSemigroup, AtomicSGElement


class OptimizationProblem(object):
    """
    This class represents an optimization problem.

    Attributes:
        sga (SemigroupAlgebra): The semigroup algebra of the optimization problem.
        relations (list[SemigroupAlgebraElement]): The relations of the optimization problem.
        semigroup (CommutativeSemigroup): The semigroup of the optimization problem.
        objective (SemigroupAlgebraElement): The objective function of the optimization problem.
        constraints (list[SemigroupAlgebraElement]): The constraints of the optimization problem.
        objective_degree (int): The degree of the objective function.
        objective_half_degree (int): The half degree of the objective function.
        constraints_degree (list[int]): The degrees of the constraints.
        constraints_half_degree (list[int]): The half degrees of the constraints.
        objective_trms_with_positive_coefficient (list[SemigroupAlgebraElement]): The terms of the objective function with positive coefficients.
        objective_trms_with_negative_coefficient (list[SemigroupAlgebraElement]): The terms of the objective function with negative coefficients.
        objective_terms_with_even_exponent (list[SemigroupAlgebraElement]): The terms of the objective function with even exponents.
        objective_terms_with_odd_exponent (list[SemigroupAlgebraElement]): The terms of the objective function with odd exponents.
        constraint_trms_with_positive_coefficient (list[SemigroupAlgebraElement]): The terms of the constraints with positive coefficients.
        constraint_trms_with_negative_coefficient (list[SemigroupAlgebraElement]): The terms of the constraints with negative coefficients.
        constraint_terms_with_even_exponent (list[SemigroupAlgebraElement]): The terms of the constraints with even exponents.
        constraint_terms_with_odd_exponent (list[SemigroupAlgebraElement]): The terms of the constraints with odd exponents.
        total_degree (int): The total degree of the optimization problem.
    """

    def __init__(self, sga: Optional[SemigroupAlgebra] = None,
                 relations: Optional[list[SemigroupAlgebraElement]] = None) -> None:
        self.sga: SemigroupAlgebra = sga
        self.relations = relations
        self.semigroup: CommutativeSemigroup = CommutativeSemigroup([])
        if self.sga is not None:
            self.semigroup = self.sga.semigroup
        self.objective = None
        self.constraints = list()
        self.objective_degree = 0
        self.objective_half_degree = 0
        self.constraints_degree = list()
        self.constraints_half_degree = list()
        self.objective_trms_with_positive_coefficient = list()
        self.objective_trms_with_negative_coefficient = list()
        self.objective_terms_with_even_exponent = list()
        self.objective_terms_with_odd_exponent = list()
        self.constraint_trms_with_positive_coefficient = list()
        self.constraint_trms_with_negative_coefficient = list()
        self.constraint_terms_with_even_exponent = list()
        self.constraint_terms_with_odd_exponent = list()
        self.total_degree = 2
        self.newton_polytope = None
        self.vertices = None

    def set_objective(self, obj: SemigroupAlgebraElement) -> None:
        """
        Sets the objective function for the optimization problem.
        
        This method is central to defining the optimization problem. It assigns the objective
        function that will be minimized or maximized, and automatically updates the semigroup
        and computes the degree information needed for polynomial optimization.
        
        Place in code structure: Core setup method that must be called before relaxation
        generation. Works with add_constraints() to fully specify the optimization problem.
        
        Args:
            obj (SemigroupAlgebraElement): A polynomial expression to be optimized, represented
                as an element in a semigroup algebra.
        
        Returns:
            None
        
        Side effects:
            - Updates self.objective with the provided expression
            - Updates self.semigroup if different from current
            - Computes and stores objective_degree (total degree of the leading monomial)
            - Computes and stores objective_half_degree (ceiling of degree/2)
        """
        self.objective = obj
        if self.semigroup != obj.semigroup:
            self.semigroup = obj.semigroup
        self.objective_degree = _degree(obj.LM())
        self.objective_half_degree = int(ceil(self.objective_degree / 2.))

    def add_constraints(self, const: list[SemigroupAlgebraElement]) -> None:
        """
        Adds polynomial inequality or equality constraints to the optimization problem.
        
        This method extends the optimization problem with additional constraints, typically
        representing feasibility conditions like bounds or domain restrictions. Each constraint
        is analyzed for its degree to assist in later relaxation construction.
        
        Place in code structure: Complements set_objective() to fully define the constrained
        optimization problem. Called during problem setup before relaxation methods.
        
        Args:
            const (list[SemigroupAlgebraElement]): List of polynomial constraint expressions,
                each represented as a SemigroupAlgebraElement.
        
        Returns:
            None
        
        Side effects:
            - Appends each constraint to self.constraints
            - Updates self.semigroup if any constraint uses a different semigroup
            - Computes and stores degree and half-degree for each constraint
        """
        for exp in const:
            if self.semigroup != exp.semigroup:
                self.semigroup = exp.semigroup
            self.constraints.append(exp)
            exp_deg = _degree(exp.LM())
            exp_half_deg = int(ceil(exp_deg / 2.))
            self.constraints_degree.append(exp_deg)
            self.constraints_half_degree.append(exp_half_deg)

    def program_degree(self) -> int:
        """
        Computes the overall degree of the optimization problem.
        
        This method determines the maximum degree among all polynomials (objective and constraints)
        in the problem, rounded up to the nearest even number. This degree is critical for
        determining the size of sum-of-squares (SOS) relaxations.
        
        Place in code structure: Utility method used during relaxation construction to determine
        the appropriate degree for SOS decompositions and moment matrices.
        
        Returns:
            int: The maximum degree among all polynomials, adjusted to be even (adds 1 if odd).
                This ensures compatibility with SOS relaxations which require even degrees.
        """
        degs = self.constraints_degree + [self.objective_degree]
        dg = max(degs)
        if dg % 2 == 0:
            return dg
        return dg + 1

    def analyse_program(self) -> None:
        """
        Performs structural analysis of the optimization problem's polynomial terms.
        
        This method categorizes all terms from the objective function and constraints based on
        two key properties: (1) sign of coefficients, and (2) parity of exponents. This analysis
        is essential for specialized polynomial optimization techniques like SONC (sums of
        non-negative circuit polynomials) and conditional relaxations.
        
        Place in code structure: Called after problem setup to preprocess the polynomial structure.
        Populates classification lists used by relaxation algorithms (relaxations.py) and
        decomposition methods (sonc.py).
        
        Returns:
            None
        
        Side effects:
            Populates the following attributes:
            - objective_trms_with_positive/negative_coefficient: Objective terms by sign
            - constraint_trms_with_positive/negative_coefficient: Constraint terms by sign
            - objective_terms_with_even/odd_exponent: Objective terms by exponent parity
            - constraint_terms_with_even/odd_exponent: Constraint terms by exponent parity
        """
        # Reset analysis buckets so repeated calls do not accumulate stale state.
        self.objective_trms_with_positive_coefficient = []
        self.objective_trms_with_negative_coefficient = []
        self.objective_terms_with_even_exponent = []
        self.objective_terms_with_odd_exponent = []
        self.constraint_trms_with_positive_coefficient = []
        self.constraint_trms_with_negative_coefficient = []
        self.constraint_terms_with_even_exponent = []
        self.constraint_terms_with_odd_exponent = []

        # Separate terms of objective and constraints based on the sign of their coefficients
        for trm in self.objective.content:
            if trm[0] > 0:
                self.objective_trms_with_positive_coefficient.append(trm)
            else:
                self.objective_trms_with_negative_coefficient.append(trm)
            if self.square_exponent(trm[1]):
                self.objective_terms_with_even_exponent.append(trm)
            else:
                self.objective_terms_with_odd_exponent.append(trm)
        # Separate terms of objective and constraints based on the exponents of the semigroup content
        for xprsn in self.constraints:
            for trm in xprsn.content:
                if trm[0] > 0:
                    self.constraint_trms_with_positive_coefficient.append(trm)
                else:
                    self.constraint_trms_with_negative_coefficient.append(trm)
                if self.square_exponent(trm[1]):
                    self.constraint_terms_with_even_exponent.append(trm)
                else:
                    self.constraint_terms_with_odd_exponent.append(trm)

    @staticmethod
    def square_exponent(xpnt: Any) -> bool:
        """
        Determines if a monomial has only even exponents (is a perfect square).
        
        This method checks whether all variables in a monomial appear with even powers,
        meaning the monomial can be expressed as a square of another monomial. This property
        is fundamental for identifying terms that are inherently non-negative and for
        constructing sum-of-squares decompositions.
        
        Place in code structure: Static utility method used throughout analysis and relaxation
        construction. Called by analyse_program(), delta(), and other methods that need to
        identify square monomials for optimization techniques.
        
        Args:
            xpnt (SemigroupAlgebraElement): A monomial whose exponents are to be checked.
        
        Returns:
            bool: True if all exponents in the monomial are even (divisible by 2),
                False if any exponent is odd.
        """
        for _ in xpnt.array_form:
            if _[1] % 2 != 0:
                return False
        return True

    @staticmethod
    def has_symbol(symb: str, mono: SemigroupAlgebraElement) -> tuple[bool, int]:
        """
        Searches for a specific variable (symbol) in a monomial and returns its position.
        
        This method determines whether a given variable name appears in a monomial's factorization,
        which is useful for variable-specific operations like substitution, elimination, or
        targeted relaxation strategies.
        
        Place in code structure: Static utility method for monomial inspection. Used by various
        algorithms that need to check variable presence or extract variable-specific information
        from polynomial terms.
        
        Args:
            symb (str): The name of the variable/symbol to search for (e.g., 'x', 'y').
            mono (SemigroupAlgebraElement): The monomial to search within.
        
        Returns:
            tuple[bool, int]: A tuple where the first element is True if the symbol is found,
                False otherwise. The second element is the index of the symbol in the monomial's
                array form if found, or -1 if not found.
        """
        idx = 0
        for _ in mono.array_form:
            if symb == _[0].name:
                return True, idx
            idx += 1
        return False, -1

    @staticmethod
    def omega(xprsn: SemigroupAlgebraElement, deg: int) -> list[tuple[float, Any]]:  # Add the term for 0
        """
        Extracts terms from a polynomial that don't match a specific univariate degree pattern.
        
        The omega function filters out univariate monomials of exactly the specified degree,
        keeping all multivariate terms, constant terms, and univariate terms of other degrees.
        This is used in conditional optimization where certain degree-specific terms need
        special treatment.
        
        Place in code structure: Static helper method used by delta() to identify terms that
        require special handling in relaxation construction, particularly for conditional
        constraints and geometric programming.
        
        Args:
            xprsn (SemigroupAlgebraElement): The polynomial expression to filter.
            deg (int): The specific univariate degree to exclude.

        Returns:
            list[tuple[int, SemigroupAlgebraElement]]: List of (coefficient, monomial) tuples
                representing terms that don't match the degree criterion.
        """
        terms = list()
        for trm in xprsn.content:
            if len(trm[1].array_form) > 1:
                terms.append(trm)
                continue
            if not trm[1].array_form:
                continue
            if trm[1].array_form[0][1] == deg:
                continue
            else:
                terms.append(trm)
        return terms

    def delta(self, xprsn: SemigroupAlgebraElement, deg: int) -> dict[str, set]:
        """
        Identifies problematic terms in polynomial expressions for relaxation construction.
        
        The delta function categorizes terms that cannot be directly represented in a
        sum-of-squares (SOS) form: terms with odd exponents or negative coefficients.
        These terms are grouped by whether their total degree equals the target degree ('=d')
        or is less than it ('<d'), which affects how they're handled in relaxations.
        
        Place in code structure: Core analysis method used by relaxation algorithms
        (relaxations.py) to identify which terms need special treatment through auxiliary
        variables or alternative decomposition strategies.
        
        Args:
            xprsn (SemigroupAlgebraElement): The polynomial expression to analyze.
            deg (int): The target degree for comparison.

        Returns:
            dict[str, set[SemigroupAlgebraElement]]: Dictionary with two keys:
                '=d': Set of monomials with degree exactly equal to deg
                '<d': Set of monomials with degree less than deg
                Both contain only non-square or negative coefficient terms.
        """
        terms = {'=d': set([]), '<d': set([])}
        omega = self.omega(xprsn, deg)
        for trm in omega:
            if not self.square_exponent(trm[1]) or trm[0] < 0:
                if self.semigroup.degree(trm[1]) == deg:
                    terms['=d'].add(trm[1])
                else:
                    terms['<d'].add(trm[1])
        return terms

    def delta_vertex(self, xprsn: SemigroupAlgebraElement, vertices: list) -> None:
        """
        Computes delta relative to Newton polytope vertices (currently unimplemented).
        
        This method is intended to analyze terms based on their relationship to the vertices
        of the Newton polytope, likely for geometric programming or sparsity exploitation.
        
        Place in code structure: Placeholder for advanced geometric analysis that would work
        with the newton() and related polytope methods for exploiting problem structure.
        
        Args:
            xprsn (SemigroupAlgebraElement): The expression to analyze.
            vertices (list): List of vertices from the Newton polytope.

        Returns:
            None (not yet implemented)
        """
        raise NotImplementedError("delta_vertex is not implemented yet")

    def mono2ord_tuple(self, mono: Any) -> tuple[int, ...]:
        """
        Converts a monomial from semigroup representation to a tuple of exponents.
        
        This method transforms a symbolic monomial into a numeric tuple where each position
        corresponds to a generator in the semigroup, containing that generator's exponent.
        This standardized format is essential for numerical computations like convex hull
        calculations and geometric analysis.
        
        Place in code structure: Conversion utility used by newton() and related geometric
        methods to transform symbolic polynomial data into numeric arrays suitable for
        scipy geometric algorithms.
        
        Args:
            mono (SemigroupAlgebraElement): The monomial to convert. Can also be a scalar
                (int or float) representing a constant term.

        Returns:
            tuple: A tuple of integers representing the exponents of each generator in order.
                For constants, returns a tuple of zeros.
        """
        n = len(self.semigroup.generators)
        if isinstance(mono, (int, float)):
            return tuple([0] * n)

        if isinstance(mono, AtomicSGElement):
            array_form = mono.content[0][1].array_form
        elif isinstance(mono, SemigroupAlgebraElement):
            if not mono.content:
                return tuple([0] * n)
            if len(mono.content) > 1:
                raise ValueError("mono2ord_tuple expects a monomial (single-term expression)")
            array_form = mono.content[0][1].array_form
        elif hasattr(mono, 'array_form'):
            array_form = mono.array_form
        else:
            raise TypeError("mono2ord_tuple expects a scalar, monomial, or semigroup expression")

        od_mono = OrderedDict(array_form)
        xpnt = []
        for gen in self.semigroup.generators:
            xpnt.append(od_mono.get(gen.ext_rep[0], 0))
        return tuple(xpnt)

    def tuple2mono(self, xpnt: Sequence[int]):
        """
        Converts a tuple of exponents back to a monomial in semigroup representation.
        
        This method performs the inverse operation of mono2ord_tuple, constructing a
        symbolic monomial from numeric exponents. Each position in the input tuple
        corresponds to the power of the respective generator in the semigroup.
        
        Place in code structure: Inverse conversion utility that works with mono2ord_tuple
        to enable round-trip conversions between symbolic and numeric representations.
        Used when translating geometric analysis results back to polynomial terms.
        
        Args:
            xpnt: A sequence (tuple or list) of integers representing exponents for each
                generator in the semigroup, in the same order as semigroup.generators.

        Returns:
            SemigroupAlgebraElement: The monomial product of generators raised to their
                corresponding powers from xpnt.
        """
        idx = 0
        elm = self.semigroup.identity()
        for s in self.semigroup.generators:
            elm = elm * (s**xpnt[idx])
            idx += 1
        return elm

    def newton(self) -> None:
        """
        Computes the Newton polytope of the optimization problem.
        
        The Newton polytope is the convex hull of all exponent vectors appearing in the
        objective function and constraints. This geometric object encodes the sparsity
        structure of the problem and can be exploited to construct more efficient relaxations.
        
        Place in code structure: Geometric analysis method that should be called after problem
        setup. Its results (stored in newton_polytope and vertices) are used by in_newton(),
        convex_combination(), and potentially specialized relaxation strategies.

        Returns:
            None

        Side effects:
            - Collects all exponent tuples from objective and constraints
            - Computes and stores the convex hull in self.newton_polytope
            - Extracts and stores sorted vertex list in self.vertices
        """
        exponents = set([])
        for _ in self.objective.content:
            exponents.add(self.mono2ord_tuple(SemigroupAlgebraElement([_], self.semigroup)))
        for g in self.constraints:
            for _ in g.content:
                exponents.add(self.mono2ord_tuple(SemigroupAlgebraElement([_], self.semigroup)))
        points = np.array([_ for _ in exponents])
        self.newton_polytope = ConvexHull(points)
        self.vertices = [list(_) for _ in points[self.newton_polytope.vertices]]
        self.vertices.sort(reverse=False)

    def in_newton(self, point: Sequence[float]) -> bool:
        """
        Tests whether a point lies inside the Newton polytope.
        
        This method uses Delaunay triangulation to efficiently determine if a given point
        (typically representing a monomial's exponents) is within the convex hull of the
        problem's support. This is useful for determining which monomials are representable
        as convex combinations of existing terms.
        
        Place in code structure: Geometric query method that depends on newton() being called
        first. Used in sparsity analysis and to determine valid monomial spaces for relaxations.
        
        Args:
            point: A numeric sequence (tuple, list, or array) representing exponent coordinates
                to test for inclusion in the Newton polytope.

        Returns:
            bool: True if the point is inside or on the boundary of the Newton polytope,
                False otherwise.
        """
        tri = Delaunay(self.vertices)
        return tri.find_simplex(point) >= 0

    def linear_combination(self, point: Sequence[float]) -> np.ndarray:
        """
        Expresses a point as a linear combination of Newton polytope vertices.
        
        This method solves a linear system to find coefficients that express the given point
        as a linear combination of polytope vertices. Note that unlike convex_combination(),
        this method doesn't enforce non-negativity or that coefficients sum to 1, so it's
        suitable for affine (not necessarily convex) combinations.
        
        Place in code structure: Geometric utility for expressing points in terms of vertices.
        Complements convex_combination() but with less restrictive constraints. Used when
        affine relationships are needed rather than strictly convex ones.
        
        Args:
            point: A numeric sequence representing coordinates to express in terms of vertices.

        Returns:
            numpy.ndarray: Array of coefficients for the linear combination. Note that these
                coefficients may be negative or sum to values other than 1.

        Note:
            Excludes the origin vertex [0, 0, ...] if present to avoid singularity issues.
        """
        if not self.vertices:
            raise ValueError("linear_combination requires non-empty vertices; call newton() first.")

        origin = [0] * len(self.semigroup.generators)
        if self.vertices[0] == origin:
            active_vertices = self.vertices[1:]
        else:
            active_vertices = self.vertices

        if not active_vertices:
            raise ValueError("linear_combination requires at least one non-origin vertex.")

        A = np.array(active_vertices, dtype=float).T
        point_arr = np.asarray(point, dtype=float)

        if point_arr.ndim != 1:
            raise ValueError("linear_combination expects a one-dimensional point.")
        if A.shape[0] != point_arr.shape[0]:
            raise ValueError(
                f"linear_combination point dimension mismatch: expected {A.shape[0]}, got {point_arr.shape[0]}."
            )
        if A.shape[0] != A.shape[1]:
            raise ValueError(
                f"linear_combination requires a square vertex matrix; got shape {A.shape}."
            )

        try:
            coeffs = np.linalg.solve(A, point_arr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("linear_combination failed due to singular vertex matrix.") from exc
        return coeffs

    def convex_combination(self, point: Sequence[float]) -> Optional[np.ndarray]:
        """
        Finds the representation of a point as a convex combination of Newton polytope vertices.
        
        This method formulates and solves a linear program to express a point as a convex
        combination of the polytope vertices, enforcing both non-negativity of coefficients
        and that they sum to 1. This is more restrictive than linear_combination() but
        guarantees the point lies within the convex hull.
        
        Place in code structure: Advanced geometric utility for Newton polytope analysis.
        Uses scipy.optimize to solve the constrained optimization problem. Results can guide
        sparse relaxation strategies by identifying how to express new terms using existing ones.
        
        Args:
            point: A numeric sequence representing coordinates in the exponent space that should
                be expressed as a convex combination of self.vertices.

        Returns:
            numpy.ndarray: Array of non-negative coefficients summing to 1 that express the point
                as a convex combination of vertices, or None if no such combination exists
                (i.e., point is outside the Newton polytope).
        """
        np_vertices = np.array(self.vertices)
        A_eq = np_vertices.T  # Transpose vertices for the equality constraint
        b_eq = point

        # Inequality constraints: coefficients >= 0
        A_ub = -np.identity(np_vertices.shape[0])
        b_ub = np.zeros(np_vertices.shape[0])

        # Equality constraint: sum of coefficients = 1
        additional_eq_constraint = np.ones((1, np_vertices.shape[0]))
        A_eq = np.vstack([A_eq, additional_eq_constraint])
        b_eq = np.append(b_eq, 1)

        # Solve the linear program
        result = optimize.linprog(
            c=np.zeros(np_vertices.shape[0]),  # Dummy objective function, we only care about feasibility
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=(0, None),  # Coefficients must be non-negative
            method='highs'
        )

        if result.success:
            return result.x
        else:
            return None

    def to_sympy(self, expr: SemigroupAlgebraElement, sym_map: dict[str, Symbol]):
        """
        Converts a SemigroupAlgebraElement polynomial to a SymPy symbolic expression.
        
        This method provides interoperability with the SymPy computer algebra system,
        enabling use of SymPy's extensive symbolic manipulation capabilities (differentiation,
        integration, simplification, etc.) on polynomials defined in the semigroup algebra
        framework.
        
        Place in code structure: Interface method for exporting to SymPy. Useful for
        visualization, symbolic analysis, or interfacing with other tools that expect
        SymPy expressions. Complements the internal semigroup algebra representation.
        
        Args:
            expr (SemigroupAlgebraElement): The polynomial expression to convert.
            sym_map (dict): Dictionary mapping generator name strings to SymPy Symbol objects,
                e.g., {'x': Symbol('x'), 'y': Symbol('y')}.

        Returns:
            sympy.Expr: A SymPy expression algebraically equivalent to the input, using the
                symbols provided in sym_map.
        """
        sympy_expr = sympify(0)
        for coeff, mono in expr.content:
            term = sympify(coeff)
            if not mono.array_form:  # constant term
                sympy_expr += term
                continue
            for gen, exp in mono.array_form:
                sym = sym_map.get(gen.name)
                if sym is None:
                    raise KeyError(f"Missing symbol mapping for '{gen.name}'")
                term *= (sym ** exp)
            sympy_expr += term
        return sympy_expr
