"""Newton polytope monomial pruning for moment matrix dimension reduction.

For a polynomial optimization problem, the Newton polytopes of the objective
and constraints identify exponent candidates that are *potentially* relevant:
an exponent outside ``2\\cdotNewt(f)`` can never occur as a term of ``g*h`` with
``supp(g), supp(h)`` inside the Minkowski-sum support, so it is never needed in
the objective coefficient vector.  Filtering on this hull reduces the moment
matrix size -- sometimes by orders of magnitude for sparse problems.

Safety contract: hull membership alone is *not* a certificate that an exponent
may be removed from the moment/Gram basis of a Lasserre SDP.  Removing exponents
drops PSD coupling entries and can introduce recession directions (or force
invalid objective values) even when every objective-support exponent is kept;
the Motzkin polynomial at order 3 is such a counterexample (the pruned SDP has
an unbounded rank-one recession ray while the full one is bounded).  Until a
complete SOS-support certificate exists, callers must pass ``required_support``
with the exponents that must be retained regardless of hull membership; the
certified safe choice at present is the full degree-bounded support.

Key references:
    - Parrilo (2000), "Structured Semidefinite Programs and Semialgebraic Geometry"
    - Lasserre (2006), "Moments, Positive Polynomials and Their Applications"
    - Kim & Kojima (2014), "Sparse SOS decompositions via Newton polytopes"

The implementation computes the Minkowski sum of scaled Newton polytopes,
then filters the full monomial basis to those inside the hull, always retaining
``required_support``.
"""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from itertools import product as iter_product


def _coerce_to_sympy(expr):
    """Normalize supported polynomial inputs to a SymPy expression.

    Accepts scalars, SymPy expressions, and semigroup-algebra elements.  Ring
    elements are converted term-by-term with the same rule as
    ``OptimizationProblem.to_sympy`` (Irene/program.py): each monomial is
    rebuilt from its generator names, so all elements of one algebra map to
    expressions in the same symbols and therefore consistent exponent columns.

    Args:
        expr: A scalar, SymPy expression, or SemigroupAlgebra element.

    Returns:
        sympy.Expr (or the original scalar).

    Raises:
        TypeError: If ``expr`` is none of the supported types.  Callers must
            not receive a silently degraded Newton polytope for input this
            module cannot understand.
    """
    import sympy as sp

    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, sp.Basic):
        return expr

    content = getattr(expr, "content", None)
    semigroup = getattr(expr, "semigroup", None)
    if content is not None and semigroup is not None:
        sym_map = {}
        sympy_expr = sp.sympify(0)
        for coeff, mono in content:
            term = sp.sympify(coeff)
            if not mono.array_form:  # identity monomial (constant term)
                sympy_expr += term
                continue
            for gen, expn in mono.array_form:
                sym = sym_map.get(gen.name)
                if sym is None:
                    sym = sp.Symbol(gen.name)
                    sym_map[gen.name] = sym
                term *= sym ** expn
            sympy_expr += term
        return sympy_expr

    raise TypeError(
        "newton_polytope() expects a scalar, a SymPy expression, or a "
        f"SemigroupAlgebra element; got {type(expr).__name__!r}. Convert the "
        "input before calling (e.g. via OptimizationProblem.to_sympy) so that "
        "a parse failure is reported here instead of producing a degenerate "
        "Newton polytope."
    )


def newton_polytope(expr, vars_list=None):
    """Compute the Newton polytope of a polynomial expression.

    The Newton polytope is the convex hull of exponent vectors of all terms
    with nonzero coefficients in the polynomial.

    Args:
        expr: A symbolic polynomial expression (SymPy or SymEngine), a scalar,
            or a SemigroupAlgebra element (converted to SymPy automatically).
        vars_list: List of variables to extract exponents for. If None, inferred.

    Returns:
        np.ndarray: Array of shape (num_terms, num_vars) of exponent vectors.

    Raises:
        ValueError: If the input cannot be parsed as a polynomial.  This
            function never returns a silently degraded (origin-only) polytope;
            unparseable input is an explicit error instead.
    """
    from .symbolic_engine import engine

    expr = _coerce_to_sympy(expr)

    # Scalar constants: the Newton polytope of any constant is exactly {origin}.
    # Use this function's 0-variable convention (same shape as the constant path
    # below); combined_newton_polytope() skips 0-column rows when combining.
    # This is a correct result, not a fallback -- genuinely unparseable input
    # still raises further down.
    import sympy as sp
    if not isinstance(expr, sp.Basic) or expr.is_number:
        return np.zeros((1, 0), dtype=int)

    try:
        poly = engine.Poly(expr)
    except Exception as exc:
        raise ValueError(
            f"newton_polytope() could not parse {type(expr).__name__!r} "
            f"(vars={list(map(str, vars_list)) if vars_list is not None else 'inferred'}): "
            f"{exc}. Pass a SymPy polynomial expression, a SemigroupAlgebra element, "
            "or a scalar."
        ) from exc

    if vars_list is None:
        vars_list = poly.gens

    nvars = len(vars_list)
    # Handle constant polynomials (no generators)
    if nvars == 0:
        return np.zeros((1, 0), dtype=int)

    exp_vectors = []

    for exp_tuple in poly.as_dict().keys():
        # as_dict returns tuple of exponents for each generator
        exp_vectors.append(np.array(exp_tuple, dtype=int))

    if not exp_vectors:
        return np.zeros((nvars,), dtype=int).reshape(1, -1)

    return np.array(exp_vectors)


def minkowski_sum(polytope_a, polytope_b):
    """Compute the Minkowski sum of two point sets.

    A \\oplus B = {a + b | a \\in A, b \\in B}

    Args:
        polytope_a, polytope_b: Arrays of shape (n, d) and (m, d).

    Returns:
        np.ndarray: All pairwise sums, shape (n*m, d). Deduplicated.
    """
    sums = []
    for a in polytope_a:
        for b in polytope_b:
            sums.append(a + b)
    arr = np.array(sums, dtype=int)
    # Deduplicate rows
    _, unique_idx = np.unique(arr, axis=0, return_index=True)
    return arr[unique_idx]


def scale_polytope(polytope, factor):
    """Scale all exponent vectors by an integer factor."""
    return polytope * factor


def combined_newton_polytope(polynomials, vars_list=None):
    """Compute the Minkowski sum of Newton polytopes of multiple polynomials.

    For moment matrix construction, we need 2\\cdot(Newt(f_0) \\oplus Newt(g_1) \\oplus ...),
    where f_0 is the objective and g_i are constraint polynomials.

    Args:
        polynomials: List of symbolic polynomial expressions.
        vars_list: Shared variable list for consistent exponent ordering.

    Returns:
        np.ndarray: Combined Newton polytope vertices (deduplicated).
    """
    if not polynomials:
        return None

    # Get all polytopes.  newton_polytope() raises on input it cannot parse;
    # propagating here keeps unparseable polynomials from being silently
    # dropped (which would shrink the hull without any warning).
    polytopes = []
    for expr in polynomials:
        pts = newton_polytope(expr, vars_list)
        # Skip degenerate polytopes (0 columns = no variables detected), e.g.
        # constant-only terms.
        if pts.shape[1] > 0:
            polytopes.append(pts)

    if not polytopes:
        return None

    # Minkowski sum of all polytopes
    result = polytopes[0]
    for pts in polytopes[1:]:
        result = minkowski_sum(result, pts)

    # Ensure the origin (constant monomial) is included.
    # The constant term 1 = x^0 is always in the moment matrix basis,
    # even when no polynomial explicitly contains a constant term.
    # Without this, polynomials like Choi-Lam whose Newton polytope
    # lacks (0,...,0) would have their entire basis pruned away.
    ncols = result.shape[1]
    include_origin = True
    for row in result:
        if np.all(row == 0):
            include_origin = False
            break
    if include_origin:
        origin = np.zeros((1, ncols), dtype=int)
        result = np.vstack([result, origin])

    # Scale by 2 (for degree-2d moment matrix)
    result = scale_polytope(result, 2)

    return result


class NewtonPruner:
    """Filter monomial basis using Newton polytope pruning.

    Given the combined Newton polytope of an optimization problem's
    polynomials, this class filters the full monomial basis to those exponent
    vectors that lie within the convex hull of ``2\\cdotNewt(f)``, always
    retaining every exponent in ``required_support`` (see module docstring for
    why hull membership alone is not a removal certificate).

    Args:
        num_vars: Number of variables in the problem.
        max_degree: Maximum degree for moment matrix construction.
        polytope_vertices: Precomputed vertices of 2\\cdotcombined Newton polytope.
            If None, will be computed from polynomials later.
        required_support: Exponent tuples that must appear in the returned basis
            regardless of hull membership (degree-bounded to ``max_degree``).
            Pass the full degree-bounded support until a complete SOS-support
            certificate is available; pass a smaller certified set once one exists.

    Attributes:
        full_basis_size: Size of unpruned monomial basis.
        pruned_basis_size: Size after Newton polytope filtering (plus required support).
        reduction_ratio: Fraction of basis retained (lower = more pruning).
    """

    def __init__(self, num_vars: int, max_degree: int,
                 polytope_vertices: Optional[np.ndarray] = None,
                 required_support: Optional[Set[Tuple[int, ...]]] = None):
        self.num_vars = num_vars
        self.max_degree = max_degree
        self.polytope_vertices = polytope_vertices
        self.required_support = set(required_support) if required_support else set()
        self._hull = None
        self.full_basis_size = 0
        self.pruned_basis_size = 0
        self.reduction_ratio = 1.0

    def _build_hull(self):
        """Build the convex hull representation for point-in-polytope testing."""
        if self.polytope_vertices is None:
            return False

        # Reject degenerate polytopes (0 columns = no variables)
        if self.polytope_vertices.shape[1] == 0:
            return False

        # Use scipy if available, otherwise fall back to bounding box check
        try:
            from scipy.spatial import ConvexHull
            if len(self.polytope_vertices) > self.num_vars:
                try:
                    self._hull = ConvexHull(self.polytope_vertices)
                    return True
                except Exception:
                    pass  # Degenerate hull (e.g., collinear points) -> bbox fallback
        except ImportError:
            pass

        # Fallback: use axis-aligned bounding box of polytope vertices
        self._bbox_min = self.polytope_vertices.min(axis=0)
        self._bbox_max = self.polytope_vertices.max(axis=0)
        self._hull = "bbox"
        return True

    def _point_in_polytope(self, point: np.ndarray) -> bool:
        """Check if a point (exponent vector) lies inside the Newton polytope."""
        # Also enforce degree bound
        if int(sum(point)) > self.max_degree:
            return False

        if self._hull is None:
            return True  # No pruning available, include everything

        if self._hull == "bbox":
            return bool(np.all(point >= self._bbox_min) and
                       np.all(point <= self._bbox_max))

        # Full ConvexHull check via half-space inequalities
        try:
            # hull.equations: each row is [normal..., offset], point p inside iff A\\cdotp <= b
            for eq in self._hull.equations:
                normal = eq[:-1]
                offset = -eq[-1]
                if np.dot(normal, point) > offset + 1e-9:
                    return False
            return True
        except Exception:
            return True  # Conservative fallback: include the point

    def compute_pruned_basis(self, vars_list=None):
        """Compute the pruned monomial basis.

        Iterates over all monomials up to ``max_degree``, filters those outside
        the Newton polytope, and then retains every exponent of
        ``required_support`` (degree-bounded) regardless of hull membership.

        Args:
            vars_list: Optional variable list (for compatibility).

        Returns:
            List of exponent tuples forming the pruned basis; always a superset
            of ``{e in required_support : sum(e) <= max_degree}``.
        """
        if self._hull is None:
            self._build_hull()

        # Generate full basis
        all_monos = []
        for exp_tuple in iter_product(range(self.max_degree + 1), repeat=self.num_vars):
            if sum(exp_tuple) <= self.max_degree:
                all_monos.append(np.array(exp_tuple, dtype=int))

        self.full_basis_size = len(all_monos)

        # Filter by polytope membership
        pruned = []
        for exp_vec in all_monos:
            if self._point_in_polytope(exp_vec):
                pruned.append(tuple(exp_vec))

        # Safety contract (see module docstring): always retain the required
        # support.  With the certified full degree-bounded support this makes
        # the result identical to the unpruned basis; with a future certified
        # subset it re-enables genuine reduction without touching this code path.
        pruned_set = set(pruned)
        for expn in sorted(self.required_support):
            if sum(expn) <= self.max_degree and expn not in pruned_set:
                pruned.append(expn)
                pruned_set.add(expn)

        self.pruned_basis_size = len(pruned)
        self.reduction_ratio = self.pruned_basis_size / max(self.full_basis_size, 1)

        # Safety: if pruning eliminated every monomial, fall back to full basis.
        # An empty basis is always pathological -- it means the Newton polytope
        # (even with origin-inclusion) is too tight for the degree bound.
        # Conservative fallback: no pruning is better than zero monomials.
        if self.pruned_basis_size == 0 and self.full_basis_size > 0:
            self.pruned_basis_size = self.full_basis_size
            self.reduction_ratio = 1.0
            pruned = [tuple(np.array(e, dtype=int)) for e in all_monos]

        return pruned

    def moment_matrix_dimension_reduction(self) -> Dict:
        """Estimate the moment matrix size reduction from Newton pruning.

        The moment matrix has dimension R\\timesR where R is the basis size.
        Pruning reduces this to R'\\timesR', so the reduction factor is (R'/R)^2.

        Returns:
            Dict with full_size, pruned_size, matrix_reduction, and savings.
        """
        if self.full_basis_size == 0:
            return {"full_size": 0, "pruned_size": 0, "matrix_reduction": 1.0}

        return {
            "full_basis_size": self.full_basis_size,
            "pruned_basis_size": self.pruned_basis_size,
            "reduction_ratio": round(self.reduction_ratio, 4),
            "matrix_entry_reduction": round(self.reduction_ratio ** 2, 4),
            "entries_saved": self.full_basis_size**2 - self.pruned_basis_size**2,
        }

    def summary(self) -> Dict:
        """Return a human-readable summary of the pruning results."""
        return {
            "num_vars": self.num_vars,
            "max_degree": self.max_degree,
            **self.moment_matrix_dimension_reduction(),
        }


def prune_basis_from_polys(polynomials, num_vars: int, max_degree: int,
                           required_support: Optional[Set[Tuple[int, ...]]] = None) -> NewtonPruner:
    """Convenience function: compute pruned basis from a list of polynomials.

    Args:
        polynomials: List of symbolic polynomial expressions (objective +
            constraints).  SymPy expressions, scalars, and SemigroupAlgebra
            elements are all accepted; ring elements are converted to SymPy
            automatically (see :func:`newton_polytope`).
        num_vars: Number of variables in the problem.
        max_degree: Maximum degree for moment matrix construction (usually 2*d).
        required_support: Exponents that must be retained regardless of hull
            membership (see :class:`NewtonPruner`).  Default ``None`` keeps only
            hull-interior exponents -- callers building relaxation bases should
            pass the certified safe set.

    Returns:
        Configured NewtonPruner with pruned basis computed.

    Raises:
        ValueError: If any polynomial cannot be parsed (no silent fallback).
    """
    # Build a canonical variable list so all polynomials use the same
    # variable ordering and dimension. Without this, polynomials that
    # reference different subsets of variables produce polytopes with
    # mismatched column counts, crashing minkowski_sum().
    from sympy import symbols
    canonical_vars = symbols(f'x0:{num_vars}')

    # Compute combined Newton polytope with shared variable list
    vertices = combined_newton_polytope(polynomials, vars_list=canonical_vars)

    pruner = NewtonPruner(num_vars, max_degree, vertices, required_support=required_support)
    pruner.compute_pruned_basis()
    return pruner


def prune_basis_from_problem(prog, max_degree: int,
                             required_support: Optional[Set[Tuple[int, ...]]] = None) -> NewtonPruner:
    """Convenience function: compute pruned basis from an OptimizationProblem.

    The objective and constraint polynomials may be SymPy expressions or
    SemigroupAlgebra elements; ring elements are converted to SymPy
    automatically (see :func:`newton_polytope`), so problems built via
    Irene's documented construction path work directly.

    Args:
        prog: An OptimizationProblem with set_objective() and add_constraint().
        max_degree: Maximum degree for moment matrix construction.
        required_support: Exponents that must be retained regardless of hull
            membership (see :class:`NewtonPruner`).  Default ``None`` keeps only
            hull-interior exponents -- callers building relaxation bases should
            pass the certified safe set.

    Returns:
        Configured NewtonPruner with pruned basis computed.

    Raises:
        ValueError: If any polynomial cannot be parsed (no silent fallback).
    """
    polys = []

    # Extract polynomial expression from objective (handles both SemigroupAlgebraElement and raw SymPy)
    if prog.objective is not None:
        obj = prog.objective
        if hasattr(obj, 'expr'):
            polys.append(obj.expr)
        elif hasattr(obj, 'to_sympy'):
            polys.append(obj.to_sympy())
        else:
            # Assume it's already a SymPy expression
            polys.append(obj)

    # Extract polynomial expressions from constraints
    for cnst in prog.constraints:
        if hasattr(cnst, 'expr'):
            polys.append(cnst.expr)
        elif hasattr(cnst, 'to_sympy'):
            polys.append(cnst.to_sympy())
        else:
            polys.append(cnst)

    nvars = prog.semigroup.numgens if hasattr(prog.semigroup, 'numgens') else len(prog.sga.gens)
    return prune_basis_from_polys(polys, nvars, max_degree, required_support=required_support)
