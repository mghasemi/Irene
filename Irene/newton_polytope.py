"""Newton polytope monomial pruning for moment matrix dimension reduction.

For a polynomial optimization problem, the Newton polytope of the objective
and constraints defines which monomials can actually appear in the relaxation.
Monomials outside 2·Newt(f) are provably unnecessary, reducing the moment
matrix size — sometimes by orders of magnitude for sparse problems.

Key references:
    - Parrilo (2000), "Structured Semidefinite Programs and Semialgebraic Geometry"
    - Lasserre (2006), "Moments, Positive Polynomials and Their Applications"
    - Kim & Kojima (2014), "Sparse SOS decompositions via Newton polytopes"

The implementation computes the Minkowski sum of scaled Newton polytopes,
then filters the full monomial basis to only those inside the hull.
"""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from itertools import product as iter_product


def newton_polytope(expr, vars_list=None):
    """Compute the Newton polytope of a polynomial expression.

    The Newton polytope is the convex hull of exponent vectors of all terms
    with nonzero coefficients in the polynomial.

    Args:
        expr: A symbolic polynomial expression (SymPy or SymEngine).
        vars_list: List of variables to extract exponents for. If None, inferred.

    Returns:
        np.ndarray: Array of shape (num_terms, num_vars) of exponent vectors.
    """
    from .symbolic_engine import engine

    try:
        poly = engine.Poly(expr)
    except Exception:
        # Constant or unsupported expression — return zero vector
        if vars_list is not None:
            return np.zeros((1, len(vars_list)), dtype=int)
        return np.zeros((1, 0), dtype=int)

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

    A ⊕ B = {a + b | a ∈ A, b ∈ B}

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

    For moment matrix construction, we need 2·(Newt(f_0) ⊕ Newt(g_1) ⊕ ...),
    where f_0 is the objective and g_i are constraint polynomials.

    Args:
        polynomials: List of symbolic polynomial expressions.
        vars_list: Shared variable list for consistent exponent ordering.

    Returns:
        np.ndarray: Combined Newton polytope vertices (deduplicated).
    """
    if not polynomials:
        return None

    # Get all polytopes
    polytopes = []
    for expr in polynomials:
        try:
            pts = newton_polytope(expr, vars_list)
            # Skip degenerate polytopes (0 columns = no variables detected)
            if pts.shape[1] > 0:
                polytopes.append(pts)
        except Exception:
            continue

    if not polytopes:
        return None

    # Minkowski sum of all polytopes
    result = polytopes[0]
    for pts in polytopes[1:]:
        result = minkowski_sum(result, pts)

    # Scale by 2 (for degree-2d moment matrix)
    result = scale_polytope(result, 2)

    return result


class NewtonPruner:
    """Filter monomial basis using Newton polytope pruning.

    Given the combined Newton polytope of an optimization problem's
    polynomials, this class filters the full monomial basis to only those
    exponent vectors that lie within the convex hull of 2·Newt(f).

    Args:
        num_vars: Number of variables in the problem.
        max_degree: Maximum degree for moment matrix construction.
        polytope_vertices: Precomputed vertices of 2·combined Newton polytope.
            If None, will be computed from polynomials later.

    Attributes:
        full_basis_size: Size of unpruned monomial basis.
        pruned_basis_size: Size after Newton polytope filtering.
        reduction_ratio: Fraction of basis retained (lower = more pruning).
    """

    def __init__(self, num_vars: int, max_degree: int,
                 polytope_vertices: Optional[np.ndarray] = None):
        self.num_vars = num_vars
        self.max_degree = max_degree
        self.polytope_vertices = polytope_vertices
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
                    pass  # Degenerate hull (e.g., collinear points) → bbox fallback
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
            # hull.equations: each row is [normal..., offset], point p inside iff A·p <= b
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

        Iterates over all monomials up to max_degree and filters those
        outside the Newton polytope.

        Args:
            vars_list: Optional variable list (for compatibility).

        Returns:
            List of exponent tuples forming the pruned basis.
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

        self.pruned_basis_size = len(pruned)
        self.reduction_ratio = self.pruned_basis_size / max(self.full_basis_size, 1)

        return pruned

    def moment_matrix_dimension_reduction(self) -> Dict:
        """Estimate the moment matrix size reduction from Newton pruning.

        The moment matrix has dimension R×R where R is the basis size.
        Pruning reduces this to R'×R', so the reduction factor is (R'/R)².

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


def prune_basis_from_polys(polynomials, num_vars: int, max_degree: int) -> NewtonPruner:
    """Convenience function: compute pruned basis from a list of polynomials.

    Args:
        polynomials: List of symbolic polynomial expressions (objective + constraints).
        num_vars: Number of variables in the problem.
        max_degree: Maximum degree for moment matrix construction (usually 2*d).

    Returns:
        Configured NewtonPruner with pruned basis computed.
    """
    # Compute combined Newton polytope
    vertices = combined_newton_polytope(polynomials)

    pruner = NewtonPruner(num_vars, max_degree, vertices)
    pruner.compute_pruned_basis()
    return pruner


def prune_basis_from_problem(prog, max_degree: int) -> NewtonPruner:
    """Convenience function: compute pruned basis from an OptimizationProblem.

    Args:
        prog: An OptimizationProblem with set_objective() and add_constraint().
        max_degree: Maximum degree for moment matrix construction.

    Returns:
        Configured NewtonPruner with pruned basis computed.
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
    return prune_basis_from_polys(polys, nvars, max_degree)
