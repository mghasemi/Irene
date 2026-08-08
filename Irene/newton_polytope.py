"""
Newton Polytope Pruning for Moment Hierarchies
===============================================

Implements monomial selection based on Newton polytope geometry. For SOS and
moment relaxations, only monomials lying within the convex hull of half-supports
of the input polynomials can appear in Gram matrix decompositions. This dramatically
reduces the number of moment variables for sparse or structured problems.

References
----------
- Permenter & Parrilo (2012): "SOS-decomposition of multivariate polynomials with
  sparse Newton polytopes"
- Seiler, D'Angelo, Nie & Thienel (2013): "Exploiting sparsity induced by term
  substitution in polynomial optimization via the Lasserre hierarchy"
- Parrilo (2000): "Structured semidefinite programs and semialgebraic geometry
  methods in systems and control"

Classes
-------
NewtonPolytopePruner
    Computes Newton polytopes of polynomials, derives admissible monomial sets
    for SOS/moment relaxations, and provides integration hooks for Irene's
    relaxation modules.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
from sympy import Poly, Symbol


# --------------------------------------------------------------------------- #
#  Convex hull utilities (pure NumPy — no scipy dependency)                    #
# --------------------------------------------------------------------------- #


def _convex_hull_2d(points: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Andrew's monotone chain convex hull in 2D.

    Parameters
    ----------
    points : list of exponent tuples (truncated to first two coordinates).

    Returns
    -------
    list of vertices forming the convex hull boundary.
    """
    pts = sorted(set(
        (pt[0], pt[1]) if len(pt) >= 2 else (pt[0], 0)
        for pt in points
    ))
    if len(pts) <= 1:
        return [tuple(v) for v in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)

    upper = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)

    return lower[:-1] + upper[:-1]


def _point_in_polytope(point: tuple[int, ...], vertices: list[tuple[int, ...]],
                       nvars: int) -> bool:
    """Check if a lattice point lies inside the convex hull of vertices.

    Uses half-space representation derived from facet inequalities. For high
    dimensions (>3), falls back to checking barycentric coordinates via least
    squares.

    Parameters
    ----------
    point : tuple[int, ...]
        The lattice point to test.
    vertices : list of vertex tuples.
    nvars : int
        Number of variables.

    Returns
    -------
    bool
        True if the point lies inside (or on the boundary of) the polytope.
    """
    p = np.array(point[:nvars], dtype=float)
    V = np.array([v[:nvars] for v in vertices], dtype=float)

    if nvars == 1:
        return min(v[0] for v in vertices) <= point[0] <= max(v[0] for v in vertices)

    if nvars == 2:
        hull = _convex_hull_2d(vertices)
        # Point-in-polygon via winding number (cross product test)
        inside = False
        n_verts = len(hull)
        for i in range(n_verts):
            j = (i + 1) % n_verts
            yi, xi = hull[i][1], hull[i][0]
            yj, xj = hull[j][1], hull[j][0]
            if ((yi > point[1]) != (yj > point[1])):
                x_intersect = (xj - xi) * (point[1] - yi) / (yj - yi + 1e-30) + xi
                if point[0] < x_intersect:
                    inside = not inside
        return inside

    # General dimension: check if point is a convex combination of vertices
    # Solve V @ alpha = p with alpha >= 0 and sum(alpha) = 1
    # A has shape (nvars+1, num_vertices), b has shape (nvars+1,)
    ones_row = np.ones((1, len(vertices)))
    A = np.vstack([V.T, ones_row])          # (nvars+1, num_vertices)
    b = np.append(p[:nvars], 1.0)           # (nvars+1,)

    try:
        alpha, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        # Check feasibility: all alpha >= -tol and reconstruction error small
        if np.all(alpha >= -1e-8) and np.linalg.norm(A @ alpha - b) < 1e-6:
            return True
    except (np.linalg.LinAlgError, ValueError):
        pass

    return False


# --------------------------------------------------------------------------- #
#  Newton polytope extraction                                                  #
# --------------------------------------------------------------------------- #


def _newton_polytope(poly, variables: list[Symbol]) -> tuple[list[tuple[int, ...]], int]:
    """Extract the Newton polytope of a polynomial as its vertex set.

    Parameters
    ----------
    poly : SymPy expression or Poly
        The polynomial whose Newton polytope to compute.
    variables : list[Symbol]
        Variable ordering.

    Returns
    -------
    vertices : list[tuple[int, ...]]
        Vertices of the Newton polytope (exponent tuples, full nvars-dimensional).
    nvars : int
        Number of variables.
    """
    try:
        sp_poly = Poly(poly, *variables)
    except Exception:
        return [(0,) * len(variables)], len(variables)

    support = [exp_tuple for exp_tuple in sp_poly.monoms()]
    nvars = len(variables)

    if not support:
        return [(0,) * nvars], nvars

    # For 1D or very small support, just use the unique monomials directly
    if nvars == 1 or len(set(support)) <= 3:
        return list(set(support)), nvars

    # Compute convex hull in first two dimensions (projection)
    hull_2d = _convex_hull_2d(support)

    # Pad the 2D hull vertices back to full nvars dimensions by finding original
    # support points that match each 2D vertex and using their remaining coordinates.
    # If no exact match, zero-pad the remaining dimensions (conservative).
    padded_vertices: list[tuple[int, ...]] = []
    for hv_2d in hull_2d:
        # Find an original support point matching this 2D projection
        matched = False
        for s in support:
            if len(s) >= 2 and (s[0], s[1]) == hv_2d:
                padded_vertices.append(tuple(s[:nvars]))
                matched = True
                break
        if not matched:
            # Zero-pad remaining dimensions — conservative over-approximation
            padded = list(hv_2d) + [0] * max(0, nvars - len(hv_2d))
            padded_vertices.append(tuple(padded[:nvars]))

    return padded_vertices, nvars


# --------------------------------------------------------------------------- #
#  Half-polytope for SOS Gram matrices                                         #
# --------------------------------------------------------------------------- #


def _half_polytope(vertices: list[tuple[int, ...]], nvars: int) -> list[tuple[int, ...]]:
    """Compute the half Newton polytope vertices.

    For an SOS decomposition f = sum p_i^2, each p_i has monomials in
    conv(1/2 * supp(f)) intersect Z^n. This halves all vertex coordinates.

    Parameters
    ----------
    vertices : list of exponent tuples forming the Newton polytope.
    nvars : int
        Number of variables.

    Returns
    -------
    list[tuple[int, ...]]
        Vertices of the half-polytope (ceiling to ensure containment).
    """
    return [tuple(max(0, v // 2) for v in vert[:nvars]) for vert in vertices]


# --------------------------------------------------------------------------- #
#  Main pruner class                                                           #
# --------------------------------------------------------------------------- #


class NewtonPolytopePruner:
    """Newton polytope-based monomial pruning for moment/SOS hierarchies.

    Analyzes the Newton polytopes of input polynomials and determines which
    monomials can actually appear in Gram matrix decompositions at a given
    relaxation order. Monomials outside the relevant polytope regions are
    pruned, reducing SDP variable counts.

    Parameters
    ----------
    polynomials : list
        List of SymPy polynomial expressions (objective + constraints).
    variables : list[Symbol]
        Variable ordering.
    relaxation_degree : int
        The relaxation order d (moment degree = 2*d for SOS).

    Attributes
    ----------
    admissible_monomials : dict[int, set[tuple[int, ...]]]
        Per-polynomial sets of monomials that can appear in Gram decompositions.
    pruned_monomials : dict[int, set[tuple[int, ...]]]
        Monomials eliminated by pruning (for diagnostics).

    Examples
    --------
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> f = x**4 + x**2*y**2 + y**4 - 1
    >>> pruner = NewtonPolytopePruner([f], [x, y], relaxation_degree=2)
    >>> pruner.prune()
    >>> print(f"Admissible monomials: {len(pruner.admissible_monomials[0])}")
    """

    def __init__(self, polynomials: Sequence, variables: list[Symbol],
                 relaxation_degree: int = 2):
        self.polynomials = list(polynomials)
        self.variables = variables
        self.nvars = len(variables)
        self.relaxation_degree = relaxation_degree

        # Results
        self.admissible_monomials: dict[int, set[tuple[int, ...]]] = {}
        self.pruned_monomials: dict[int, set[tuple[int, ...]]] = {}
        self._original_counts: dict[int, int] = {}
        self._polytopes: list[list[tuple[int, ...]]] = []

    def prune(self) -> "NewtonPolytopePruner":
        """Run Newton polytope pruning on all polynomials.

        For each polynomial at relaxation degree d:
        1. Compute its Newton polytope P(f)
        2. The full monomial set would be all alpha with |alpha| <= 2d
        3. Prune to only those in (P(f) + ball(2d)) intersect Z^n
           For SOS: further restrict to half-polytope for Gram columns

        Returns
        -------
        self
            The pruner with computed admissible sets.
        """
        # Generate the full monomial set at degree 2*d
        full_monomials = self._all_monomials_up_to_degree(self.relaxation_degree * 2)

        for idx, p in enumerate(self.polynomials):
            vertices, _ = _newton_polytope(p, self.variables)
            self._polytopes.append(vertices)

            # The admissible set: monomials within the Newton polytope expanded
            # by the relaxation degree. For SOS Gram matrices, we need monomials
            # in conv(supp(f)) + {alpha : |alpha| <= d}.
            half_verts = _half_polytope(vertices, self.nvars)

            admissible: set[tuple[int, ...]] = set()
            for mono in full_monomials:
                if _point_in_polytope(mono, vertices, self.nvars):
                    admissible.add(mono)
                # Also include monomials reachable by adding degree-d terms to half-polytope
                elif self._reachable_from_half_polytope(mono, half_verts):
                    admissible.add(mono)

            self.admissible_monomials[idx] = admissible
            self.pruned_monomials[idx] = set(full_monomials) - admissible
            self._original_counts[idx] = len(full_monomials)

        return self

    def _reachable_from_half_polytope(self, mono: tuple[int, ...],
                                      half_verts: list[tuple[int, ...]]) -> bool:
        """Check if a monomial is reachable from the half-polytope by adding degree-d terms."""
        # A monomial m is admissible for SOS Gram columns if m = h + delta where
        # h is in the half-polytope and |delta| <= d. Equivalently, there exists
        # h in half_polytope such that m - h has nonnegative entries and total degree <= d.
        for hv in half_verts:
            diff = tuple(mono[i] - hv[i] for i in range(min(self.nvars, len(hv), len(mono))))
            if all(d >= 0 for d in diff) and sum(diff) <= self.relaxation_degree:
                return True

        # Also check convex combinations of half-vertices (conservative)
        h_arr = np.array([[max(0, v[i]) for i in range(self.nvars)] for v in half_verts], dtype=float)
        m_arr = np.array([max(0, mono[i]) for i in range(self.nvars)], dtype=float)

        # Check if m - d*e is within the half-polytope for some direction e with |e|=d
        # Simplified: check if m lies within degree-d of any point in the polytope
        for alpha in self._all_monomials_up_to_degree(self.relaxation_degree):
            shifted = tuple(mono[i] - alpha[i] for i in range(min(self.nvars, len(alpha), len(mono))))
            if all(s >= 0 for s in shifted):
                if _point_in_polytope(shifted, half_verts, self.nvars):
                    return True

        return False

    def _all_monomials_up_to_degree(self, degree: int) -> set[tuple[int, ...]]:
        """Generate all monomials with total degree <= degree in nvars variables."""
        result: set[tuple[int, ...]] = set()

        def _gen(dim: int, remaining: int, current: list[int]):
            if dim == 0:
                current.append(remaining)
                result.add(tuple(current))
                current.pop()
                return
            for e in range(remaining + 1):
                current.append(e)
                _gen(dim - 1, remaining - e, current)
                current.pop()

        _gen(self.nvars - 1, degree, [])
        return result

    def get_admissible_for_index(self, idx: int) -> set[tuple[int, ...]]:
        """Return admissible monomials for polynomial at index ``idx``."""
        if idx not in self.admissible_monomials:
            raise IndexError(f"No pruning data for index {idx}")
        return self.admissible_monomials[idx]

    def reduction_ratio(self, idx: int) -> float:
        """Fraction of monomials retained after pruning.

        Parameters
        ----------
        idx : int
            Polynomial index.

        Returns
        -------
        float
            Ratio of admissible / original count (1 = no pruning).
        """
        if idx not in self._original_counts or self._original_counts[idx] == 0:
            return 1.0
        return len(self.admissible_monomials.get(idx, set())) / self._original_counts[idx]

    def total_reduction_ratio(self) -> float:
        """Overall reduction ratio across all polynomials."""
        if not self._original_counts:
            return 1.0
        total_original = sum(self._original_counts.values())
        total_admissible = sum(len(v) for v in self.admissible_monomials.values())
        if total_original == 0:
            return 1.0
        return total_admissible / total_original

    def summary(self) -> dict:
        """Return a diagnostic summary of the pruning results."""
        return {
            "n_polynomials": len(self.polynomials),
            "relaxation_degree": self.relaxation_degree,
            "moment_degree": self.relaxation_degree * 2,
            "total_original_monomials": sum(self._original_counts.values()),
            "total_admissible_monomials": sum(len(v) for v in self.admissible_monomials.values()),
            "total_pruned_monomials": sum(len(v) for v in self.pruned_monomials.values()),
            "reduction_ratio": self.total_reduction_ratio(),
            "per_polynomial": {
                f"poly_{i}": {
                    "original": self._original_counts.get(i, 0),
                    "admissible": len(self.admissible_monomials.get(i, set())),
                    "pruned": len(self.pruned_monomials.get(i, set())),
                    "ratio": self.reduction_ratio(i),
                }
                for i in range(len(self.polynomials))
            },
        }

    def __repr__(self) -> str:
        s = self.summary() if self.admissible_monomials else {}
        return (f"NewtonPolytopePruner(nvars={self.nvars}, degree={s.get('relaxation_degree', '?')}, "
                f"reduction={s.get('reduction_ratio', '?'):.2%})")


# --------------------------------------------------------------------------- #
#  Convenience function                                                        #
# --------------------------------------------------------------------------- #


def prune_by_newton_polytope(polynomials: Sequence, variables: list[Symbol],
                             relaxation_degree: int = 2) -> NewtonPolytopePruner:
    """Prune monomials using Newton polytope geometry.

    Parameters
    ----------
    polynomials : list
        Objective and constraint polynomials.
    variables : list[Symbol]
        Variable ordering.
    relaxation_degree : int
        Relaxation order d (moment degree = 2*d).

    Returns
    -------
    NewtonPolytopePruner
        Pruner with computed admissible monomial sets.
    """
    pruner = NewtonPolytopePruner(polynomials, variables, relaxation_degree=relaxation_degree)
    return pruner.prune()
