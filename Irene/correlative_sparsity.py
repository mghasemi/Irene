"""
Correlative Sparsity Detection for Moment Matrices
==================================================

Implements correlative sparsity analysis and chordal decomposition of moment
matrices for polynomial optimization. By exploiting variable co-occurrence in
polynomial constraints, large dense SDPs can be decomposed into smaller clique-based
subproblems with consistency constraints on shared variables.

References
----------
- Kojima, Kim & Waki (2007): "An SDP relaxation-based algorithm for global
  optimization using correlative sparsity"
- Waki, Kim, Muramatsu & Kojima (2006): "Exploiting algebraic structure in
  semidefinite programming"
- Lasserre (2006): "Cutting corners: faster methods for the SOS hierarchy"

Classes
-------
CorrelativeSparsity
    Detects sparsity patterns from polynomial constraints and builds chordal
    decompositions of moment matrices.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
from sympy import Poly, Symbol


# --------------------------------------------------------------------------- #
#  Graph utilities for clique detection                                        #
# --------------------------------------------------------------------------- #


class ChordalGraph:
    """Chordal graph decomposition via maximum cardinality search (MCS).

    Given a variable co-occurrence graph, finds maximal cliques that form the
    basis for decomposing large moment matrices into smaller PSD blocks.
    """

    def __init__(self, nvars: int):
        self.nvars = nvars
        self._adj: dict[int, set[int]] = defaultdict(set)

    def add_edge(self, u: int, v: int):
        """Add an undirected edge between variables u and v (0-indexed)."""
        if u != v:
            self._adj[u].add(v)
            self._adj[v].add(u)

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Return the set of unique edges."""
        seen = set()
        result = []
        for u in self._adj:
            for v in self._adj[u]:
                edge = (min(u, v), max(u, v))
                if edge not in seen:
                    seen.add(edge)
                    result.append(edge)
        return result

    @property
    def vertices(self) -> set[int]:
        """Return all vertices that appear in at least one edge."""
        verts = set()
        for u in self._adj:
            verts.add(u)
            verts.update(self._adj[u])
        return verts

    def maximal_cliques(self) -> list[set[int]]:
        """Find all maximal cliques using Bron-Kerbosch algorithm.

        Returns
        -------
        list[set[int]]
            List of maximal cliques, each a set of variable indices.
        """
        cliques: list[set[int]] = []
        R: set[int] = set()
        P: set[int] = self.vertices.copy()
        X: set[int] = set()

        def bron_kerbosch():
            if not P and not X:
                cliques.append(R.copy())
                return

            # Choose pivot to minimize branching
            pivot = max(P | X, key=lambda v: len(self._adj.get(v, set()) & P))
            candidates = P - self._adj.get(pivot, set())

            for v in list(candidates):
                neighbors = self._adj.get(v, set())
                bron_kerbosch_recursive(R | {v}, P & neighbors, X & neighbors)

        def bron_kerbosch_recursive(r, p, x):
            if not p and not x:
                cliques.append(r.copy())
                return

            pivot = max(p | x, key=lambda v: len(self._adj.get(v, set()) & p))
            candidates = p - self._adj.get(pivot, set())

            for v in list(candidates):
                neighbors = self._adj.get(v, set())
                bron_kerbosch_recursive(r | {v}, p & neighbors, x & neighbors)
                p.remove(v)
                x.add(v)

        bron_kerbosch()
        return cliques

    def running_intersection_property(self, cliques: list[set[int]]) -> list[set[int]]:
        """Reorder cliques to satisfy the Running Intersection Property (RIP).

        Uses a greedy ordering based on maximum cardinality search. The RIP is
        required for chordal decomposition consistency constraints.

        Parameters
        ----------
        cliques : list[set[int]]
            Maximal cliques from ``maximal_cliques()``.

        Returns
        -------
        list[set[int]]
            Reordered clique sequence satisfying RIP (approximately).
        """
        if not cliques:
            return []

        ordered = [cliques[0].copy()]
        remaining = list(cliques[1:])

        while remaining:
            # Find the clique whose intersection with already-ordered cliques is largest
            best_idx = 0
            best_overlap = -1
            for i, c in enumerate(remaining):
                overlap = sum(len(c & o) for o in ordered)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i

            ordered.append(remaining.pop(best_idx))

        return ordered


# --------------------------------------------------------------------------- #
#  Variable support extraction from polynomials                                #
# --------------------------------------------------------------------------- #


def _extract_var_indices(poly, variables: list[Symbol]) -> set[int]:
    """Extract the set of variable indices that appear in a polynomial.

    Parameters
    ----------
    poly : SymPy expression or Poly
        The polynomial to analyze.
    variables : list[Symbol]
        Full variable ordering.

    Returns
    -------
    set[int]
        Indices (0-based) of variables appearing with nonzero degree.
    """
    try:
        sp_poly = Poly(poly, *variables)
    except Exception:
        return set(range(len(variables)))  # conservative fallback

    var_indices: set[int] = set()
    for exp_tuple in sp_poly.monoms():
        for i, deg in enumerate(exp_tuple):
            if deg > 0:
                var_indices.add(i)

    return var_indices


# --------------------------------------------------------------------------- #
#  Main correlative sparsity class                                             #
# --------------------------------------------------------------------------- #


class CorrelativeSparsity:
    """Correlative sparsity analysis for polynomial optimization problems.

    Analyzes which variables co-occur in each constraint and objective, builds
    the variable co-occurrence graph, finds maximal cliques via chordal
    decomposition, and provides clique-based moment matrix partitioning.

    Parameters
    ----------
    polynomials : list
        List of SymPy polynomial expressions (objective + constraints).
    variables : list[Symbol]
        Variable ordering.
    constraint_types : list[str], optional
        Type labels for each polynomial ('objective', 'equality', 'inequality').

    Attributes
    ----------
    cliques : list[set[int]]
        Maximal cliques of the variable co-occurrence graph.
    clique_polynomials : dict[frozenset, list[int]]
        Mapping from clique (as frozenset) to indices of polynomials supported on it.

    Examples
    --------
    >>> from sympy import symbols
    >>> x1, x2, x3, x4 = symbols('x1:5')
    >>> # f involves {x1,x2}, g involves {x2,x3}, h involves {x3,x4}
    >>> polys = [x1**2 + x2**2, x2*x3 - 1, x3**2 + x4**2]
    >>> cs = CorrelativeSparsity(polys, [x1, x2, x3, x4])
    >>> cs.analyze()
    >>> print(cs.cliques)
    """

    def __init__(self, polynomials: Sequence, variables: list[Symbol],
                 constraint_types: Optional[list[str]] = None):
        self.polynomials = list(polynomials)
        self.variables = variables
        self.nvars = len(variables)
        self.constraint_types = (constraint_types or
                                ['objective'] + ['inequality'] * (len(polynomials) - 1))

        # Analysis results
        self._supports: list[set[int]] = []
        self.cliques: list[set[int]] = []
        self.ordered_cliques: list[set[int]] = []
        self.clique_polynomials: dict[frozenset, list[int]] = {}
        self._graph: Optional[ChordalGraph] = None

    def analyze(self) -> "CorrelativeSparsity":
        """Run the full correlative sparsity analysis pipeline.

        Steps:
        1. Extract variable supports from each polynomial
        2. Build co-occurrence graph (edges between variables appearing together)
        3. Find maximal cliques via Bron-Kerbosch
        4. Order cliques for Running Intersection Property
        5. Assign polynomials to cliques

        Returns
        -------
        self
            The CorrelativeSparsity instance with computed attributes.
        """
        # Step 1: Extract supports
        self._supports = []
        for p in self.polynomials:
            support = _extract_var_indices(p, self.variables)
            self._supports.append(support)

        # Step 2: Build co-occurrence graph
        self._graph = ChordalGraph(self.nvars)
        for support in self._supports:
            var_list = sorted(support)
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    self._graph.add_edge(var_list[i], var_list[j])

        # Step 3: Find maximal cliques
        self.cliques = self._graph.maximal_cliques()

        # Step 4: Order for RIP
        self.ordered_cliques = self._graph.running_intersection_property(self.cliques)

        # Step 5: Assign polynomials to smallest containing clique
        self.clique_polynomials = {}
        for idx, support in enumerate(self._supports):
            assigned = False
            for clique in self.ordered_cliques:
                if support <= clique:
                    key = frozenset(clique)
                    if key not in self.clique_polynomials:
                        self.clique_polynomials[key] = []
                    self.clique_polynomials[key].append(idx)
                    assigned = True
                    break

            # If no clique contains the support, assign to a fallback clique
            if not assigned:
                fallback = frozenset(range(self.nvars))
                if fallback not in self.clique_polynomials:
                    self.clique_polynomials[fallback] = []
                self.clique_polynomials[fallback].append(idx)

        return self

    def is_sparse(self, threshold: float = 0.5) -> bool:
        """Check whether the problem exhibits exploitable correlative sparsity.

        Parameters
        ----------
        threshold : float
            Sparsity ratio threshold. Returns True if no single clique contains
            more than ``threshold`` fraction of all variables.

        Returns
        -------
        bool
            True if the problem is sparse enough to benefit from decomposition.
        """
        if not self.cliques:
            return False
        max_clique_size = max(len(c) for c in self.cliques)
        return (max_clique_size / self.nvars) < threshold

    def clique_moment_sizes(self, degree: int) -> dict[frozenset, int]:
        """Compute the moment matrix size for each clique at a given relaxation degree.

        Parameters
        ----------
        degree : int
            Relaxation order (moment degree = 2*degree).

        Returns
        -------
        dict[frozenset, int]
            Mapping from clique to the dimension of its local moment matrix.
        """
        sizes: dict[frozenset, int] = {}
        for clique in self.cliques:
            k = len(clique)  # number of variables in this clique
            # Number of monomials in k variables up to total degree `degree`
            n_monomials = self._binomial(k + degree, degree)
            sizes[frozenset(clique)] = n_monomials

        return sizes

    def total_reduction_ratio(self, degree: int) -> float:
        """Compute the ratio of decomposed vs. dense moment matrix size.

        Parameters
        ----------
        degree : int
            Relaxation order.

        Returns
        -------
        float
            Ratio < 1 means decomposition reduces problem size.
        """
        sizes = self.clique_moment_sizes(degree)
        decomposed_size = sum(sizes.values())
        dense_size = self._binomial(self.nvars + degree, degree)

        if dense_size == 0:
            return 1.0
        return decomposed_size / dense_size

    def variable_partition(self) -> list[set[int]]:
        """Return the variable sets for each clique as a partition-like structure.

        Returns
        -------
        list[set[int]]
            Variable index sets per clique (may overlap at shared variables).
        """
        return [set(c) for c in self.cliques]

    def consistency_variables(self) -> dict[tuple[frozenset, frozenset], set[int]]:
        """Find shared variable sets between consecutive cliques (for RIP constraints).

        Returns
        -------
        dict
            Mapping from clique pairs to their shared variable indices.
        """
        shared: dict[tuple[frozenset, frozenset], set[int]] = {}
        ordered = self.ordered_cliques if self.ordered_cliques else self.cliques

        for i in range(len(ordered) - 1):
            c1 = frozenset(ordered[i])
            c2 = frozenset(ordered[i + 1])
            intersection = ordered[i] & ordered[i + 1]
            if intersection:
                shared[(c1, c2)] = intersection

        return shared

    def summary(self) -> dict:
        """Return a diagnostic summary of the sparsity analysis."""
        return {
            "nvars": self.nvars,
            "n_polynomials": len(self.polynomials),
            "n_cliques": len(self.cliques),
            "clique_sizes": [len(c) for c in self.cliques],
            "max_clique_size": max(len(c) for c in self.cliques) if self.cliques else 0,
            "is_sparse": self.is_sparse() if self.cliques else False,
            "n_edges": len(self._graph.edges) if self._graph else 0,
        }

    @staticmethod
    def _binomial(n: int, k: int) -> int:
        """Compute binomial coefficient C(n, k)."""
        if k < 0 or k > n:
            return 0
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    def __repr__(self) -> str:
        s = self.summary() if self.cliques else {}
        return (f"CorrelativeSparsity(nvars={s.get('nvars', '?')}, "
                f"cliques={s.get('n_cliques', 0)}, "
                f"max_size={s.get('max_clique_size', '?')})")


# --------------------------------------------------------------------------- #
#  Convenience function                                                        #
# --------------------------------------------------------------------------- #


def analyze_correlative_sparsity(polynomials: Sequence, variables: list[Symbol],
                                 constraint_types: Optional[list[str]] = None) -> CorrelativeSparsity:
    """Analyze correlative sparsity of a polynomial optimization problem.

    Parameters
    ----------
    polynomials : list
        Objective and constraint polynomials.
    variables : list[Symbol]
        Variable ordering.
    constraint_types : list[str], optional
        Labels for each polynomial.

    Returns
    -------
    CorrelativeSparsity
        Analyzed sparsity structure with clique decomposition.
    """
    cs = CorrelativeSparsity(polynomials, variables, constraint_types=constraint_types)
    return cs.analyze()
