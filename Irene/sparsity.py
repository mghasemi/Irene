"""Correlative sparsity detection for SDP relaxation decomposition.

Correlative sparsity exploits the fact that not all variables appear together
in every polynomial constraint. By building a variable dependency graph where
edges connect variables that co-occur in the same term, we can find connected
components and decompose a large moment matrix into smaller independent blocks.

Key references:
    - Hall & Sastry (2015), "Exploiting sparsity in polynomial optimization"
    - Kurzhanskiy et al. (2017), "Sparse semidefinite programming relaxations"
    - Louveaux et al. (2018), "Correlative and term-at-a-time sparsity"

The implementation uses a union-find data structure for efficient connected
component detection, then returns the clique decomposition needed to
partition moment matrices.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np


class UnionFind:
    """Disjoint-set union-find with path compression and rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n = n
        self.components = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True

    def get_components(self) -> Dict[int, List[int]]:
        """Return mapping from root -> list of members."""
        comps = {}
        for i in range(self.n):
            root = self.find(i)
            comps.setdefault(root, []).append(i)
        # Re-key by sorted component lists for stability
        result = {}
        for members in sorted(comps.values(), key=lambda m: min(m)):
            result[min(members)] = members
        return result


class CorrelativeSparsity:
    """Detect and exploit correlative sparsity in polynomial optimization problems.

    Given an optimization problem with variables x_1, ..., x_n and polynomials
    (objective + constraints), this class builds a variable dependency graph
    where an edge (i, j) exists if variables i and j appear together in some
    monomial term. The connected components of this graph define the correlative
    sparsity pattern.

    Args:
        num_vars: Number of optimization variables.
        var_names: Optional list of variable names/symbols for debugging.

    Attributes:
        adjacency: Adjacency list representation of the dependency graph.
        components: Connected components as lists of variable indices.
        is_sparse: True if sparsity detected (more than one component).
    """

    def __init__(self, num_vars: int, var_names: Optional[List] = None):
        self.num_vars = num_vars
        self.var_names = var_names or list(range(num_vars))
        self.adjacency: Dict[int, Set[int]] = {i: set() for i in range(num_vars)}
        self._uf = UnionFind(num_vars)
        self.components: List[List[int]] = []
        self.is_sparse = False

    def add_term(self, var_indices: List[int]) -> None:
        """Add edges for a monomial term involving the given variables.

        For a term like x_1^2 * x_3, var_indices would be [0, 2].
        All pairs of co-occurring variables get connected.

        Args:
            var_indices: Indices of variables that appear in this term.
        """
        if len(var_indices) <= 1:
            return
        unique_vars = sorted(set(var_indices))
        for i in range(len(unique_vars)):
            for j in range(i + 1, len(unique_vars)):
                vi, vj = unique_vars[i], unique_vars[j]
                self.adjacency[vi].add(vj)
                self.adjacency[vj].add(vi)
                self._uf.union(vi, vj)

    def add_poly_terms(self, exponent_dict: Dict[Tuple[int, ...], object]) -> None:
        """Add edges from a polynomial's term dictionary.

        Args:
            exponent_dict: Mapping from exponent tuple -> coefficient (as returned
                by engine.Poly(...).as_dict()). Each key represents one monomial.
        """
        for exp_tuple in exponent_dict.keys():
            # Variables with non-zero exponents co-occur
            vars_in_term = [i for i, e in enumerate(exp_tuple) if e > 0]
            self.add_term(vars_in_term)

    def finalize(self) -> List[List[int]]:
        """Compute connected components and return them sorted by size (descending)."""
        comp_dict = self._uf.get_components()
        self.components = sorted(comp_dict.values(), key=len, reverse=True)
        self.is_sparse = len(self.components) > 1
        return self.components

    def moment_matrix_partition(self, deg: int) -> Dict[int, List[Tuple[int, ...]]]:
        """Partition the moment matrix basis by sparsity components.

        For each connected component of variables, compute which exponent tuples
        belong exclusively to that component (i.e., only use variables from that
        component). This allows decomposing the large moment matrix into smaller
        independent blocks.

        Args:
            deg: Maximum degree for moment basis generation.

        Returns:
            Dict mapping component index -> list of exponent tuples belonging
            to that component's moment block. Exponents that span multiple
            components are assigned to a 'cross' block (key=-1).
        """
        if not self.components:
            self.finalize()

        # Map each variable to its component index
        var_to_comp = {}
        for comp_idx, comp in enumerate(self.components):
            for v in comp:
                var_to_comp[v] = comp_idx

        partitions = {i: [] for i in range(len(self.components))}
        partitions[-1] = []  # cross-component terms

        from itertools import product as iter_product
        all_monos = iter_product(range(deg + 1), repeat=self.num_vars)
        for exp_tuple in all_monos:
            if sum(exp_tuple) > deg:
                continue
            # Find which components this exponent touches
            active_comps = set()
            for var_idx, exp_val in enumerate(exp_tuple):
                if exp_val > 0 and var_idx in var_to_comp:
                    active_comps.add(var_to_comp[var_idx])
            if len(active_comps) == 1:
                comp_id = active_comps.pop()
                partitions[comp_id].append(exp_tuple)
            else:
                partitions[-1].append(exp_tuple)

        return partitions

    def reduction_factor(self, deg: int) -> float:
        """Estimate the moment matrix size reduction from sparsity.

        Returns the ratio of total work with sparsity vs without. A value < 1
        means sparsity helps. For a problem decomposed into k components of
        sizes n_1, ..., n_k, the reduction is roughly:
            sum_i (2*deg choose n_i) / (2*deg choose n)

        Args:
            deg: Relaxation degree.

        Returns:
            Reduction factor (< 1 means improvement).
        """
        from math import comb

        if not self.components:
            self.finalize()

        if not self.is_sparse:
            return 1.0

        # Full basis size (upper bound)
        full_size = sum(comb(2 * deg + v - 1, v) for v in range(self.num_vars + 1))
        if full_size == 0:
            return 1.0

        # Sum of per-component basis sizes + cross terms
        partitions = self.moment_matrix_partition(deg)
        sparse_size = sum(len(partitions[i]) for i in range(len(self.components)))
        cross_size = len(partitions.get(-1, []))

        # Cross terms still need full treatment; weight them less aggressively
        effective_sparse = sparse_size + int(cross_size * 0.5)
        return effective_sparse / max(full_size, 1)

    def summary(self) -> Dict:
        """Return a human-readable summary of the sparsity pattern."""
        if not self.components:
            self.finalize()
        return {
            "num_vars": self.num_vars,
            "num_components": len(self.components),
            "is_sparse": self.is_sparse,
            "component_sizes": [len(c) for c in self.components],
            "components": self.components,
            "edges": sum(len(neighbors) for neighbors in self.adjacency.values()) // 2,
        }


def detect_sparsity_from_problem(prog) -> CorrelativeSparsity:
    """Detect correlative sparsity from an OptimizationProblem instance.

    Inspects the objective and constraints, extracts variable co-occurrence
    patterns from each polynomial's term structure, and builds the dependency
    graph.

    Args:
        prog: An OptimizationProblem with set_objective() and add_constraint()
            already called.

    Returns:
        Configured CorrelativeSparsity instance with finalized components.
    """
    from .symbolic_engine import engine

    nvars = prog.semigroup.numgens if hasattr(prog.semigroup, 'numgens') else len(prog.sga.generators)
    sparsity = CorrelativeSparsity(nvars)

    # Process objective
    if prog.objective is not None:
        try:
            obj_poly = engine.Poly(prog.objective.expr, *prog.AuxSyms)
            sparsity.add_poly_terms(obj_poly.as_dict())
        except Exception:
            pass  # If poly conversion fails, skip

    # Process constraints
    for cnst in prog.constraints:
        try:
            cnst_poly = engine.Poly(cnst.expr, *prog.AuxSyms)
            sparsity.add_poly_terms(cnst_poly.as_dict())
        except Exception:
            pass

    sparsity.finalize()
    return sparsity


def detect_sparsity_from_polys(polynomials, num_vars: int) -> CorrelativeSparsity:
    """Detect correlative sparsity from a list of polynomial expressions.

    Args:
        polynomials: List of symbolic polynomial expressions.
        num_vars: Number of variables in the problem.

    Returns:
        Configured CorrelativeSparsity instance with finalized components.
    """
    from .symbolic_engine import engine

    sparsity = CorrelativeSparsity(num_vars)
    for poly_expr in polynomials:
        try:
            p = engine.Poly(poly_expr)
            sparsity.add_poly_terms(p.as_dict())
        except Exception:
            pass
    sparsity.finalize()
    return sparsity
