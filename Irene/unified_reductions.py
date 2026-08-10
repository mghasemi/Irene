"""
Unified Reductions API for Polynomial Optimization
===================================================

Provides a single entry point that chains correlative sparsity detection, Newton
polytope pruning, and border basis reduction into Irene's relaxation pipeline.
The unified API automatically selects which reductions are applicable based on
problem structure and degree bounds.

Architecture
------------
1. Sparsity analysis → chordal decomposition of moment matrix
2. Newton polytope pruning → monomial set reduction per clique
3. Border basis reduction → algebraic simplification for equality constraints
4. Integrated SDP construction with all active reductions

Classes
-------
UnifiedReductions
    Orchestrates the full reduction pipeline and produces a reduced relaxation
    problem that can be fed to Irene's existing SDP solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Sequence

import numpy as np
from sympy import Poly, Symbol


# --------------------------------------------------------------------------- #
#  Reduction flags and configuration                                           #
# --------------------------------------------------------------------------- #


class ReductionType(Enum):
    """Available reduction strategies."""
    CORRELATIVE_SPARSITY = auto()
    NEWTON_POLYTOPE = auto()
    BORDER_BASIS = auto()


@dataclass
class ReductionConfig:
    """Configuration for the unified reductions pipeline.

    Parameters
    ----------
    enable_sparsity : bool
        Enable correlative sparsity detection and chordal decomposition.
    enable_newton_polytope : bool
        Enable Newton polytope monomial pruning.
    enable_border_basis : bool
        Enable border basis reduction for equality constraints.
    relaxation_degree : int
        Relaxation order d (moment degree = 2*d).
    sparsity_threshold : float
        Minimum sparsity ratio to activate chordal decomposition (< 1.0 means
        at least one clique is smaller than the full variable set).
    border_basis_max_degree : Optional[int]
        Degree bound for border basis computation (None = auto-detect).

    Examples
    --------
    >>> config = ReductionConfig(relaxation_degree=3, enable_border_basis=True)
    """
    enable_sparsity: bool = True
    enable_newton_polytope: bool = True
    enable_border_basis: bool = False
    relaxation_degree: int = 2
    sparsity_threshold: float = 0.8
    border_basis_max_degree: Optional[int] = None

    def active_reductions(self) -> list[ReductionType]:
        """Return the list of enabled reduction types."""
        result = []
        if self.enable_sparsity:
            result.append(ReductionType.CORRELATIVE_SPARSITY)
        if self.enable_newton_polytope:
            result.append(ReductionType.NEWTON_POLYTOPE)
        if self.enable_border_basis:
            result.append(ReductionType.BORDER_BASIS)
        return result


# --------------------------------------------------------------------------- #
#  Reduction state container                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class ReductionState:
    """Container for reduction results and diagnostics.

    Attributes
    ----------
    cliques : list[set[int]]
        Maximal cliques from correlative sparsity (empty if not applied).
    admissible_monomials : dict[int, set[tuple[int, ...]]]
        Per-polynomial admissible monomial sets after Newton pruning.
    border_basis_result : Optional[object]
        Border basis computation result for equality constraints.
    original_moment_size : int
        Dimension of the full (unreduced) moment matrix.
    reduced_moment_size : int
        Total dimension after all reductions applied.
    reduction_ratio : float
        Fraction of variables retained (< 1 means reduction occurred).
    active_reductions : list[ReductionType]
        Which reductions were actually applied (may differ from config if
        auto-detection disabled some).
    """
    cliques: list = field(default_factory=list)
    admissible_monomials: dict = field(default_factory=dict)
    border_basis_result: Optional[object] = None
    original_moment_size: int = 0
    reduced_moment_size: int = 0
    reduction_ratio: float = 1.0
    active_reductions: list = field(default_factory=list)

    def summary(self) -> dict:
        """Return a diagnostic summary."""
        return {
            "original_moment_size": self.original_moment_size,
            "reduced_moment_size": self.reduced_moment_size,
            "reduction_ratio": f"{self.reduction_ratio:.2%}",
            "n_cliques": len(self.cliques),
            "active_reductions": [r.name for r in self.active_reductions],
        }


# --------------------------------------------------------------------------- #
#  Unified reductions orchestrator                                             #
# --------------------------------------------------------------------------- #


class UnifiedReductions:
    """Unified reduction pipeline for polynomial optimization relaxations.

    Chains correlative sparsity, Newton polytope pruning, and border basis
    reduction into a single coherent workflow. Automatically detects which
    reductions are beneficial based on problem structure.

    Parameters
    ----------
    polynomials : list
        List of SymPy polynomials (objective + constraints).
    variables : list[Symbol]
        Variable ordering.
    constraint_types : list[str], optional
        Labels: 'objective', 'equality', 'inequality'.
    config : ReductionConfig, optional
        Pipeline configuration.

    Attributes
    ----------
    state : ReductionState
        Computed reduction results and diagnostics.

    Examples
    --------
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1:4')
    >>> polys = [x1**2 + x2**2, x2*x3 - 1, x3**2]
    >>> ureductions = UnifiedReductions(polys, [x1, x2, x3])
    >>> ureductions.run()
    >>> print(ureductions.state.summary())
    """

    def __init__(self, polynomials: Sequence, variables: list[Symbol],
                 constraint_types: Optional[list[str]] = None,
                 config: Optional[ReductionConfig] = None):
        self.polynomials = list(polynomials)
        self.variables = variables
        self.nvars = len(variables)
        self.constraint_types = (constraint_types or
                                ['objective'] + ['inequality'] * (len(polynomials) - 1))
        self.config = config or ReductionConfig()

        # Lazy imports — only load heavy modules when needed
        self._sparsity_module = None
        self._newton_module = None
        self._border_module = None

        # Computed state
        self.state = ReductionState()

    def _load_sparsity(self):
        """Lazy-load correlative sparsity module."""
        if self._sparsity_module is None:
            from Irene.correlative_sparsity import CorrelativeSparsity as CS
            self._sparsity_module = CS

    def _load_newton(self):
        """Lazy-load Newton polytope pruner."""
        if self._newton_module is None:
            from Irene.newton_polytope import NewtonPolytopePruner as NPP
            self._newton_module = NPP

    def _load_border(self):
        """Lazy-load border basis module."""
        if self._border_module is None:
            from Irene.border_basis import BorderBasis as BB
            self._border_module = BB

    # ------------------------------------------------------------------ #
    #  Pipeline execution                                                  #
    # ------------------------------------------------------------------ #

    def run(self, verbose: bool = False) -> ReductionState:
        """Execute the full reduction pipeline.

        Steps (in order):
        1. Correlative sparsity analysis → chordal decomposition
        2. Newton polytope pruning → monomial set reduction per clique
        3. Border basis reduction → algebraic simplification for equalities
        4. Compute final reduced moment matrix dimensions

        Parameters
        ----------
        verbose : bool
            Print progress information during pipeline execution.

        Returns
        -------
        ReductionState
            Computed reduction results and diagnostics.
        """
        if verbose:
            print("[UnifiedReductions] Starting reduction pipeline")
            print(f"  nvars={self.nvars}, n_polys={len(self.polynomials)}, "
                  f"degree={self.config.relaxation_degree}")

        # Compute baseline moment matrix size
        self.state.original_moment_size = self._moment_matrix_dim(
            self.nvars, self.config.relaxation_degree)

        if verbose:
            print(f"  Original moment matrix dimension: {self.state.original_moment_size}")

        # Step 1: Correlative sparsity
        if self.config.enable_sparsity:
            self._apply_correlative_sparsity(verbose=verbose)

        # Step 2: Newton polytope pruning
        if self.config.enable_newton_polytope:
            self._apply_newton_polytope_pruning(verbose=verbose)

        # Step 3: Border basis for equality constraints
        if self.config.enable_border_basis:
            self._apply_border_basis_reduction(verbose=verbose)

        # Compute final reduced size
        self._compute_reduced_size()

        if verbose:
            print(f"[UnifiedReductions] Pipeline complete. "
                  f"Reduction ratio: {self.state.reduction_ratio:.2%}")

        return self.state

    # ------------------------------------------------------------------ #
    #  Individual reduction steps                                          #
    # ------------------------------------------------------------------ #

    def _apply_correlative_sparsity(self, verbose: bool = False):
        """Step 1: Correlative sparsity analysis and chordal decomposition."""
        if verbose:
            print("  [Step 1] Correlative sparsity analysis")

        self._load_sparsity()
        assert self._sparsity_module is not None

        cs = self._sparsity_module(
            self.polynomials, self.variables,
            constraint_types=self.constraint_types)
        cs.analyze()

        # Only activate if the problem is actually sparse enough
        if cs.is_sparse(threshold=self.config.sparsity_threshold):
            self.state.cliques = cs.cliques
            self.state.active_reductions.append(ReductionType.CORRELATIVE_SPARSITY)
            if verbose:
                print(f"    Found {len(cs.cliques)} cliques, "
                      f"max size = {max(len(c) for c in cs.cliques)}")
        else:
            # Even if not sparse enough for full decomposition, store the
            # co-occurrence structure for diagnostics
            self.state.cliques = [set(range(self.nvars))]  # single full clique

    def _apply_newton_polytope_pruning(self, verbose: bool = False):
        """Step 2: Newton polytope monomial pruning."""
        if verbose:
            print("  [Step 2] Newton polytope pruning")

        self._load_newton()
        assert self._newton_module is not None

        pruner = self._newton_module(
            self.polynomials, self.variables,
            relaxation_degree=self.config.relaxation_degree)
        pruner.prune()

        self.state.admissible_monomials = pruner.admissible_monomials
        ratio = pruner.total_reduction_ratio()

        if ratio < 1.0:
            self.state.active_reductions.append(ReductionType.NEWTON_POLYTOPE)
            if verbose:
                print(f"    Pruning retained {ratio:.2%} of monomials")
        else:
            if verbose:
                print("    No pruning benefit (full support)")

    def _apply_border_basis_reduction(self, verbose: bool = False):
        """Step 3: Border basis reduction for equality constraints."""
        # Extract equality constraint polynomials
        equality_polys = []
        for idx, ctype in enumerate(self.constraint_types):
            if ctype == 'equality':
                equality_polys.append(self.polynomials[idx])

        if not equality_polys:
            if verbose:
                print("  [Step 3] No equality constraints — border basis skipped")
            return

        if verbose:
            print(f"  [Step 3] Border basis for {len(equality_polys)} equalities")

        self._load_border()
        assert self._border_module is not None

        max_deg = (self.config.border_basis_max_degree or
                   self.config.relaxation_degree * 2)

        try:
            bb = self._border_module(
                equality_polys, variables=self.variables,
                max_degree=max_deg)
            bb.compute(verbose=False)
            self.state.border_basis_result = bb
            self.state.active_reductions.append(ReductionType.BORDER_BASIS)

            if verbose:
                print(f"    Border basis computed. Dimension = {bb.dimension()}")
        except Exception as e:
            if verbose:
                print(f"    Border basis failed: {e}")

    # ------------------------------------------------------------------ #
    #  Size computation                                                    #
    # ------------------------------------------------------------------ #

    def _compute_reduced_size(self):
        """Compute the total reduced moment matrix dimension after all reductions."""
        d = self.config.relaxation_degree

        if self.state.cliques and len(self.state.cliques) > 1:
            # Chordal decomposition: sum of per-clique moment sizes
            total = 0
            for clique in self.state.cliques:
                k = len(clique)
                clique_dim = self._moment_matrix_dim(k, d)

                # Apply Newton pruning within this clique if available
                if self.state.admissible_monomials:
                    # Conservative estimate: use average reduction ratio
                    ratios = [len(v) for v in self.state.admissible_monomials.values()]
                    avg_ratio = (sum(ratios) / len(ratios)) if ratios else 1.0
                    full_clique_dim = self._moment_matrix_dim(k, d)
                    clique_dim = max(1, int(full_clique_dim * min(avg_ratio, 1.0)))

                total += clique_dim

            self.state.reduced_moment_size = total
        elif self.state.admissible_monomials:
            # Newton pruning only (no sparsity decomposition)
            ratios = [len(v) for v in self.state.admissible_monomials.values()]
            avg_ratio = (sum(ratios) / len(ratios)) if ratios else 1.0
            full_dim = self._moment_matrix_dim(self.nvars, d)
            self.state.reduced_moment_size = max(1, int(full_dim * min(avg_ratio, 1.0)))
        else:
            # No effective reduction
            self.state.reduced_moment_size = self.state.original_moment_size

        if self.state.original_moment_size > 0:
            self.state.reduction_ratio = (self.state.reduced_moment_size /
                                         self.state.original_moment_size)
        else:
            self.state.reduction_ratio = 1.0

    @staticmethod
    def _moment_matrix_dim(nvars: int, degree: int) -> int:
        """Compute the dimension of a moment matrix for nvars variables at given degree.

        This equals the number of monomials in ``nvars`` variables with total
        degree up to ``degree``, i.e., C(nvars + degree, degree).
        """
        from Irene.correlative_sparsity import CorrelativeSparsity
        return CorrelativeSparsity._binomial(nvars + degree, degree)

    # ------------------------------------------------------------------ #
    #  Integration helpers                                                 #
    # ------------------------------------------------------------------ #

    def get_clique_polynomials(self) -> dict[frozenset, list[int]]:
        """Return polynomial indices assigned to each clique.

        Returns
        -------
        dict
            Mapping from clique (frozenset of var indices) to polynomial indices.
        Empty if correlative sparsity was not applied.
        """
        if not self.state.cliques or len(self.state.cliques) <= 1:
            return {}

        # Re-run the assignment logic
        from Irene.correlative_sparsity import _extract_var_indices
        result: dict[frozenset, list[int]] = {}

        for idx, p in enumerate(self.polynomials):
            support = _extract_var_indices(p, self.variables)
            assigned = False
            for clique in self.state.cliques:
                if support <= clique:
                    key = frozenset(clique)
                    if key not in result:
                        result[key] = []
                    result[key].append(idx)
                    assigned = True
                    break

            if not assigned:
                fallback = frozenset(range(self.nvars))
                if fallback not in result:
                    result[fallback] = []
                result[fallback].append(idx)

        return result

    def get_normal_form(self, poly) -> Optional[dict]:
        """Compute the normal form of a polynomial using border basis.

        Parameters
        ----------
        poly : SymPy expression
            The polynomial to reduce.

        Returns
        -------
        dict or None
            Coefficient dictionary if border basis was computed, else None.
        """
        bb = self.state.border_basis_result
        if bb is not None:
            nf_method = getattr(bb, 'normal_form', None)
            if nf_method is not None:
                return nf_method(poly)
        return None

    def __repr__(self) -> str:
        s = self.state.summary()
        return (f"UnifiedReductions(nvars={self.nvars}, "
                f"reductions={[r.name for r in self.state.active_reductions]}, "
                f"ratio={s.get('reduction_ratio', 'N/A')})")


# --------------------------------------------------------------------------- #
#  Convenience function                                                        #
# --------------------------------------------------------------------------- #


def apply_unified_reductions(polynomials: Sequence, variables: list[Symbol],
                             constraint_types: Optional[list[str]] = None,
                             config: Optional[ReductionConfig] = None,
                             verbose: bool = False) -> UnifiedReductions:
    """Apply the full unified reductions pipeline.

    Parameters
    ----------
    polynomials : list
        Objective and constraint polynomials.
    variables : list[Symbol]
        Variable ordering.
    constraint_types : list[str], optional
        Labels for each polynomial.
    config : ReductionConfig, optional
        Pipeline configuration.
    verbose : bool
        Print progress information.

    Returns
    -------
    UnifiedReductions
        Instance with computed reduction state and diagnostics.

    Examples
    --------
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1:4')
    >>> polys = [x1**2 + x2**2 - 1, x2*x3 - 1, x3**2]
    >>> ureductions = apply_unified_reductions(polys, [x1, x2, x3], verbose=True)
    """
    ureductions = UnifiedReductions(
        polynomials, variables,
        constraint_types=constraint_types, config=config)
    ureductions.run(verbose=verbose)
    return ureductions
