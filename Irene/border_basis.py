"""
Border Basis Algorithm for Zero-Dimensional Ideals
===================================================

Implements the Greuel-Plesser border basis algorithm as a numerically stable
alternative to Gröbner bases for zero-dimensional polynomial ideals. Border
bases work with order ideals (finite sets of monomials closed under division)
rather than term orderings, making them suitable for local/degree-compatible
orderings and numerical computation.

References
----------
- Möller & Trager (1987): "The element structure of finite dimensional
  associative algebras"
- Greuel & Plesser (1992): "Standard bases for zero-dimensional ideals"
- Trinks (2003): "Border Bases — An Introduction"

Classes
-------
BorderBasis
    Main class implementing the border basis algorithm with support for
    multiplication matrices, normal forms, and root extraction.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sympy import Poly, S, Symbol, symbols


# --------------------------------------------------------------------------- #
#  Monomial utilities                                                          #
# --------------------------------------------------------------------------- #


class Monomial:
    """Immutable multivariate monomial represented by its exponent tuple.

    Parameters
    ----------
    exp : tuple[int, ...]
        Exponent vector $(\\alpha_1, \\dots, \\alpha_n)$.
    nvars : int
        Number of variables (pads/expands ``exp`` as needed).
    """

    __slots__ = ("exp", "nvars")

    def __init__(self, exp: tuple[int, ...], nvars: int):
        if len(exp) < nvars:
            exp = exp + (0,) * (nvars - len(exp))
        elif len(exp) > nvars:
            exp = exp[:nvars]
        self.exp: tuple[int, ...] = exp
        self.nvars: int = nvars

    @property
    def total_degree(self) -> int:
        return sum(self.exp)

    def divides(self, other: "Monomial") -> bool:
        """True if ``self`` divides ``other``."""
        return all(s <= o for s, o in zip(self.exp, other.exp))

    def __mul__(self, other: "Monomial") -> "Monomial":
        new_exp = tuple(a + b for a, b in zip(self.exp, other.exp))
        return Monomial(new_exp, self.nvars)

    def __div__(self, other: "Monomial") -> Optional["Monomial"]:
        """Exact division; returns None if not divisible."""
        if not other.divides(self):
            return None
        new_exp = tuple(a - b for a, b in zip(self.exp, other.exp))
        return Monomial(new_exp, self.nvars)

    def __truediv__(self, other):
        return self.__div__(other)

    def __hash__(self) -> int:
        return hash(self.exp)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Monomial):
            return NotImplemented
        return self.exp == other.exp

    def __lt__(self, other: "Monomial") -> bool:
        """Degree-lexicographic comparison."""
        if self.total_degree != other.total_degree:
            return self.total_degree < other.total_degree
        return self.exp < other.exp

    def __repr__(self) -> str:
        parts = []
        for i, e in enumerate(self.exp):
            if e > 0:
                var_name = f"x{i + 1}"
                parts.append(f"{var_name}^{e}" if e > 1 else var_name)
        return " * ".join(parts) if parts else "1"


# --------------------------------------------------------------------------- #
#  Order ideal and border                                                      #
# --------------------------------------------------------------------------- #


class OrderIdeal:
    """A finite set of monomials closed under division (an order ideal).

    Parameters
    ----------
    nvars : int
        Number of variables.
    max_degree : Optional[int]
        Maximum total degree to consider when computing borders.
    """

    def __init__(self, nvars: int, max_degree: Optional[int] = None):
        self.nvars = nvars
        self.max_degree = max_degree
        self._monomials: list[Monomial] = [Monomial((0,) * nvars, nvars)]  # always contains 1
        self._mono_set: set[Monomial] = {self._monomials[0]}

    @property
    def monomials(self) -> list[Monomial]:
        return list(self._monomials)

    @property
    def size(self) -> int:
        return len(self._monomials)

    def add(self, mono: Monomial) -> bool:
        """Add a monomial and all its divisors. Returns True if changed."""
        if mono in self._mono_set:
            return False
        # Add the monomial itself
        self._monomials.append(mono)
        self._mono_set.add(mono)
        # Ensure closure under division
        changed = True
        while changed:
            changed = False
            for m in list(self._monomials):
                for i in range(self.nvars):
                    if m.exp[i] > 0:
                        divisor_exp = list(m.exp)
                        divisor_exp[i] -= 1
                        divisor = Monomial(tuple(divisor_exp), self.nvars)
                        if divisor not in self._mono_set:
                            self._monomials.append(divisor)
                            self._mono_set.add(divisor)
                            changed = True
        return True

    def border(self) -> list[Monomial]:
        """Compute the border :math:`\\partial \\mathcal{O}`.

        The border is the set of monomials not in ``self`` but divisible by
        at least one element of ``self``, with total degree equal to
        ``max_degree(self) + 1``.
        """
        target_deg = self._max_total_degree() + 1
        if self.max_degree is not None:
            target_deg = min(target_deg, self.max_degree)

        border_set: set[Monomial] = set()
        for m in self._monomials:
            if m.total_degree == target_deg - 1:
                for i in range(self.nvars):
                    new_exp = list(m.exp)
                    new_exp[i] += 1
                    b = Monomial(tuple(new_exp), self.nvars)
                    if b not in self._mono_set:
                        border_set.add(b)

        return sorted(border_set)

    def _max_total_degree(self) -> int:
        return max(m.total_degree for m in self._monomials) if self._monomials else 0

    def __contains__(self, mono: Monomial) -> bool:
        return mono in self._mono_set

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"OrderIdeal(nvars={self.nvars}, size={self.size})"


# --------------------------------------------------------------------------- #
#  Border basis polynomial representation                                     #
# --------------------------------------------------------------------------- #


class BorderBasisPoly:
    """Polynomial expressed as a linear combination of monomials.

    Used internally for border basis computation. Coefficients are stored
    in a dictionary mapping Monomial -> float.
    """

    __slots__ = ("coeffs", "nvars")

    def __init__(self, nvars: int):
        self.coeffs: dict[Monomial, float] = {}
        self.nvars = nvars

    @classmethod
    def from_sympy(cls, poly, nvars: int) -> "BorderBasisPoly":
        """Create from a SymPy polynomial."""
        bb_poly = cls(nvars)
        if hasattr(poly, 'as_dict'):
            for exp_tuple, coeff in poly.as_dict().items():
                mono = Monomial(exp_tuple, nvars)
                bb_poly.coeffs[mono] = float(coeff)
        else:
            from sympy import Poly as SPoly
            try:
                all_syms = symbols(f'x1:{nvars + 1}')
                sp_poly = SPoly(poly, *all_syms)
                return cls.from_sympy(sp_poly, nvars)
            except Exception:
                raise ValueError(f"Cannot convert {type(poly)} to BorderBasisPoly")
        return bb_poly

    def add_term(self, mono: Monomial, coeff: float):
        if abs(coeff) < 1e-14:
            self.coeffs.pop(mono, None)
            return
        old = self.coeffs.get(mono, 0.0)
        new_val = old + coeff
        if abs(new_val) < 1e-14:
            self.coeffs.pop(mono, None)
        else:
            self.coeffs[mono] = new_val

    def scale(self, factor: float):
        for mono in list(self.coeffs.keys()):
            self.coeffs[mono] *= factor

    def leading_monomial(self) -> Optional[Monomial]:
        """Return the largest monomial (by degree-lex) with nonzero coefficient."""
        if not self.coeffs:
            return None
        return max(self.coeffs, key=lambda m: (m.total_degree, m.exp))

    def leading_coefficient(self) -> float:
        lm = self.leading_monomial()
        return self.coeffs[lm] if lm else 0.0

    def copy(self) -> "BorderBasisPoly":
        c = BorderBasisPoly(self.nvars)
        c.coeffs = dict(self.coeffs)
        return c

    def __repr__(self) -> str:
        terms = []
        for mono in sorted(self.coeffs, key=lambda m: (m.total_degree, m.exp), reverse=True):
            cv = self.coeffs[mono]
            if abs(cv - round(cv)) < 1e-10:
                c_str = f"{int(round(cv))}"
            else:
                c_str = f"{cv:.4g}"
            terms.append(f"{c_str}*{mono}")
        return " + ".join(terms) if terms else "0"


# --------------------------------------------------------------------------- #
#  Main Border Basis class                                                     #
# --------------------------------------------------------------------------- #


class BorderBasis:
    """Border basis of a zero-dimensional polynomial ideal.

    Computes the border basis with respect to an order ideal :math:`\\mathcal{O}`,
    providing multiplication matrices, normal forms, and root extraction.

    Parameters
    ----------
    polynomials : list
        List of SymPy polynomials generating the ideal.
    variables : list[Symbol], optional
        Variable ordering. If None, inferred from polynomials.
    max_degree : Optional[int]
        Maximum degree bound for the order ideal extension.

    Attributes
    ----------
    order_ideal : OrderIdeal
        The computed order ideal :math:`\\mathcal{O}`.
    border_basis : list[BorderBasisPoly]
        The border basis polynomials :math:`g_j = b_j - \\sum c_{jt} t`.
    multiplication_matrices : dict[str, np.ndarray]
        Multiplication matrices for each variable.

    Examples
    --------
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> F = [x**2 - 2, y**2 - x]
    >>> bb = BorderBasis(F)
    >>> bb.compute()
    >>> roots = bb.roots()
    """

    def __init__(self, polynomials: Sequence, variables=None, max_degree: Optional[int] = None):
        self.polynomials = list(polynomials)
        self.variables = variables
        self.max_degree = max_degree
        self.nvars = 0
        self._sympy_polys: list[Poly] = []
        self.order_ideal: Optional[OrderIdeal] = None
        self.border_basis: list[BorderBasisPoly] = []
        self.multiplication_matrices: dict[str, np.ndarray] = {}
        self._mono_to_idx: dict[Monomial, int] = {}

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def _setup(self):
        """Initialize variables and convert polynomials."""
        if self.variables:
            self.nvars = len(self.variables)
        else:
            all_syms: set[Symbol] = set()
            for p in self.polynomials:
                try:
                    all_syms.update(p.free_symbols)
                except AttributeError:
                    pass
            self.variables = sorted(all_syms, key=lambda s: str(s))
            self.nvars = len(self.variables)

        if not self.nvars:
            raise ValueError("No variables found in polynomials")

        for p in self.polynomials:
            try:
                sp_poly = Poly(p, *self.variables)
            except Exception:
                sp_poly = Poly(p, *self.variables)
            self._sympy_polys.append(sp_poly)

        if self.max_degree is None:
            max_deg = max(p.total_degree() for p in self._sympy_polys)
            self.max_degree = max_deg + 2

    def _init_order_ideal(self) -> OrderIdeal:
        """Initialize order ideal from polynomial supports."""
        oi = OrderIdeal(self.nvars, max_degree=self.max_degree)
        for sp_poly in self._sympy_polys:
            for exp_tuple in sp_poly.monoms():
                mono = Monomial(exp_tuple, self.nvars)
                oi.add(mono)
        return oi

    # ------------------------------------------------------------------ #
    #  Reduction — substitute border elements with order-ideal combos      #
    # ------------------------------------------------------------------ #

    def _reduce(self, poly: BorderBasisPoly, basis: list[BorderBasisPoly]) -> BorderBasisPoly:
        """Reduce a polynomial modulo the current border basis.

        Each basis element g_j has leading monomial b_j (a border element) and
        remaining terms in the order ideal. Reduction repeatedly finds any term
        in ``poly`` whose monomial matches some b_j and substitutes it with the
        order-ideal combination from g_j.
        """
        result = poly.copy()

        # Build lookup: leading_monomial -> (basis_element, leading_coeff)
        lm_lookup: dict[Monomial, tuple[BorderBasisPoly, float]] = {}
        for g in basis:
            gl = g.leading_monomial()
            if gl is not None:
                lc = g.leading_coefficient()
                if abs(lc) > 1e-14:
                    lm_lookup[gl] = (g, lc)

        max_steps = 5000
        for _ in range(max_steps):
            # Find any monomial in result that matches a basis leading term
            matched_mono = None
            for mono in list(result.coeffs.keys()):
                if mono in lm_lookup:
                    matched_mono = mono
                    break

            if matched_mono is None:
                break  # nothing to reduce

            g, glc = lm_lookup[matched_mono]
            factor = result.coeffs[matched_mono] / glc

            # Subtract factor * g from result
            for g_mono, g_coeff in g.coeffs.items():
                result.add_term(g_mono, -factor * g_coeff)

        return result

    # ------------------------------------------------------------------ #
    #  Main computation — Greuel-Plesser algorithm                        #
    # ------------------------------------------------------------------ #

    def compute(self, verbose: bool = False) -> "BorderBasis":
        """Compute the border basis using the Greuel-Plesser algorithm.

        The algorithm iteratively extends the order ideal and reduces border
        elements until a stable basis is found.

        Parameters
        ----------
        verbose : bool
            Print progress information during computation.

        Returns
        -------
        self
            The BorderBasis instance with computed attributes.
        """
        self._setup()

        if verbose:
            print(f"[BorderBasis] Computing for {len(self.polynomials)} polynomials "
                  f"in {self.nvars} variables (max_degree={self.max_degree})")

        # Step 1: Initialize order ideal from supports
        self.order_ideal = self._init_order_ideal()

        if verbose:
            print(f"[BorderBasis] Initial order ideal size: {self.order_ideal.size}")

        # Step 2: Convert input polynomials to BorderBasisPoly format
        bb_polys: list[BorderBasisPoly] = []
        for sp_poly in self._sympy_polys:
            bb_p = BorderBasisPoly.from_sympy(sp_poly, self.nvars)
            bb_polys.append(bb_p)

        # Step 3: Iterative border basis computation (Greuel-Plesser)
        basis: list[BorderBasisPoly] = []
        max_iterations = (self.max_degree or 10) * 20
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            border = self.order_ideal.border()

            if not border:
                break

            # For each border element b, find an input polynomial f_i and a monomial
            # t in supp(f_i) such that b/t is in the order ideal. Then compute
            # (b/t) * f_i and reduce modulo current basis. If nonzero, add to basis.
            new_basis_elements: list[BorderBasisPoly] = []

            for b in border:
                found = False
                for p in bb_polys:
                    if found:
                        break
                    # Check each monomial in supp(p) to see if it divides b
                    for t in list(p.coeffs.keys()):
                        if not t.divides(b):
                            continue
                        quotient = b / t
                        if quotient is None or (self.order_ideal is not None and quotient not in self.order_ideal):
                            continue

                        # Multiply p by quotient and reduce
                        scaled = BorderBasisPoly(self.nvars)
                        for pm, pc in p.coeffs.items():
                            new_m = pm * quotient
                            if self.max_degree is None or new_m.total_degree <= self.max_degree:
                                scaled.add_term(new_m, pc)

                        r = self._reduce(scaled, basis + new_basis_elements)
                        if r.coeffs:
                            # Normalize so leading coefficient = 1
                            lc = r.leading_coefficient()
                            if abs(lc) > 1e-14:
                                r.scale(1.0 / lc)
                                new_basis_elements.append(r)
                                found = True
                        break

                if not found and verbose:
                    print(f"  [BorderBasis] Border element {b} could not be reduced")

            # Convergence check
            if not new_basis_elements:
                break

            basis.extend(new_basis_elements)

            if verbose and iteration % 3 == 0:
                print(f"[BorderBasis] Iteration {iteration}: "
                      f"basis size = {len(basis)}, order ideal = {self.order_ideal.size}")

        self.border_basis = basis

        if verbose:
            print(f"[BorderBasis] Computed in {iteration} iterations. "
                  f"Basis has {len(basis)} elements, order ideal size = {self.order_ideal.size}")

        # Step 4: Build monomial index mapping
        self._mono_to_idx = {m: i for i, m in enumerate(self.order_ideal.monomials)}

        # Step 5: Compute multiplication matrices
        self._compute_multiplication_matrices()

        return self

    # ------------------------------------------------------------------ #
    #  Multiplication matrices                                             #
    # ------------------------------------------------------------------ #

    def _compute_multiplication_matrices(self):
        """Compute multiplication matrices for each variable.

        For each variable :math:`x_i`, the matrix :math:`M_{x_i}` represents
        multiplication by :math:`x_i` in the quotient algebra with respect to
        the order ideal basis.
        """
        if self.order_ideal is None:
            raise RuntimeError("Order ideal not computed. Call compute() first.")

        n = self.order_ideal.size
        monos = self.order_ideal.monomials

        for i, var in enumerate(self.variables or []):
            M = np.zeros((n, n), dtype=float)
            var_mono = Monomial(tuple([0] * i + [1] + [0] * (self.nvars - i - 1)), self.nvars)

            for j, m in enumerate(monos):
                product = m * var_mono
                if product in self.order_ideal:
                    k = self._mono_to_idx[product]
                    M[k][j] = 1.0
                else:
                    # Reduce using border basis
                    reduced_poly = BorderBasisPoly(self.nvars)
                    reduced_poly.add_term(product, 1.0)
                    r = self._reduce(reduced_poly, self.border_basis)
                    for r_mono, r_coeff in r.coeffs.items():
                        if r_mono in self._mono_to_idx:
                            k = self._mono_to_idx[r_mono]
                            M[k][j] += r_coeff

            self.multiplication_matrices[str(var)] = M

    # ------------------------------------------------------------------ #
    #  Normal form                                                         #
    # ------------------------------------------------------------------ #

    def normal_form(self, poly) -> dict[Monomial, float]:
        """Compute the normal form of a polynomial modulo the ideal.

        Parameters
        ----------
        poly : SymPy expression or BorderBasisPoly
            The polynomial to reduce.

        Returns
        -------
        dict[Monomial, float]
            Coefficient dictionary expressing the normal form as a linear
            combination of order ideal monomials.
        """
        if isinstance(poly, BorderBasisPoly):
            bb_poly = poly
        else:
            sp_poly = Poly(poly, *self.variables)
            bb_poly = BorderBasisPoly.from_sympy(sp_poly, self.nvars)

        r = self._reduce(bb_poly, self.border_basis)
        return dict(r.coeffs)

    # ------------------------------------------------------------------ #
    #  Root extraction                                                     #
    # ------------------------------------------------------------------ #

    def roots(self, tolerance: float = 1e-8) -> list[dict[str, float]]:
        """Extract approximate roots from the multiplication matrices.

        Uses the common eigenvector approach: for a zero-dimensional ideal,
        the roots correspond to simultaneous eigenvalues of the multiplication
        matrices.

        Parameters
        ----------
        tolerance : float
            Numerical tolerance for eigenvalue clustering.

        Returns
        -------
        list[dict[str, float]]
            List of root dictionaries mapping variable names to approximate values.
        """
        if not self.multiplication_matrices:
            raise RuntimeError("Multiplication matrices not computed. Call compute() first.")

        var_names = list(self.multiplication_matrices.keys())
        M_first = self.multiplication_matrices[var_names[0]]

        eigenvalues, eigenvectors = np.linalg.eig(M_first)

        roots: list[dict[str, float]] = []
        for idx in range(len(eigenvalues)):
            root: dict[str, float] = {}
            ev_real = float(np.real(eigenvalues[idx]))
            root[var_names[0]] = ev_real

            # Verify consistency with other multiplication matrices via eigenvectors
            ev_vec = eigenvectors[:, idx]

            for vn in var_names[1:]:
                M_vn = self.multiplication_matrices[vn]
                Mv_ev = M_vn @ ev_vec
                val = None
                for k in range(len(ev_vec)):
                    if abs(ev_vec[k]) > tolerance:
                        val = float(np.real(Mv_ev[k] / ev_vec[k]))
                        break
                if val is not None:
                    root[vn] = val

            roots.append(root)

        return roots

    # ------------------------------------------------------------------ #
    #  Dimension                                                           #
    # ------------------------------------------------------------------ #

    def dimension(self) -> int:
        """Return the vector space dimension of the quotient algebra.

        This equals the number of points in the variety (counting multiplicity).
        """
        if self.order_ideal is None:
            return 0
        return self.order_ideal.size

    def __repr__(self) -> str:
        status = "computed" if self.border_basis else "not computed"
        return (f"BorderBasis(nvars={self.nvars}, polys={len(self.polynomials)}, "
                f"dims={self.dimension()}, {status})")


# --------------------------------------------------------------------------- #
#  Convenience function                                                        #
# --------------------------------------------------------------------------- #


def border_basis(polynomials: Sequence, variables=None, max_degree: Optional[int] = None,
                 verbose: bool = False) -> BorderBasis:
    """Compute the border basis of a zero-dimensional ideal.

    Parameters
    ----------
    polynomials : list
        Polynomials generating the ideal.
    variables : list[Symbol], optional
        Variable ordering.
    max_degree : Optional[int]
        Degree bound for order ideal extension.
    verbose : bool
        Print progress information.

    Returns
    -------
    BorderBasis
        Computed border basis with multiplication matrices and root data.

    Examples
    --------
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> bb = border_basis([x**2 - 2, y**2 - x], variables=[x, y])
    >>> roots = bb.roots()
    """
    bb = BorderBasis(polynomials, variables=variables, max_degree=max_degree)
    return bb.compute(verbose=verbose)
