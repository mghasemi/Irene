"""
This module provides a framework for polynomial optimization using the techniques introduced by
Ghasemi, Lasserre, and Marshall, using Geometric Programming.
"""
from typing import Any

import numpy as np
from gpkit import VectorVariable, Variable, Model
from gpkit.constraints.bounded import Bounded, ConstraintSet

from .grouprings import _degree
from .program import OptimizationProblem


class GPRelaxations(object):
    r"""
    This class aims to provide a framework for polynomial optimization using the techniques
    introduced by Ghasemi, Lasserre, and Marshall, using Geometric Programming.
    """

    def __init__(self, prog: OptimizationProblem, **kwargs) -> None:
        """
        Initializes the GPRelaxations class.

        Args:
            prog: The OptimizationProblem object to be relaxed.
            **kwargs: Keyword arguments.
                - H: The transformation matrix to be used.
                - auto_transform: Boolean indicating whether to automatically transform the program.
        """
        self.prog = prog
        self.program_size = len(prog.constraints) + 1
        self.g = [-prog.objective] + prog.constraints
        self.h = list()
        self.Ord = self.prog.program_degree()
        self.error_bound = 1e-10
        self.solution = None
        self.f_gp_g = None
        self.H = kwargs.get('H', np.identity(self.program_size))
        self.auto_transform = kwargs.get('auto_transform', True)
        self.verbosity = kwargs.get('verbosity', 1)

    def transform_program(self) -> None:
        """
        Transform the program using the transformation matrix H.
        """
        self.h = []
        for k in range(self.program_size):
            tmp_h = 0.
            for j in range(self.program_size):
                tmp_h = tmp_h + self.H[j, k] * self.g[j]
            self.h.append(tmp_h)

    def h_plus(self, xprsn: Any, idn: Any = None) -> float:
        """
        Compute the h_plus function for a given expression.

        Args:
            xprsn: The expression to evaluate.
            idn: The identity element of the semigroup.

        Returns:
            The value of h_plus(xprsn).
        """
        if idn is None:
            if not hasattr(self.prog, 'semigroup'):
                raise ValueError("OptimizationProblem must define 'semigroup'")
            idn = self.prog.semigroup.G.identity
        return max(0., xprsn[idn])

    @staticmethod
    def compare_diags(vec: list[float]) -> tuple[int, list[float]]:
        """
        Compare two vectors by the number of non-zero elements.

        Args:
            vec: The first vector.

        Returns:
            The number of non-zero elements in the vector.
        """
        nz = sum(1 for _ in vec if _ != 0)
        return nz, vec

    @staticmethod
    def _append_symbolic_constraint(constraints: list, constraint: Any) -> None:
        """Append only symbolic constraints and ignore plain boolean results."""
        if isinstance(constraint, (bool, np.bool_)):
            return
        constraints.append(constraint)

    def _build_delta_sets(self) -> dict[str, set]:
        delta = self.prog.delta(self.prog.objective, self.Ord)
        for xprsn in self.prog.constraints:
            xp_delta = self.prog.delta(-xprsn, self.Ord)
            delta = {
                '=d': delta['=d'].union(xp_delta['=d']),
                '<d': delta['<d'].union(xp_delta['<d'])
            }
        return delta

    def _initialize_variables(self, delta: dict[str, set]) -> tuple[Any, dict, dict, set]:
        m = self.program_size
        mu = VectorVariable(m, 'mu', '', "Lagrangian coefficients")
        all_delta = delta['=d'].union(delta['<d'])
        w = {alpha: Variable(f'w_{str(alpha)}') for alpha in all_delta}
        z = {
            alpha: VectorVariable(len(alpha.array_form), f'z_{alpha}', '', "Auxiliary variables")
            for alpha in all_delta
        }
        return mu, w, z, all_delta

    def _add_variable_bounds(self, constraints: list, mu: Any, z: dict, all_delta: set) -> None:
        for j in range(1, self.program_size):
            constraints.append(mu[j] <= 1e10)
        for alpha in all_delta:
            for idx in range(len(alpha.array_form)):
                constraints.append(z[alpha][idx] <= 1e10)
                constraints.append(z[alpha][idx] >= self.error_bound)

    def _build_objective(self, mu: Any, w: dict, z: dict, delta: dict[str, set]) -> Any:
        obj = 0
        for j in range(1, self.program_size):
            obj = obj + self.h_plus(self.h[j]) * mu[j]
        for alpha in delta['<d']:
            residual_pow = self.Ord - _degree(alpha)
            w_part = (w[alpha] / self.Ord) ** (self.Ord / residual_pow)
            z_part = 1.
            for idx, mono in enumerate(alpha.array_form):
                z_part = z_part * (mono[1] / z[alpha][idx]) ** (mono[1] / residual_pow)
            obj = obj + residual_pow * w_part * z_part
        return obj

    def _solve_model(self, obj: Any, constraints: list) -> None:
        mdl = Model(obj, Bounded(ConstraintSet(constraints), upper=1 / self.error_bound))
        try:
            self.solution = mdl.solve()
        except Exception as exc:
            raise RuntimeError(f"GP solve failed: {exc}") from exc
        if self.solution is None:
            raise RuntimeError("GP solver returned no solution")

    def auto_transform_matrix(self) -> np.ndarray:
        """
        Compute the automatic transformation matrix H.

        Returns:
            The transformation matrix H.
        """
        # Form the sorted diagonal part of the program
        diag = [[] for _ in range(self.program_size)]
        n = len(self.prog.sga.gens)
        for symb in self.prog.sga.gens:
            mono = self.prog.sga[symb] ** self.Ord
            for j in range(self.program_size):
                diag[j].append(self.g[j][mono])
        cnst_diag = diag[1:]
        cnst_diag.sort(key=self.compare_diags, reverse=True)
        sorted_diag = [diag[0]] + cnst_diag
        a = np.identity(self.program_size)
        # Check the (*) condition in 4.2
        for k in range(1, self.program_size):
            for j in range(k + 1, self.program_size):
                a[j, k] = min(
                    [-(sorted_diag[k][i] + sum(a[jp, k] * sorted_diag[jp][i] for jp in range(k, j))) /
                     sorted_diag[j][i] for i in range(n)] + [0.])
        return a

    def solve(self) -> float:
        """
        Form the geometric program relaxation.

        Returns:
            The optimal value of the relaxation.
        """
        if self.auto_transform:
            self.H = self.auto_transform_matrix()
        self.transform_program()

        delta = self._build_delta_sets()
        mu, w, z, all_delta = self._initialize_variables(delta)
        constraints = list()
        self._add_variable_bounds(constraints, mu, z, all_delta)
        obj = self._build_objective(mu, w, z, delta)

        if self.verbosity > 0:
            print('obj=', obj)
            print('-' * 30)

        constraints.append(mu[0] == 1.)
        # First set of constraints in (3) -- double check
        for symb in self.prog.sga.gens:
            rhs1 = 0.
            lhs1 = 0.  # self.error_bound
            sg_symb = self.prog.semigroup.__getattribute__(symb) ** self.Ord
            for alpha in all_delta:
                ind, idx = self.prog.has_symbol(symb, alpha)
                if ind:
                    lhs1 = lhs1 + z[alpha][idx]
            for j in range(self.program_size):
                mu_j_cf = - self.h[j][sg_symb]
                if mu_j_cf < 0:
                    lhs1 = lhs1 + (-mu_j_cf) * mu[j]
                elif mu_j_cf > 0:
                    rhs1 = rhs1 + mu_j_cf * mu[j]
            if not bool(lhs1):
                lhs1 = lhs1 + self.error_bound
            if self.verbosity > 0:
                print(lhs1, rhs1)
            cns = lhs1 <= rhs1
            self._append_symbolic_constraint(constraints, cns)
            if self.verbosity > 0 and not isinstance(cns, bool):
                print(cns)
        if self.verbosity > 0:
            print('-' * 30)

        # Second set on constraints in (3)
        for alpha in delta['=d']:
            lhs2 = 1.
            temp_idx = 0
            for _ in alpha.array_form:
                lhs2 = lhs2 * (z[alpha][temp_idx] / _[1]) ** _[1]
                temp_idx += 1
            rhs2 = (w[alpha] / self.Ord) ** self.Ord
            cns = lhs2 >= rhs2
            self._append_symbolic_constraint(constraints, cns)
            if self.verbosity > 0 and not isinstance(cns, bool):
                print(cns)
        if self.verbosity > 0:
            print('-' * 30)

        # Third set of constraints in (3)
        for alpha in all_delta:
            lhs3 = w[alpha]
            H_alpha_plus = 0.
            H_alpha_minus = 0.
            for j in range(self.program_size):
                h_j_alpha = self.h[j][alpha]
                if h_j_alpha < 0:
                    H_alpha_plus = H_alpha_plus + (-h_j_alpha) * mu[j]
                elif h_j_alpha > 0:
                    H_alpha_minus = H_alpha_minus + h_j_alpha * mu[j]
            if bool(H_alpha_plus):
                constraints.append(lhs3 >= H_alpha_plus)
                if self.verbosity > 0:
                    print(lhs3 >= H_alpha_plus)
            if bool(H_alpha_minus):
                constraints.append(lhs3 >= H_alpha_minus)
                if self.verbosity > 0:
                    print(lhs3 >= H_alpha_minus)
        if self.verbosity > 0:
            print('-' * 30)

        # Fourth set of constraints in (3)
        for j in range(self.program_size):
            lhs4 = 0.
            rhs4 = 0.
            # lhs4 = sum(self.H[j][k] * mu[k] for k in range(self.program_size))
            for k in range(self.program_size):
                if self.H[j][k] >= 0:
                    lhs4 = lhs4 + self.H[j][k] * mu[k]
                else:
                    rhs4 = rhs4 + (-self.H[j][k]) * mu[k]
            if bool(rhs4):
                if self.verbosity > 0:
                    print(lhs4 >= rhs4)
                constraints.append(lhs4 >= rhs4)
            else:
                if self.verbosity > 0:
                    print(lhs4)  # >= self.error_bound)
                constraints.append(lhs4 >= self.error_bound)

        if self.verbosity > 0:
            print('-' * 30)

        self._solve_model(obj, constraints)
        self.f_gp_g = -self.h[0].constant() - self.solution['cost']
        return self.f_gp_g
