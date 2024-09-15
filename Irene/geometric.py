"""
This module provides a framework for polynomial optimization using the techniques introduced by
Ghasemi, Lasserre, and Marshall, using Geometric Programming.
"""
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

    def __init__(self, prog: OptimizationProblem, **kwargs):
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

    def transform_program(self):
        """
        Transform the program using the transformation matrix H.
        """
        self.h = []
        for k in range(self.program_size):
            tmp_h = 0.
            for j in range(self.program_size):
                tmp_h = tmp_h + self.H[j, k] * self.g[j]
            self.h.append(tmp_h)

    def h_plus(self, xprsn, idn=None):
        """
        Compute the h_plus function for a given expression.

        Args:
            xprsn: The expression to evaluate.
            idn: The identity element of the semigroup.

        Returns:
            The value of h_plus(xprsn).
        """
        if idn is None:
            idn = self.prog.semigroup.G.identity
        return max(0., xprsn[idn])

    def form_gp(self):
        """
        Form the geometric program relaxation.

        Returns:
            The optimal value of the relaxation.
        """
        if self.auto_transform:
            self.auto_transform_matrix()
        self.transform_program()
        # initialize gp variables
        m = self.program_size
        delta = self.prog.delta(self.prog.objective, self.Ord)
        for xprsn in self.prog.constraints:
            xp_delta = self.prog.delta(-xprsn, self.Ord)
            delta = {'=d': delta['=d'].union(xp_delta['=d']), '<d': delta['<d'].union(xp_delta['<d'])}
        mu = VectorVariable(m, 'mu', '', "Lagrangian coefficients")
        all_delta = delta['=d'].union(delta['<d'])
        w = {alpha: Variable(f'w_{str(alpha)}') for alpha in all_delta}
        z = {alpha: VectorVariable(len(alpha.array_form), f'z_{alpha}', '', "Auxiliary variables") for alpha in
             all_delta}
        constraints = list()
        for _ in range(1, self.program_size):
            constraints.append(mu[_] <= 1e10)
        for alpha in all_delta:
            for _ in range(len(alpha.array_form)):
                constraints.append(z[alpha][_] <= 1e10)
                constraints.append(z[alpha][_] >= self.error_bound)
        # Define the objective function
        obj = 0
        for j in range(1, self.program_size):
            obj = obj + self.h_plus((self.h[j])) * mu[j]
        for alpha in delta['<d']:
            residual_pow = self.Ord - _degree(alpha)
            w_part0 = ((w[alpha] / self.Ord) ** (self.Ord / residual_pow))
            z_part0 = 1.
            temp_idx = 0
            for _ in alpha.array_form:
                z_part0 = z_part0 * (_[1] / z[alpha][temp_idx]) ** (_[1] / residual_pow)
                temp_idx += 1
            obj = obj + residual_pow * w_part0 * z_part0
        print('obj=', obj)
        constraints.append(mu[0] == 1.)
        print('-' * 30)
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
            print(lhs1, rhs1)
            if type(lhs1 <= rhs1) is not bool:
                constraints.append(lhs1 <= rhs1)
                print(lhs1 <= rhs1)
        print('-' * 30)
        # Second set on constraints in (3)
        for alpha in delta['=d']:
            lhs2 = 1.
            temp_idx = 0
            for _ in alpha.array_form:
                lhs2 = lhs2 * (z[alpha][temp_idx] / _[1]) ** _[1]
                temp_idx += 1
            rhs2 = (w[alpha] / self.Ord) ** self.Ord
            if type(lhs2 >= rhs2) is not bool:
                constraints.append(lhs2 >= rhs2)
                print(lhs2 >= rhs2)
        print('-' * 30)
        # Third set of constraints in (3)
        for alpha in all_delta:
            lhs3 = w[alpha]
            H_alpha_plus = 0.
            H_alpha_minus = 0.
            rhs31 = None
            rhs32 = None
            for j in range(self.program_size):
                h_j_alpha = self.h[j][alpha]
                if h_j_alpha < 0:
                    H_alpha_plus = H_alpha_plus + (-h_j_alpha) * mu[j]
                elif h_j_alpha > 0:
                    H_alpha_minus = H_alpha_minus + h_j_alpha * mu[j]
                rhs31 = H_alpha_plus
                rhs32 = H_alpha_minus
            if bool(rhs31):
                constraints.append(lhs3 >= rhs31)
                print(lhs3 >= rhs31)
            if bool(rhs32):
                constraints.append(lhs3 >= rhs32)
                print(lhs3 >= rhs32)
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
                print(lhs4 >= rhs4)
                constraints.append(lhs4 >= rhs4)
            else:
                print(lhs4)  # >= self.error_bound)
                constraints.append(lhs4 >= self.error_bound)
        mdl = Model(obj, Bounded(ConstraintSet(constraints), upper=1 / self.error_bound))
        # mdl = Model(obj, constraints)
        self.solution = mdl.solve()
        print(self.h[0])
        print(self.g[0])
        print(self.prog.objective)
        print(self.solution['cost'])
        self.f_gp_g = -self.h[0].constant() - self.solution['cost']
        return self.f_gp_g

    @staticmethod
    def compare_diags(vec):
        """
        Compare two vectors by the number of non-zero elements.

        Args:
            vec: The first vector.

        Returns:
            The number of non-zero elements in the vector.
        """
        nz = sum(1 for _ in vec if _ != 0)
        return nz, vec

    def auto_transform_matrix(self):
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
