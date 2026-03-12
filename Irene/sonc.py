from typing import Any

import numpy as np
from scipy import optimize
from gpkit import VectorVariable, Variable, Model, SignomialsEnabled
from gpkit.constraints.bounded import Bounded, ConstraintSet

from .program import OptimizationProblem


class SONCRelaxations(object):
    r"""
    Framework for constrained SONC relaxations formulated as geometric programs.
    """

    def __init__(self, prog: OptimizationProblem, **kwargs) -> None:
        self.prog = prog
        self.program_size = len(prog.constraints) + 1
        # Section 3 uses G(mu) = f - sum_i mu_i * g_i.
        self.g = [prog.objective] + [-c for c in prog.constraints]
        self.Ord = self.prog.program_degree()
        self.error_bound = kwargs.get('error_bound', 1e-10)
        self.max_bound = kwargs.get('max_bound', 1e10)
        self.verbosity = kwargs.get('verbosity', 1)
        self.use_local_solve = kwargs.get('use_local_solve', True)
        self.solution = None
        self.f_sonc_g = None

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

    def _build_support_points(self) -> tuple[list[tuple[int, ...]], list, int, bool]:
        supports = []
        hull_computed = False
        try:
            self.prog.newton()
            if self.prog.vertices:
                supports = [tuple(int(v) for v in point) for point in self.prog.vertices]
                hull_computed = True
        except Exception:
            # Fallback for lower-dimensional supports where ConvexHull may fail.
            supports = []

        if not supports:
            exponents = set()
            for coeff, mono in self.prog.objective.content:
                _ = coeff
                exponents.add(tuple(self.prog.mono2ord_tuple(mono)))
            for xprsn in self.prog.constraints:
                for coeff, mono in xprsn.content:
                    _ = coeff
                    exponents.add(tuple(self.prog.mono2ord_tuple(mono)))
            supports = sorted(exponents)

        n = len(self.prog.semigroup.generators)
        origin = tuple([0] * n)
        if origin not in supports:
            supports = [origin] + supports

        alpha = [self.prog.tuple2mono(pt) for pt in supports]
        origin_idx = supports.index(origin)
        return supports, alpha, origin_idx, hull_computed

    def _convex_combination(self, point: tuple[int, ...], supports: list[tuple[int, ...]]) -> np.ndarray | None:
        np_vertices = np.array(supports, dtype=float)
        A_eq = np_vertices.T
        b_eq = np.array(point, dtype=float)

        A_ub = -np.identity(np_vertices.shape[0])
        b_ub = np.zeros(np_vertices.shape[0])

        additional_eq_constraint = np.ones((1, np_vertices.shape[0]))
        A_eq = np.vstack([A_eq, additional_eq_constraint])
        b_eq = np.append(b_eq, 1.0)

        result = optimize.linprog(
            c=np.zeros(np_vertices.shape[0]),
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=(0, None),
            method='highs'
        )

        if result.success:
            return result.x
        return None

    def _build_beta_info(
        self,
        beta_terms: list,
        supports: list[tuple[int, ...]],
        origin_idx: int
    ) -> dict:
        beta_info = {}
        for beta in beta_terms:
            beta_tuple = self.prog.mono2ord_tuple(beta)
            lambdas = self._convex_combination(beta_tuple, supports)
            if lambdas is None:
                raise RuntimeError(f"Unable to express beta={beta} as convex combination of support points")

            lambdas = np.where(lambdas < self.error_bound, 0.0, lambdas)
            s = float(np.sum(lambdas))
            if s <= 0:
                raise RuntimeError(f"Degenerate convex-combination weights for beta={beta}")
            lambdas = lambdas / s

            nz = [idx for idx, val in enumerate(lambdas) if val > self.error_bound]
            if not nz:
                raise RuntimeError(f"No positive convex-combination weights for beta={beta}")

            beta_info[beta] = {
                'tuple': beta_tuple,
                'lambdas': lambdas,
                'nz': nz,
                'lambda0': float(lambdas[origin_idx])
            }
        return beta_info

    def _initialize_variables(self, beta_terms: list, beta_info: dict, origin_idx: int) -> tuple[Any, dict, dict]:
        mu = VectorVariable(self.program_size, 'mu', '', "Lagrangian coefficients")
        a = {}
        b = {}

        for idx, beta in enumerate(beta_terms):
            b[beta] = Variable(f'b_{idx}')
            for j in beta_info[beta]['nz']:
                if j == origin_idx:
                    continue
                a[(beta, j)] = Variable(f'a_{idx}_{j}')

        return mu, a, b

    def _add_variable_bounds(self, constraints: list, mu: Any, a: dict, b: dict) -> None:
        for j in range(self.program_size):
            constraints.append(mu[j] >= self.error_bound)
            if j > 0:
                constraints.append(mu[j] <= self.max_bound)

        for var in a.values():
            constraints.append(var >= self.error_bound)
            constraints.append(var <= self.max_bound)

        for var in b.values():
            constraints.append(var >= self.error_bound)
            constraints.append(var <= self.max_bound)

    def _g_split(self, mu: Any, mono: Any) -> tuple[Any, Any]:
        """Split G(mu)_mono into positive and negative parts.

        G(mu) = f - sum_i mu_i * g_i where g_i are original constraints.
        """
        g_plus = 0.0
        g_minus = 0.0

        f_coeff = self.prog.objective[mono]
        if f_coeff > 0:
            g_plus = g_plus + f_coeff
        elif f_coeff < 0:
            g_minus = g_minus + (-f_coeff)

        for i, cns in enumerate(self.prog.constraints, start=1):
            coeff = cns[mono]
            # Contribution is -(coeff * mu[i]).
            if coeff > 0:
                g_minus = g_minus + coeff * mu[i]
            elif coeff < 0:
                g_plus = g_plus + (-coeff) * mu[i]
        return g_plus, g_minus

    def _build_objective(self, mu: Any, a: dict, b: dict, beta_terms: list, beta_info: dict, origin_idx: int) -> Any:
        obj = 0.0
        alpha0 = self.prog.semigroup.G.identity

        for i, cns in enumerate(self.prog.constraints, start=1):
            g_i_plus_alpha0 = max(0.0, cns[alpha0])
            obj = obj + mu[i] * g_i_plus_alpha0

        for beta in beta_terms:
            lambda0 = beta_info[beta]['lambda0']
            if lambda0 <= self.error_bound:
                continue

            term = b[beta] ** (1.0 / lambda0)
            for j in beta_info[beta]['nz']:
                if j == origin_idx:
                    continue
                lam = float(beta_info[beta]['lambdas'][j])
                if lam <= self.error_bound:
                    continue
                term = term * (lam / a[(beta, j)]) ** (lam / lambda0)
            obj = obj + lambda0 * term

        return obj

    def _build_constraints(
        self,
        constraints: list,
        mu: Any,
        a: dict,
        b: dict,
        beta_terms: list,
        beta_info: dict,
        alpha: list,
        origin_idx: int
    ) -> None:
        constraints.append(mu[0] == 1.0)

        # Constraint family (1): sum_j a_{beta,j} <= G(mu)_{alpha(j)} with sign split.
        for j, alpha_j in enumerate(alpha):
            if j == origin_idx:
                continue
            lhs = 0.0
            for beta in beta_terms:
                if (beta, j) in a:
                    lhs = lhs + a[(beta, j)]

            g_plus, g_minus = self._g_split(mu, alpha_j)
            lhs_total = lhs + g_minus
            rhs_total = g_plus

            if not bool(lhs_total):
                lhs_total = lhs_total + self.error_bound
            if not bool(rhs_total):
                rhs_total = rhs_total + self.error_bound

            with SignomialsEnabled():
                cns = lhs_total <= rhs_total
            self._append_symbolic_constraint(constraints, cns)

        # Constraint family (2): product terms for lambda_0(beta) = 0 branch.
        for beta in beta_terms:
            if beta_info[beta]['lambda0'] > self.error_bound:
                continue

            lhs = 1.0
            for j in beta_info[beta]['nz']:
                if j == origin_idx:
                    continue
                lam = float(beta_info[beta]['lambdas'][j])
                if lam <= self.error_bound:
                    continue
                lhs = lhs * (a[(beta, j)] / lam) ** lam

            self._append_symbolic_constraint(constraints, lhs >= b[beta])

        # Constraint families (3) and (4): positive/negative parts of G(mu)_beta.
        for beta in beta_terms:
            g_plus, g_minus = self._g_split(mu, beta)
            if bool(g_plus):
                constraints.append(b[beta] >= g_plus)
            if bool(g_minus):
                constraints.append(b[beta] >= g_minus)

    def _solve_model(self, obj: Any, constraints: list, verbosity: int) -> None:
        mdl = Model(obj, Bounded(ConstraintSet(constraints), upper=1 / self.error_bound))
        try:
            if self.use_local_solve:
                try:
                    self.solution = mdl.localsolve(verbosity=verbosity)
                except Exception as local_exc:
                    # Fallback to global GP solve when possible.
                    try:
                        self.solution = mdl.solve(verbosity=verbosity)
                    except Exception:
                        raise
            else:
                self.solution = mdl.solve(verbosity=verbosity)
        except Exception as exc:
            raise RuntimeError(f"SONC GP solve failed: {exc}") from exc
        if self.solution is None:
            raise RuntimeError("SONC GP solver returned no solution")

    def solve(self, verbosity: int | None = None) -> float:
        """Build and solve the constrained SONC geometric program."""
        if verbosity is None:
            verbosity = self.verbosity
        try:
            delta = self._build_delta_sets()
            beta_terms = sorted(delta['=d'].union(delta['<d']), key=str)

            supports, alpha, origin_idx, hull_computed = self._build_support_points()
            # Filter beta_terms to exclude Newton polytope vertices (support exponents).
            # Only do this when the convex hull was successfully computed; in the
            # fallback case all exponents act as supports and filtering would wrongly
            # empty the set.
            if hull_computed:
                support_tuples = set(supports)
                beta_terms = [b_mon for b_mon in beta_terms
                              if tuple(self.prog.mono2ord_tuple(b_mon)) not in support_tuples]
            if not beta_terms:
                raise RuntimeError("Delta(G) is empty; SONC relaxation needs at least one interior beta term")

            beta_info = self._build_beta_info(beta_terms, supports, origin_idx)
            mu, a, b = self._initialize_variables(beta_terms, beta_info, origin_idx)

            constraints = []
            self._add_variable_bounds(constraints, mu, a, b)
            self._build_constraints(constraints, mu, a, b, beta_terms, beta_info, alpha, origin_idx)
            obj = self._build_objective(mu, a, b, beta_terms, beta_info, origin_idx)

            if verbosity > 0:
                print('obj=', obj)
                print('-' * 30)

            self._solve_model(obj, constraints, verbosity)
            self.f_sonc_g = self.prog.objective.constant() - self.solution['cost']
            return float(self.f_sonc_g)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"SONC GP solve failed: {exc}") from exc

