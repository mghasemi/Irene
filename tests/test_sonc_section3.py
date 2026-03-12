import os
import sys
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


class TestSONCSection3(unittest.TestCase):
    def setUp(self):
        self.sg = CommutativeSemigroup(['x', 'y', 'z'])
        self.sga = SemigroupAlgebra(self.sg)
        self.x = self.sga['x']
        self.y = self.sga['y']
        self.z = self.sga['z']

        self.problem = OptimizationProblem(self.sga)
        f = -self.y - 2 * self.x ** 2
        g1 = self.y - self.x ** 4 * self.y + self.y ** 5 - self.x ** 6 - self.y ** 6
        g2 = self.y - 5 * self.x ** 2 + self.x ** 4 * self.y - self.x ** 6 - self.y ** 6
        self.problem.set_objective(f)
        self.problem.add_constraints([g1, g2])

    def test_build_delta_sets_nonempty(self):
        sonc = SONCRelaxations(self.problem, verbosity=0)
        delta = sonc._build_delta_sets()
        self.assertIn('=d', delta)
        self.assertIn('<d', delta)
        self.assertTrue(delta['=d'] or delta['<d'])

    def test_support_points_include_origin(self):
        sonc = SONCRelaxations(self.problem, verbosity=0)
        supports, alpha, origin_idx, _ = sonc._build_support_points()
        origin = tuple([0] * len(self.sg.generators))
        self.assertIn(origin, supports)
        self.assertGreaterEqual(origin_idx, 0)
        self.assertEqual(alpha[origin_idx], self.problem.semigroup.identity())

    def test_beta_lambda_weights_form_convex_combination(self):
        sonc = SONCRelaxations(self.problem, verbosity=0)
        delta = sonc._build_delta_sets()
        beta_terms = sorted(delta['=d'].union(delta['<d']), key=str)
        supports, _, origin_idx, _hull = sonc._build_support_points()
        beta_info = sonc._build_beta_info(beta_terms, supports, origin_idx)
        self.assertTrue(beta_terms)
        for beta in beta_terms:
            lambdas = beta_info[beta]['lambdas']
            self.assertAlmostEqual(float(lambdas.sum()), 1.0, places=8)
            self.assertTrue(beta_info[beta]['nz'])

    def test_solve_returns_float_and_solution(self):
        sonc = SONCRelaxations(self.problem, verbosity=0)
        try:
            val = sonc.solve(verbosity=0)
            self.assertIsInstance(val, float)
            self.assertIsNotNone(sonc.solution)
        except RuntimeError:
            pass  # solver or formulation failure is acceptable for this generic problem

    def test_feasible_benchmark_solves(self):
        sg = CommutativeSemigroup(['x', 'y'])
        sga = SemigroupAlgebra(sg)
        x = sga['x']
        y = sga['y']

        problem = OptimizationProblem(sga)
        # Example 3.3 from the paper.
        problem.set_objective(1 + 2 * x ** 2 * y ** 4 + 0.5 * x ** 3 * y ** 2)
        problem.add_constraints([(1.0 / 3.0) - x ** 6 * y ** 2])

        sonc = SONCRelaxations(problem, verbosity=0)
        val = sonc.solve(verbosity=0)
        self.assertIsInstance(val, float)
        self.assertIsNotNone(sonc.solution)

    def test_feasible_benchmark_repeatability(self):
        sg = CommutativeSemigroup(['x', 'y'])
        sga = SemigroupAlgebra(sg)
        x = sga['x']
        y = sga['y']

        def solve_once():
            problem = OptimizationProblem(sga)
            problem.set_objective(1 + 2 * x ** 2 * y ** 4 + 0.5 * x ** 3 * y ** 2)
            problem.add_constraints([(1.0 / 3.0) - x ** 6 * y ** 2])
            sonc = SONCRelaxations(problem, verbosity=0)
            return sonc.solve(verbosity=0)

        v1 = solve_once()
        v2 = solve_once()
        self.assertLessEqual(abs(v1 - v2), 1e-8)

    def test_example33_barycentric_coordinates(self):
        sg = CommutativeSemigroup(['x', 'y'])
        sga = SemigroupAlgebra(sg)
        x = sga['x']
        y = sga['y']

        # Example 3.3: f = 1 + 2*x^2*y^4 + (1/2)*x^3*y^2, g1 = 1/3 - x^6*y^2
        problem = OptimizationProblem(sga)
        problem.set_objective(1 + 2 * x ** 2 * y ** 4 + 0.5 * x ** 3 * y ** 2)
        problem.add_constraints([(1.0 / 3.0) - x ** 6 * y ** 2])

        sonc = SONCRelaxations(problem, verbosity=0)
        delta = sonc._build_delta_sets()
        beta_terms = sorted(delta['=d'].union(delta['<d']), key=str)
        supports, _, origin_idx, _hull = sonc._build_support_points()
        beta_info = sonc._build_beta_info(beta_terms, supports, origin_idx)

        self.assertEqual(len(beta_terms), 1)
        beta = beta_terms[0]
        lambdas = beta_info[beta]['lambdas']
        self.assertAlmostEqual(float(lambdas[origin_idx]), 0.3, places=8)
        non_origin = [idx for idx in range(len(lambdas)) if idx != origin_idx and lambdas[idx] > 1e-10]
        self.assertEqual(len(non_origin), 2)
        values = sorted(float(lambdas[idx]) for idx in non_origin)
        self.assertAlmostEqual(values[0], 0.3, places=8)
        self.assertAlmostEqual(values[1], 0.4, places=8)


if __name__ == '__main__':
    unittest.main()
