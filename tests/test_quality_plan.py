import unittest

from sympy import symbols

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra, SemigroupAlgebraElement
from Irene.program import OptimizationProblem


class TestGroupRingsFixes(unittest.TestCase):
    def setUp(self):
        self.sg = CommutativeSemigroup(['x', 'y'])
        self.sga = SemigroupAlgebra(self.sg)
        self.x = self.sga['x']
        self.y = self.sga['y']

    def test_atomic_division_with_remainder_returns_none(self):
        self.assertIsNone(self.x / self.y)

    def test_atomic_getitem_with_single_term_expression(self):
        self.assertEqual(self.x[self.x.LT()], 1.0)

    def test_semigroup_one_is_identity(self):
        self.assertEqual(self.sga.one.constant(), 1.0)

    def test_semigroup_element_bool_and_zero_equality(self):
        zero = SemigroupAlgebraElement([], self.sg)
        self.assertFalse(bool(zero))
        self.assertTrue(zero == 0)

    def test_semigroup_equality_checks_full_terms(self):
        a = self.x + self.y
        b = self.x + self.y
        c = 2 * self.x + self.y
        self.assertTrue(a == b)
        self.assertFalse(a == c)


class TestProgramFixes(unittest.TestCase):
    def setUp(self):
        self.sg = CommutativeSemigroup(['x', 'y'])
        self.sga = SemigroupAlgebra(self.sg)
        self.x = self.sga['x']
        self.y = self.sga['y']
        self.problem = OptimizationProblem(self.sga)

    def test_analyse_program_reads_objective_terms_safely(self):
        self.problem.set_objective(self.x + 2 * self.y)
        self.problem.add_constraints([1 - self.x, 1 - self.y])
        self.problem.analyse_program()
        self.assertEqual(len(self.problem.objective_trms_with_positive_coefficient), 2)
        self.assertGreaterEqual(len(self.problem.constraint_terms_with_even_exponent), 1)

    def test_delta_vertex_is_explicitly_unimplemented(self):
        with self.assertRaises(NotImplementedError):
            self.problem.delta_vertex(self.x + self.y, [])

    def test_mono2ord_tuple_scalar_returns_tuple(self):
        t = self.problem.mono2ord_tuple(1)
        self.assertIsInstance(t, tuple)
        self.assertEqual(t, (0, 0))

    def test_mono2ord_tuple_rejects_multiterm_expression(self):
        with self.assertRaises(ValueError):
            self.problem.mono2ord_tuple(self.x + self.y)

    def test_to_sympy_raises_for_missing_symbol_map(self):
        expr = self.x + self.y
        with self.assertRaises(KeyError):
            self.problem.to_sympy(expr, {'x': symbols('x')})


if __name__ == '__main__':
    unittest.main()
