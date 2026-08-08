"""Tests for the unified RelaxationEngine API."""

import unittest
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxation_api import (
    RelaxationEngine,
    RelaxResult,
    RelaxMethod,
    relax,
    compare_all,
)


class TestRelaxResult(unittest.TestCase):
    def test_default_values(self):
        r = RelaxResult()
        self.assertEqual(r.value, -float("inf"))
        self.assertEqual(r.status, "error")
        self.assertEqual(r.error_code, 2)
        self.assertFalse(r.success)

    def test_success_property(self):
        r = RelaxResult(value=-1.5, method="sos", status="optimal", error_code=0)
        self.assertTrue(r.success)

        r_fail = RelaxResult(error_code=2)
        self.assertFalse(r_fail.success)

        r_inf = RelaxResult(value=-float("inf"), error_code=0)
        self.assertFalse(r_inf.success)

    def test_repr(self):
        r = RelaxResult(value=1.234, method="sos", status="optimal", runtime=0.5)
        repr_str = repr(r)
        self.assertIn("value=1.234000", repr_str)
        self.assertIn("method='sos'", repr_str)


class TestRelaxationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build a simple bivariate problem once for all tests."""
        cls.sg = CommutativeSemigroup(["x", "y"])
        cls.sga = SemigroupAlgebra(cls.sg)
        cls.x = cls.sga["x"]
        cls.y = cls.sga["y"]

        # Objective: x^4 + y^4 - 3*x^2*y + x^2 + y^2 (known to have min >= 0)
        cls.prog = OptimizationProblem(cls.sga)
        cls.prog.set_objective(
            cls.x**4 + cls.y**4 - 3 * cls.x**2 * cls.y + cls.x**2 + cls.y**2
        )

    def test_engine_construction(self):
        engine = RelaxationEngine(self.prog, order=1, solver="cvxopt")
        self.assertEqual(engine.order, 1)
        self.assertEqual(engine.solver, "cvxopt")

    def test_solve_sos_returns_result(self):
        engine = RelaxationEngine(self.prog, order=1, verbosity=0)
        result = engine.solve("sos")
        self.assertIsInstance(result, RelaxResult)
        self.assertEqual(result.method, "sos")
        self.assertGreaterEqual(result.value, -float("inf"))

    def test_solve_sonc_returns_result(self):
        engine = RelaxationEngine(self.prog, order=1, verbosity=0)
        result = engine.solve("sonc")
        self.assertIsInstance(result, RelaxResult)
        self.assertEqual(result.method, "sonc")

    def test_solve_unknown_method_raises(self):
        engine = RelaxationEngine(self.prog)
        with self.assertRaises(ValueError):
            engine.solve("nonexistent_method")

    def test_compare_returns_all_four(self):
        engine = RelaxationEngine(self.prog, order=1, verbosity=0)
        results = engine.compare()
        expected_keys = {
            "sos",
            "sonc",
            "sosonc_sos_first",
            "sosonc_sonc_first",
        }
        self.assertEqual(set(results.keys()), expected_keys)

    def test_relax_convenience(self):
        result = relax(self.prog, method="sonc", verbosity=0)
        self.assertIsInstance(result, RelaxResult)
        self.assertEqual(result.method, "sonc")

    def test_compare_all_convenience(self):
        results = compare_all(self.prog, order=1, verbosity=0)
        self.assertEqual(len(results), 4)


class TestRelaxMethodEnum(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(RelaxMethod.SOS.value, "sos")
        self.assertEqual(RelaxMethod.SONC.value, "sonc")

    def test_enum_as_method_arg(self):
        sg = CommutativeSemigroup(["x", "y"])
        sga = SemigroupAlgebra(sg)
        prog = OptimizationProblem(sga)
        prog.set_objective(sga["x"]**2 + sga["y"]**2)

        engine = RelaxationEngine(prog, verbosity=0)
        result = engine.solve(RelaxMethod.SONC)
        self.assertEqual(result.method, "sonc")


if __name__ == "__main__":
    unittest.main()
