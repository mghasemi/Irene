import unittest
from unittest.mock import patch
import tempfile
import os
import subprocess
import sys

from sympy import Abs, pi, symbols
from sympy.core.relational import Equality
import numpy as np

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra, SemigroupAlgebraElement
from Irene.base import LaTeX, base as IreneBase
from Irene.program import OptimizationProblem
from Irene.relaxations import SDPRelaxations
from Irene.sdp import sdp


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

    def test_linear_combination_rejects_missing_vertices(self):
        self.problem.vertices = []

        with self.assertRaisesRegex(ValueError, 'non-empty vertices'):
            self.problem.linear_combination([1.0, 1.0])

    def test_linear_combination_rejects_singular_vertex_matrix(self):
        self.problem.vertices = [[0, 0], [1, 1], [2, 2]]

        with self.assertRaisesRegex(ValueError, 'singular vertex matrix'):
            self.problem.linear_combination([1.0, 1.0])

    def test_linear_combination_rejects_only_origin_vertices(self):
        self.problem.vertices = [[0, 0]]

        with self.assertRaisesRegex(ValueError, 'at least one non-origin vertex'):
            self.problem.linear_combination([0.0, 0.0])

    def test_linear_combination_rejects_non_one_dimensional_point(self):
        self.problem.vertices = [[0, 0], [1, 0], [0, 1]]

        with self.assertRaisesRegex(ValueError, 'one-dimensional point'):
            self.problem.linear_combination([[0.2, 0.3]])

    def test_linear_combination_rejects_dimension_mismatch(self):
        self.problem.vertices = [[0, 0], [1, 0], [0, 1]]

        with self.assertRaisesRegex(ValueError, 'point dimension mismatch'):
            self.problem.linear_combination([0.2, 0.3, 0.5])

    def test_linear_combination_rejects_non_square_vertex_matrix(self):
        self.problem.vertices = [[1, 0], [0, 1], [1, 1]]

        with self.assertRaisesRegex(ValueError, 'square vertex matrix'):
            self.problem.linear_combination([0.2, 0.3])

    def test_linear_combination_uses_non_origin_vertices(self):
        self.problem.vertices = [[0, 0], [1, 0], [0, 1]]

        coeffs = self.problem.linear_combination([0.2, 0.3])

        self.assertTrue(np.allclose(coeffs, np.array([0.2, 0.3])))

    def test_linear_combination_accepts_numpy_vertices_array(self):
        self.problem.vertices = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)

        coeffs = self.problem.linear_combination([0.2, 0.3])

        self.assertTrue(np.allclose(coeffs, np.array([0.2, 0.3])))

    def test_linear_combination_rejects_empty_numpy_vertices_array(self):
        self.problem.vertices = np.empty((0, 2), dtype=float)

        with self.assertRaisesRegex(ValueError, 'non-empty vertices'):
            self.problem.linear_combination([0.2, 0.3])

    def test_convex_combination_returns_solver_solution(self):
        self.problem.vertices = [[0, 0], [1, 0], [0, 1]]

        class DummyResult:
            def __init__(self):
                self.success = True
                self.x = np.array([0.5, 0.2, 0.3])

        with patch('Irene.program.optimize.linprog', return_value=DummyResult()) as linprog:
            coeffs = self.problem.convex_combination(np.array([0.2, 0.3]))

        self.assertTrue(np.allclose(coeffs, np.array([0.5, 0.2, 0.3])))
        self.assertAlmostEqual(linprog.call_args.kwargs['b_eq'][-1], 1.0)

    def test_convex_combination_returns_none_when_solver_fails(self):
        self.problem.vertices = [[0, 0], [1, 0], [0, 1]]

        class DummyResult:
            def __init__(self):
                self.success = False
                self.x = np.array([])

        with patch('Irene.program.optimize.linprog', return_value=DummyResult()):
            coeffs = self.problem.convex_combination([2.0, 2.0])

        self.assertIsNone(coeffs)

    def test_convex_combination_rejects_missing_vertices(self):
        self.problem.vertices = []

        with self.assertRaisesRegex(ValueError, 'non-empty vertices'):
            self.problem.convex_combination([0.2, 0.3])

    def test_convex_combination_rejects_non_one_dimensional_point(self):
        self.problem.vertices = [[0, 0], [1, 0], [0, 1]]

        with self.assertRaisesRegex(ValueError, 'one-dimensional point'):
            self.problem.convex_combination([[0.2, 0.3]])

    def test_convex_combination_rejects_dimension_mismatch(self):
        self.problem.vertices = [[0, 0], [1, 0], [0, 1]]

        with self.assertRaisesRegex(ValueError, 'point dimension mismatch'):
            self.problem.convex_combination([0.2])

    def test_convex_combination_accepts_numpy_vertices_array(self):
        self.problem.vertices = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)

        class DummyResult:
            def __init__(self):
                self.success = True
                self.x = np.array([0.5, 0.2, 0.3])

        with patch('Irene.program.optimize.linprog', return_value=DummyResult()):
            coeffs = self.problem.convex_combination([0.2, 0.3])

        self.assertTrue(np.allclose(coeffs, np.array([0.5, 0.2, 0.3])))

    def test_in_newton_rejects_empty_vertices(self):
        self.problem.vertices = []

        with self.assertRaisesRegex(ValueError, 'non-empty vertices'):
            self.problem.in_newton([0.5, 0.5])

    def test_in_newton_rejects_degenerate_vertices(self):
        # Coplanar points in 3D cannot form a valid Delaunay triangulation
        self.problem.vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]

        with self.assertRaisesRegex(ValueError, 'Delaunay triangulation failed'):
            self.problem.in_newton([0.5, 0.5, 0])

    def test_in_newton_accepts_numpy_vertices_array(self):
        self.problem.vertices = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)

        self.assertTrue(self.problem.in_newton([0.2, 0.2]))

    def test_newton_polytope_insufficient_points_guard(self):
        # Verify guard works when setting vertices with insufficient dimensional support
        # 3D problem should reject only 2 points (need at least 4 for 3D polytope)
        sg = CommutativeSemigroup(['x', 'y', 'z'])
        self.problem.semigroup = sg
        # Manually set vertices to just 2 points (bypassing newton() for this test)
        self.problem.vertices = [[0, 0, 0], [1, 1, 1]]
        
        # in_newton should fail on degenerate geometry
        with self.assertRaisesRegex(ValueError, 'Delaunay triangulation failed'):
            self.problem.in_newton([0.5, 0.5, 0.5])


class TestBaseFixes(unittest.TestCase):
    def test_latex_prefers_duck_typed_latex_method(self):
        class DummyLatexObject:
            def __latex__(self):
                return 'dummy-latex'

        self.assertEqual(LaTeX(DummyLatexObject()), 'dummy-latex')

    def test_latex_handles_sympy_objects(self):
        x = symbols('x')
        self.assertEqual(LaTeX(x), 'x')

    def test_available_sdp_solvers_non_windows_uses_binary_lookup(self):
        base_obj = IreneBase()
        base_obj.os = 'linux'

        def fake_which(binary_name):
            if binary_name == 'sdpa':
                return '/usr/bin/sdpa'
            return None

        with patch.dict(sys.modules, {'cvxopt': object()}), \
                patch.object(base_obj, 'which', side_effect=fake_which):
            existing = base_obj.AvailableSDPSolvers()

        self.assertEqual(existing, ['CVXOPT', 'SDPA'])

    def test_available_sdp_solvers_windows_uses_configured_paths(self):
        base_obj = IreneBase()
        base_obj.os = 'win32'
        base_obj.Path = {'sdpa': 'C:/sdpa.exe', 'csdp': 'C:/csdp.exe'}

        def fake_isfile(path):
            return path == 'C:/sdpa.exe'

        with patch.dict(sys.modules, {'cvxopt': object()}), \
                patch('os.path.isfile', side_effect=fake_isfile):
            existing = base_obj.AvailableSDPSolvers()

        self.assertEqual(existing, ['CVXOPT', 'SDPA'])


class TestRelaxationsFixes(unittest.TestCase):
    class FakeQueue:
        def __init__(self, values=None, error=None):
            self.values = list(values or [])
            self.error = error
            self.closed = False
            self.joined = False

        def get(self):
            if self.error is not None:
                raise self.error
            return self.values.pop(0)

        def close(self):
            self.closed = True

        def join_thread(self):
            self.joined = True

    class FakeProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.started = False
            self.joined = False
            self.terminated = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started and not self.joined and not self.terminated

        def terminate(self):
            self.terminated = True

        def join(self):
            self.joined = True

    def setUp(self):
        self.relaxation = SDPRelaxations.__new__(SDPRelaxations)
        self.relaxation.NumCores = 2

    def test_parallel_calpha_results_joins_workers_on_success(self):
        queue = self.FakeQueue(values=[[0, 'alpha0'], [1, 'alpha1']])
        processes = []

        def make_process(target, args):
            process = self.FakeProcess(target, args)
            processes.append(process)
            return process

        with patch('Irene.relaxations.mp.Queue', return_value=queue), \
                patch('Irene.relaxations.mp.Process', side_effect=make_process):
            results = self.relaxation._parallel_calpha_results(['e0', 'e1'], 'mmnt')

        self.assertEqual(results, ['alpha0', 'alpha1'])
        self.assertTrue(all(process.started for process in processes))
        self.assertTrue(all(process.joined for process in processes))
        self.assertFalse(any(process.terminated for process in processes))
        self.assertTrue(queue.closed)
        self.assertTrue(queue.joined)

    def test_parallel_calpha_results_terminates_workers_on_failure(self):
        queue = self.FakeQueue(error=KeyboardInterrupt())
        processes = []

        def make_process(target, args):
            process = self.FakeProcess(target, args)
            processes.append(process)
            return process

        with patch('Irene.relaxations.mp.Queue', return_value=queue), \
                patch('Irene.relaxations.mp.Process', side_effect=make_process):
            with self.assertRaises(KeyboardInterrupt):
                self.relaxation._parallel_calpha_results(['e0', 'e1'], 'mmnt')

        self.assertTrue(all(process.started for process in processes))
        self.assertTrue(all(process.terminated for process in processes))
        self.assertTrue(all(process.joined for process in processes))
        self.assertTrue(queue.closed)
        self.assertTrue(queue.joined)

    def test_commit_stage_state_commits_once_on_success(self):
        calls = []

        def commit_stub(blk, c, idx):
            calls.append((blk, c, idx))

        self.relaxation.Commit = commit_stub
        self.relaxation._commit_stage_state('blk', 'c', 3)

        self.assertEqual(calls, [('blk', 'c', 3)])

    def test_commit_stage_state_retries_then_raises_keyboard_interrupt(self):
        calls = []

        def commit_stub(blk, c, idx):
            calls.append((blk, c, idx))
            if len(calls) == 1:
                raise RuntimeError('first commit failed')

        self.relaxation.Commit = commit_stub

        with self.assertRaises(KeyboardInterrupt):
            self.relaxation._commit_stage_state('blk', 'c', 4)

        self.assertEqual(calls, [('blk', 'c', 4), ('blk', 'c', 4)])

    def test_add_constraint_accepts_equality_subclass(self):
        class EqualitySubclass(Equality):
            pass

        x = symbols('x')
        relaxation = SDPRelaxations([x], name='eq_subclass_relaxation')
        relaxation.AddConstraint(EqualitySubclass(x, 1))

        self.assertEqual(len(relaxation.Constraints), 2)
        self.assertEqual(len(relaxation.CnsDegs), 2)

    def test_localized_moment_rejects_non_polynomial_localizer(self):
        x = symbols('x')
        relaxation = SDPRelaxations([x], name='localized_moment_validation')
        relaxation.MmntOrd = 1
        localizer = Abs(relaxation.AuxSyms[0])

        with self.assertRaises(ValueError):
            relaxation.LocalizedMoment(localizer)

        with self.assertRaises(ValueError):
            relaxation.LocalizedMoment_(localizer)

    def test_save_resume_state_roundtrip_preserves_checkpoint(self):
        x = symbols('x')
        with tempfile.TemporaryDirectory() as tmpdir:
            base_name = os.path.join(tmpdir, 'persist_roundtrip')
            relaxation = SDPRelaxations([x], name=base_name)
            relaxation.Stage = 'MomConst'
            relaxation.InitIdx = 7

            relaxation.SaveState()

            self.assertTrue(os.path.exists(base_name + '.rlx'))
            resumed = relaxation.Resume()
            self.assertEqual(resumed.PrevStage, 'MomConst')
            self.assertEqual(resumed.LastIdxVal, 7)
            self.assertEqual(relaxation.State(), ('MomConst', 7))

    def test_init_sdp_keyboard_interrupt_persists_latest_checkpoint(self):
        x = symbols('x')
        with tempfile.TemporaryDirectory() as tmpdir:
            base_name = os.path.join(tmpdir, 'persist_interrupt')
            relaxation = SDPRelaxations([x], name=base_name)
            relaxation.Stage = 'PSDMom'
            relaxation.InitIdx = 3
            relaxation.Parallel = True

            with patch.object(SDPRelaxations, 'pInitSDP', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    relaxation.InitSDP()

            self.assertTrue(os.path.exists(base_name + '.rlx'))
            self.assertEqual(relaxation.State(), ('PSDMom', 3))


class TestSdpFixes(unittest.TestCase):
    def test_solver_path_is_copied_on_init(self):
        solver_path = {'csdp': 'custom_csdp', 'sdpa': 'custom_sdpa'}
        with patch.object(sdp, 'AvailableSDPSolvers', return_value=['CVXOPT']):
            problem = sdp(solver='cvxopt', solver_path=solver_path)

        solver_path['csdp'] = 'mutated'
        self.assertIsNot(problem.Path, solver_path)
        self.assertEqual(problem.Path['csdp'], 'custom_csdp')

    def test_invalid_solver_raises_value_error(self):
        with self.assertRaises(ValueError):
            sdp(solver='invalid_solver')

    def test_sparse_writer_ignores_near_zero_entries(self):
        with patch.object(sdp, 'AvailableSDPSolvers', return_value=['CVXOPT']):
            problem = sdp(solver='cvxopt')

        problem.BlockStruct = [2]
        problem.b = [1.0]
        problem.C = [np.array([[1.0, 1e-14], [1e-14, 0.0]])]
        problem.A = [[np.array([[0.0, 2e-12], [2e-12, 0.0]])]]

        fd, path = tempfile.mkstemp(suffix='.dat-s')
        os.close(fd)
        try:
            problem.write_sdpa_dat_sparse(path)
            with open(path, 'r') as data_file:
                lines = [line.strip() for line in data_file if line.strip()]
        finally:
            os.remove(path)

        data_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                try:
                    int(parts[0])
                    int(parts[1])
                    int(parts[2])
                    int(parts[3])
                    float(parts[4])
                except ValueError:
                    continue
                data_lines.append(line)

        self.assertEqual(len(data_lines), 2)

    def test_sparse_writer_coerces_symbolic_objective_coefficients(self):
        with patch.object(sdp, 'AvailableSDPSolvers', return_value=['CVXOPT']):
            problem = sdp(solver='cvxopt')

        problem.BlockStruct = [1]
        problem.b = [-2 * pi ** 2]
        problem.C = [np.array([[1.0]])]
        problem.A = [[np.array([[0.0]])]]

        fd, path = tempfile.mkstemp(suffix='.dat-s')
        os.close(fd)
        try:
            problem.write_sdpa_dat_sparse(path)
            with open(path, 'r') as data_file:
                lines = [line.rstrip('\n') for line in data_file]
        finally:
            os.remove(path)

        self.assertNotIn('pi', lines[3])
        self.assertAlmostEqual(float(lines[3].strip()), float(-2 * pi ** 2))

    def test_csdp_failure_raises_runtime_error_before_parsing(self):
        with patch.object(sdp, 'AvailableSDPSolvers', return_value=['CSDP']):
            problem = sdp(solver='csdp', solver_path={'csdp': 'fake_csdp'})

        problem.BlockStruct = [1]
        problem.C = [np.array([[1.0]])]
        with patch.object(problem, 'write_sdpa_dat_sparse') as write_sparse, \
                patch.object(problem, 'read_csdp_out') as read_output, \
                patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, ['fake_csdp'])):
            with self.assertRaises(RuntimeError):
                problem.csdp()

        write_sparse.assert_called_once_with('prg.dat-s')
        read_output.assert_not_called()

    def test_sdpa_failure_raises_runtime_error_before_parsing(self):
        with patch.object(sdp, 'AvailableSDPSolvers', return_value=['SDPA']):
            problem = sdp(solver='sdpa', solver_path={'sdpa': 'fake_sdpa'})

        problem.BlockStruct = [1]
        with patch.object(problem, 'sdpa_param') as write_params, \
                patch.object(problem, 'write_sdpa_dat') as write_data, \
                patch.object(problem, 'read_sdpa_out') as read_output, \
                patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, ['fake_sdpa'])):
            with self.assertRaises(RuntimeError):
                problem.sdpa()

        write_params.assert_called_once_with()
        write_data.assert_called_once_with('prg.dat')
        read_output.assert_not_called()

    def test_parse_solution_matrix_rejects_incomplete_matrix(self):
        rows = iter([
            '{{1.0,2.0}\n',
            '}\n',
        ])

        with self.assertRaises(ValueError):
            sdp.parse_solution_matrix(rows)

    def test_read_csdp_out_accepts_irregular_whitespace(self):
        with patch.object(sdp, 'AvailableSDPSolvers', return_value=['CVXOPT']):
            problem = sdp(solver='cvxopt')

        problem.BlockStruct = [2]
        problem.Info = {}
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            with open(path, 'w') as data_file:
                data_file.write('1.0   2.0   \n')
                data_file.write('1   1  1   2   3.5   \n')
                data_file.write('\n')
                data_file.write('2  1   2  1  4.5\n')

            problem.read_csdp_out(
                path,
                'Success\nPrimal objective value: 1.5\nDual objective value: 1.0\nTotal time: 0.25\n',
            )
        finally:
            os.remove(path)

        self.assertEqual(problem.Info['Status'], 'Optimal')
        self.assertEqual(problem.Info['PObj'], 1.5)
        self.assertEqual(problem.Info['DObj'], 1.0)
        self.assertEqual(problem.Info['CPU'], 0.25)
        self.assertTrue(np.array_equal(problem.Info['y'], np.array([1.0, 2.0])))
        self.assertEqual(problem.Info['Z'][0][0][1], 3.5)
        self.assertEqual(problem.Info['X'][0][1][0], 4.5)


if __name__ == '__main__':
    unittest.main()
