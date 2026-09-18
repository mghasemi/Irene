import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sosonc import SOSONCRelaxations


def build_schick_separating_problem():
    """Build Schick's separating SOS+SONC example.

    f = 1/2 * (1 + 2xy + x^2 y)^2 + M,
    M = x^4 y^2 + x^2 y^4 + 1 - 3 x^2 y^2.
    """
    sg = CommutativeSemigroup(['x', 'y'])
    sga = SemigroupAlgebra(sg)
    x = sga['x']
    y = sga['y']

    motzkin = x ** 4 * y ** 2 + x ** 2 * y ** 4 + 1 - 3 * x ** 2 * y ** 2
    poly = 0.5 * (1 + 2 * x * y + x ** 2 * y) ** 2 + motzkin

    problem = OptimizationProblem(sga)
    problem.set_objective(poly)
    return problem


def run_example():
    # Reported by Schick toolbox runs (LightRAG ref):
    # SOS: infeasible (-inf), SONC: about -2.9878,
    # direct SOS+SONC: about 7.5e-8.
    expected_sonc = -2.9878

    problem = build_schick_separating_problem()
    engine = SOSONCRelaxations(problem, verbosity=0, relaxation_order=3)

    sos = engine.globalMinSOS()
    sonc = engine.globalMinSONC()
    sos_first = engine.globalMinSOSPSONC(first='sos')
    sonc_first = engine.globalMinSOSPSONC(first='sonc')

    print('Schick separating polynomial benchmark')
    print('SOS      :', sos.val, sos.status, sos.error_code)
    print('SONC     :', sonc.val, sonc.status, sonc.error_code)
    print('SOS-first:', sos_first.val, sos_first.status, sos_first.error_code)
    print('SONC-first:', sonc_first.val, sonc_first.status, sonc_first.error_code)

    # Consistency checks w.r.t. Schick values and Irene implementation scope.
    if sos.status not in ('infeasible', 'error'):
        raise AssertionError('Expected SOS to be infeasible/error on this example')

    if sonc.status != 'optimal' or math.isinf(sonc.val):
        raise AssertionError('Expected finite optimal SONC bound')

    if abs(sonc.val - expected_sonc) > 5e-3:
        raise AssertionError(
            f'SONC value mismatch: got {sonc.val}, expected around {expected_sonc}'
        )

    # Current Irene SOS+SONC implementation uses two-step preprocess variants,
    # not the direct joint SOS+SONC cone program from Schick's MATLAB toolbox.
    if abs(sos_first.val - sonc.val) > 1e-6:
        raise AssertionError('Expected SOS-first fallback to match SONC bound here')
    if abs(sonc_first.val - sonc.val) > 1e-6:
        raise AssertionError('Expected SONC-first to match SONC bound here')

    print('Consistency checks passed.')


if __name__ == '__main__':
    run_example()
