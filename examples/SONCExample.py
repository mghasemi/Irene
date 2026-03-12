import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


if __name__ == '__main__':
    # Simple 1D SONC benchmark.
    S = CommutativeSemigroup(['x'])
    SA = SemigroupAlgebra(S)
    x = SA['x']

    problem = OptimizationProblem(SA)
    problem.set_objective(-x + x ** 2)
    problem.add_constraints([1 + x ** 2])

    sonc = SONCRelaxations(problem, verbosity=0)
    print('SONC 1D benchmark')
    print('objective:', -x + x ** 2)
    print('constraint:', 1 + x ** 2, '>= 0')
    try:
        value = sonc.solve(verbosity=0)
        print('sonc lower bound:', value)
    except RuntimeError as exc:
        print('sonc solve status: RuntimeError')
        print(exc)
