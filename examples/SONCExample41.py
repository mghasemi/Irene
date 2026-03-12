import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


if __name__ == '__main__':
    # Example 4.1
    # f = 1 + x^4 y^2 + x^2 y^4 - 3 x^2 y^2
    # g1 = x^3 y^2
    # K = {(x,y): x >= 0 or y = 0}
    S = CommutativeSemigroup(['x', 'y'])
    SA = SemigroupAlgebra(S)
    x = SA['x']
    y = SA['y']

    f = 1 + x ** 4 * y ** 2 + x ** 2 * y ** 4 - 3 * x ** 2 * y ** 2
    g1 = x ** 3 * y ** 2

    problem = OptimizationProblem(SA)
    problem.set_objective(f)
    problem.add_constraints([g1])

    sonc = SONCRelaxations(problem, verbosity=0)

    print('Example 4.1')
    print('f:', f)
    print('g1:', g1)
    print('Known target from paper: s(f,g1) = 0')

    try:
        value = sonc.solve(verbosity=0)
        print('sonc lower bound:', value)
    except RuntimeError as exc:
        print('sonc solve status: RuntimeError')
        print(exc)
