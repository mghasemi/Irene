import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


if __name__ == '__main__':
    # Example 4.2
    # f = 1 + x^4 y^2 + x y
    # g1 = 1/2 + x^2 y^4 - x^2 y^6
    S = CommutativeSemigroup(['x', 'y'])
    SA = SemigroupAlgebra(S)
    x = SA['x']
    y = SA['y']

    f = 1 + x ** 4 * y ** 2 + x * y
    g1 = 0.5 + x ** 2 * y ** 4 - x ** 2 * y ** 6

    problem = OptimizationProblem(SA)
    problem.set_objective(f)
    problem.add_constraints([g1])

    sonc = SONCRelaxations(problem, verbosity=0)

    print('Example 4.2')
    print('f:', f)
    print('g1:', g1)
    print('Paper reports f_0 - gamma_sonc ~= 0.4474')

    try:
        value = sonc.solve(verbosity=0)
        print('sonc lower bound:', value)
    except RuntimeError as exc:
        print('sonc solve status: RuntimeError')
        print(exc)
