import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


if __name__ == '__main__':
    # Example 4.3
    # f = 1 + x^2 z^2 + y^2 z^2 + x^2 y^2 - 8 x y z
    # g1 = x^2 y z + x y^2 z + x^2 y^2 - 2 + x y z
    S = CommutativeSemigroup(['x', 'y', 'z'])
    SA = SemigroupAlgebra(S)
    x = SA['x']
    y = SA['y']
    z = SA['z']

    f = 1 + x ** 2 * z ** 2 + y ** 2 * z ** 2 + x ** 2 * y ** 2 - 8 * x * y * z
    g1 = x ** 2 * y * z + x * y ** 2 * z + x ** 2 * y ** 2 - 2 + x * y * z

    problem = OptimizationProblem(SA)
    problem.set_objective(f)
    problem.add_constraints([g1])

    sonc = SONCRelaxations(problem, verbosity=0)

    print('Example 4.3')
    print('f:', f)
    print('g1:', g1)
    print('Paper target: f_K* = -15')

    try:
        value = sonc.solve(verbosity=0)
        print('sonc lower bound:', value)
    except RuntimeError as exc:
        print('sonc solve status: RuntimeError')
        print(exc)
