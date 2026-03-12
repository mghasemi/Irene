import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


if __name__ == '__main__':
    # Example 4.5 (illustrative instance)
    # Section statement is structural: New(G(mu)) = conv{0, 2d e1, ..., 2d en}.
    # Here we instantiate n=2, d=2 with simplex vertices (0,0), (4,0), (0,4).
    S = CommutativeSemigroup(['x', 'y'])
    SA = SemigroupAlgebra(S)
    x = SA['x']
    y = SA['y']

    # Supports on a 2d-scaled standard simplex plus one interior term.
    f = 1 + x ** 4 + y ** 4 - 3 * x ** 2 * y ** 2
    g1 = 1 + x ** 4 + y ** 4

    problem = OptimizationProblem(SA)
    problem.set_objective(f)
    problem.add_constraints([g1])

    sonc = SONCRelaxations(problem, verbosity=0)

    print('Example 4.5 (structural illustration)')
    print('f:', f)
    print('g1:', g1)

    supports, _, _, _ = sonc._build_support_points()
    print('\nsupport points:', supports)
    print('\nPaper note: scaled standard simplex with Ghasemi-Marshall approach')

    try:
        value = sonc.solve(verbosity=0)
        print('sonc lower bound:', value)
    except RuntimeError as exc:
        print('sonc solve status: RuntimeError')
        print('(Known structured example with numerical challenges)')
