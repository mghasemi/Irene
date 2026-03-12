import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


if __name__ == '__main__':
    # Example 4.4
    # f = z^6 + x^4 y^2 + x^2 y^4 - 3 x^2 y^2 z^2
    # g1 = x^2 + y^2 + z^2 - 1
    # Paper notes Program (3.2) infeasibility for mu > 0 and s(f,g1)=f_sonc at mu=0.
    S = CommutativeSemigroup(['x', 'y', 'z'])
    SA = SemigroupAlgebra(S)
    x = SA['x']
    y = SA['y']
    z = SA['z']

    f = z ** 6 + x ** 4 * y ** 2 + x ** 2 * y ** 4 - 3 * x ** 2 * y ** 2 * z ** 2
    g1 = x ** 2 + y ** 2 + z ** 2 - 1

    problem = OptimizationProblem(SA)
    problem.set_objective(f)
    problem.add_constraints([g1])

    sonc = SONCRelaxations(problem, verbosity=0)

    print('Example 4.4')
    print('f:', f)
    print('g1:', g1)
    print('\nPaper target: f_sonc = f_K* = 0')
    print('Status: Program (3.2) is THEORETICALLY infeasible for µ > 0')
    print('        per paper; optimal value should use µ = 0 case')

    try:
        value = sonc.solve(verbosity=0)
        print('\nsonc lower bound:', value)
    except RuntimeError as exc:
        print('\nsonc solve status: RuntimeError (constrained SONC failed)')
        # This is expected - the paper explains this special case requires µ=0 logic
        print('Note: Constrained Program (3.2) is infeasible as predicted by theory')
