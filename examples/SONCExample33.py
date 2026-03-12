import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


if __name__ == '__main__':
    # Section 3, Example 3.3 from the paper:
    # f = 1 + 2*x^2*y^4 + (1/2)*x^3*y^2
    # g1 = (1/3) - x^6*y^2
    S = CommutativeSemigroup(['x', 'y'])
    SA = SemigroupAlgebra(S)
    x = SA['x']
    y = SA['y']

    f = 1 + 2 * x ** 2 * y ** 4 + 0.5 * x ** 3 * y ** 2
    g1 = (1.0 / 3.0) - x ** 6 * y ** 2

    problem = OptimizationProblem(SA)
    problem.set_objective(f)
    problem.add_constraints([g1])

    sonc = SONCRelaxations(problem, verbosity=0)

    # Show the extracted barycentric coordinates for beta=(3,2),
    # which should align with lambda = (3/10, 3/10, 2/5).
    delta = sonc._build_delta_sets()
    beta_terms = sorted(delta['=d'].union(delta['<d']), key=str)
    supports, _, origin_idx, _ = sonc._build_support_points()
    beta_info = sonc._build_beta_info(beta_terms, supports, origin_idx)

    print('Section 3 Example 3.3')
    print('f:', f)
    print('g1:', g1)
    print('beta terms:', [str(b) for b in beta_terms])
    for beta in beta_terms:
        lambdas = beta_info[beta]['lambdas']
        print('beta:', beta)
        print('tuple:', beta_info[beta]['tuple'])
        print('lambda vector:', [float(v) for v in lambdas])
        print('lambda0:', float(beta_info[beta]['lambda0']))

    try:
        value = sonc.solve(verbosity=0)
        print('sonc lower bound:', value)
    except RuntimeError as exc:
        print('sonc solve status: RuntimeError')
        print(exc)
