import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


def solve_piece(SA, piece_poly, name):
    """Solve unconstrained SONC GP on a single ST-polynomial piece (Corollary 2.7)."""
    problem = OptimizationProblem(SA)
    problem.set_objective(piece_poly)
    sonc = SONCRelaxations(problem, verbosity=0)
    value = sonc.solve(verbosity=0)
    m_star = float(piece_poly.constant()) - value
    print(f'  {name}: constant={float(piece_poly.constant()):.4f}, '
          f'm* = {m_star:.4f}, bound contribution = {value:.4f}')
    return value


if __name__ == '__main__':
    # Example 5.4 (Section 5, non-ST-polynomial via triangulation)
    #
    # f = 6 + x1^2*x2^6 + 2*x1^4*x2^6 + x1^8*x2^2
    #       - 1.2*x1^2*x2^3 - 0.85*x1^3*x2^5
    #       - 0.9*x1^4*x2^3 - 0.73*x1^5*x2^2 - 1.14*x1^7*x2^2
    #
    # Triangulation (paper notation, vertices in each simplex):
    #   T1: {(0,0), (2,6), (4,6), (2,3), (3,5)}
    #   T2: {(0,0), (4,6), (8,2), (2,3), (4,3), (5,2), (7,2)}
    #
    # Equal coefficient split:
    #   g1 = 3 + x^2*y^6 + x^4*y^6 - 0.6*x^2*y^3 - 0.85*x^3*y^5
    #   g2 = 3 + x^4*y^6 + x^8*y^2 - 0.6*x^2*y^3
    #           - 0.9*x^4*y^3 - 0.73*x^5*y^2 - 1.14*x^7*y^2
    #
    # Paper results:  m*1 ≈ 0.2121,  m*2 ≈ 2.5193
    #                 fsonc ≈ 6 - 2.731 = 3.269
    #
    # Improved split (full weight of -1.2*x^2*y^3 into g̃1):
    #   g̃1 = 3 + x^2*y^6 + x^4*y^6 - 1.2*x^2*y^3 - 0.85*x^3*y^5
    #   g̃2 = 3 + x^4*y^6 + x^8*y^2
    #           - 0.9*x^4*y^3 - 0.73*x^5*y^2 - 1.14*x^7*y^2
    #
    # Paper result: f̃sonc ≈ 3.572

    S = CommutativeSemigroup(['x', 'y'])
    SA = SemigroupAlgebra(S)
    x = SA['x']
    y = SA['y']

    f = (6
         + x**2 * y**6 + 2 * x**4 * y**6 + x**8 * y**2
         - 1.2 * x**2 * y**3 - 0.85 * x**3 * y**5
         - 0.9 * x**4 * y**3 - 0.73 * x**5 * y**2 - 1.14 * x**7 * y**2)

    print('Example 5.4')
    print('f:', f)
    print()

    # ----- Variant 1: equal coefficient split -----
    g1 = (3
          + x**2 * y**6 + x**4 * y**6
          - 0.6 * x**2 * y**3 - 0.85 * x**3 * y**5)

    g2 = (3
          + x**4 * y**6 + x**8 * y**2
          - 0.6 * x**2 * y**3 - 0.9 * x**4 * y**3
          - 0.73 * x**5 * y**2 - 1.14 * x**7 * y**2)

    print('Equal split: g1 =', g1)
    print('             g2 =', g2)
    print('Solving...')
    try:
        b1 = solve_piece(SA, g1, 'g1')
        b2 = solve_piece(SA, g2, 'g2')
        fsonc = b1 + b2
        print(f'\nfsonc = {b1:.4f} + {b2:.4f} = {fsonc:.4f}')
        print(f'Paper target: fsonc ≈ 3.269')
    except RuntimeError as exc:
        print(f'RuntimeError: {exc}')

    print()

    # ----- Variant 2: full -1.2*x^2*y^3 weight into g̃1 -----
    g_tilde1 = (3
                + x**2 * y**6 + x**4 * y**6
                - 1.2 * x**2 * y**3 - 0.85 * x**3 * y**5)

    g_tilde2 = (3
                + x**4 * y**6 + x**8 * y**2
                - 0.9 * x**4 * y**3 - 0.73 * x**5 * y**2 - 1.14 * x**7 * y**2)

    print('Improved split: g̃1 =', g_tilde1)
    print('                g̃2 =', g_tilde2)
    print('Solving...')
    try:
        b1t = solve_piece(SA, g_tilde1, 'g̃1')
        b2t = solve_piece(SA, g_tilde2, 'g̃2')
        f_tilde_sonc = b1t + b2t
        print(f'\nf̃sonc = {b1t:.4f} + {b2t:.4f} = {f_tilde_sonc:.4f}')
        print(f'Paper target: f̃sonc ≈ 3.572')
    except RuntimeError as exc:
        print(f'RuntimeError: {exc}')
