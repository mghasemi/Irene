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
    # Example 5.5 (Section 5, non-ST-polynomial via triangulation)
    #
    # f = 1 + 3*x1^2*x2^6 + 2*x1^6*x2^2 + 6*x1^2*x2^2
    #       - x1*x2^2 - 2*x1^2*x2 - 3*x1^3*x2^3
    #
    # Triangulation (verts of each simplex in bold in paper):
    #   T1: {(0,0), (2,2), (2,6), (1,2)}
    #   T2: {(0,0), (2,2), (6,2), (2,1)}
    #   T3: {(2,2), (2,6), (6,2), (3,3)}
    #
    # Equal coefficient split (variant 1):
    #   g1  = 0.5 + 1.5*x^2*y^6 + 2*x^2*y^2   - x*y^2
    #   g2  = 0.5 + x^6*y^2     + 2*x^2*y^2   - 2*x^2*y
    #   g3  =       1.5*x^2*y^6 + x^6*y^2     + 2*x^2*y^2 - 3*x^3*y^3  (no constant)
    #
    # g1 and g2 contribute to bound on f_origin; g3 certifies nonnegativity of x^2*y^6 term.
    # Paper: m*1 ≈ 0.0722, m*2 ≈ 0.3536
    #        fsonc = 1 - (0.0722 + 0.3536) = 0.5742  (paper: 0.5732)
    #
    # Improved distribution (variant 2):
    #   g̃1 = 0.25 + 2*x^2*y^6   + 1.217*x^2*y^2 - x*y^2
    #   g̃2 = 0.75 + x^6*y^2     + 3.652*x^2*y^2 - 2*x^2*y
    #   g̃3 =        x^2*y^6     + x^6*y^2       + 1.131*x^2*y^2 - 3*x^3*y^3 (no constant)
    #
    # Paper: m̃*1 ≈ 0.0801, m̃*2 ≈ 0.2616
    #        f̃sonc = 1 - (0.0801 + 0.2616) = 0.6583

    S = CommutativeSemigroup(['x', 'y'])
    SA = SemigroupAlgebra(S)
    x = SA['x']
    y = SA['y']

    f = (1
         + 3 * x**2 * y**6 + 2 * x**6 * y**2 + 6 * x**2 * y**2
         - x * y**2 - 2 * x**2 * y - 3 * x**3 * y**3)

    print('Example 5.5')
    print('f:', f)
    print()

    # ----- Variant 1: equal coefficient split -----
    g1 = 0.5 + 1.5 * x**2 * y**6 + 2 * x**2 * y**2 - x * y**2
    g2 = 0.5 + x**6 * y**2 + 2 * x**2 * y**2 - 2 * x**2 * y
    # g3 has no constant; it certifies the x^2*y^6 coefficient but is not
    # used for the lower bound on the constant term of f.
    g3 = 1.5 * x**2 * y**6 + x**6 * y**2 + 2 * x**2 * y**2 - 3 * x**3 * y**3

    print('Variant 1 (equal split):')
    print('  g1 =', g1)
    print('  g2 =', g2)
    print('  g3 =', g3, '  (no constant; certifies x^2*y^6 term)')
    print('Solving g1 and g2 for lower bound on f_origin...')
    try:
        b1 = solve_piece(SA, g1, 'g1')
        b2 = solve_piece(SA, g2, 'g2')
        fsonc = b1 + b2
        print(f'\nfsonc = {b1:.4f} + {b2:.4f} = {fsonc:.4f}')
        print(f'Paper target: fsonc ≈ 0.5732')
    except RuntimeError as exc:
        print(f'RuntimeError: {exc}')

    print()

    # ----- Variant 2: improved distribution -----
    # Shift constant split to 0.25/0.75 and redistribute x^2*y^2 coefficients.
    # Negative terms stay assigned to their triangulation: g̃1 keeps -xy², g̃2 keeps -2x^2y.
    # g̃3 receives the remaining positive coefficients (no constant).
    # g̃1 + g̃2 + g̃3 = f verified: 1 + 3x²y⁶ + 2x⁶y² + 6x²y² - xy² - 2x²y - 3x³y³
    g_tilde1 = 0.25 + 2 * x**2 * y**6 + 1.217 * x**2 * y**2 - x * y**2
    g_tilde2 = 0.75 + x**6 * y**2 + 3.652 * x**2 * y**2 - 2 * x**2 * y
    g_tilde3 = x**2 * y**6 + x**6 * y**2 + 1.131 * x**2 * y**2 - 3 * x**3 * y**3

    print('Variant 2 (improved distribution):')
    print('  g̃1 =', g_tilde1)
    print('  g̃2 =', g_tilde2)
    print('  g̃3 =', g_tilde3, '  (no constant; certifies x^2*y^6 term)')
    print('Solving g̃1 and g̃2 for lower bound on f_origin...')
    try:
        b1t = solve_piece(SA, g_tilde1, 'g̃1')
        b2t = solve_piece(SA, g_tilde2, 'g̃2')
        f_tilde_sonc = b1t + b2t
        print(f'\nf̃sonc = {b1t:.4f} + {b2t:.4f} = {f_tilde_sonc:.4f}')
        print(f'Paper target: f̃sonc ≈ 0.6583')
    except RuntimeError as exc:
        print(f'RuntimeError: {exc}')
