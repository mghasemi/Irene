import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# SOS decomposition
from sympy import *
from Irene import SDPRelaxations, get_gram_matrix, is_psd_numeric, is_psd_symbolic, find_psd_gram_matrix, numpy_to_latex
from math import lcm

if __name__ == '__main__':
    

    def remove_small_coefficient_terms(expr, threshold=1e-6):
        """
        Removes terms from a sympy expression where the coefficient's
        absolute value is less than a given threshold.

        Args:
            expr: The sympy expression.
            threshold: The tolerance for the coefficients.

        Returns:
            A new sympy expression with small-coefficient terms removed.
        """
        p = Poly(expr)
        
        # as_dict() gives {exponent_tuple: coefficient}
        terms_dict = p.as_dict()
        
        # Filter the dictionary
        filtered_terms = {
            exponents: coeff 
            for exponents, coeff in terms_dict.items() 
            if abs(coeff) >= threshold
        }
        
        # Recreate the polynomial from the filtered terms
        if not filtered_terms:
            return sympify(0)
            
        return Poly(filtered_terms, p.gens).as_expr()

    def Mean(p: int, Y: list, w: list):
        sm = 0
        if p!=0:
            for idx in range(len(Y)):
                sm += w[idx] * Y[idx] ** p
            sm /= sum(w)
        else:
            sm = 1
            for idx in range(len(Y)):
                sm *= Y[idx] ** w[idx]
            # sm **= (1 / sum(w))
        return sm

    def PsdMean(q: int, p: int, Y: list, w: list):
        c = q
        if p != 0:
            c = lcm(q, p)
            res = Mean(q, Y, w) ** (c // q) - Mean(p, Y, w) ** (c // p)
        else:
            res = Mean(q, Y, w) ** (c // q) - Mean(p, Y, w) ** (c // int(sum(w)))
        return res

    # define the symbolic variables and functions
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    w = Symbol('w')

    Rlx = SDPRelaxations([x, y, z])
    Rlx.Parallel = False
    f = PsdMean(6, 1, [x, y, z], [2, 2, 2])
    print(f)
    Q1, Q2 = get_gram_matrix(f)
    print(Q1)
    print(is_psd_numeric(Q1))
    print(is_psd_symbolic(Q2))
    Q3 = find_psd_gram_matrix(f)
    print(numpy_to_latex(Q3))
    Rlx.SetObjective(f)
    Rlx.InitSDP()
    Rlx.Minimize()
    print(Rlx.Solution)
 