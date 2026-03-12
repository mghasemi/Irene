import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# SOS decomposition
from sympy import *
from Irene import SDPRelaxations
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

    Rlx = SDPRelaxations([x, y, z, w])
    f = PsdMean(6, 3, [x, y, z, w], [2, 2, 1, 1])
    f = PsdMean(4, 1, [x, y, z, w], [1, 1, 1, 1])
    print(f)
    Rlx.SetObjective(f)
    #Rlx.SetObjective(x ** 3 + x ** 2 * y ** 2 + z ** 2 * x * y - x * z)
    #Rlx.AddConstraint(9 - (x ** 2 + y ** 2 + z ** 2) >= 0)
    # initiate the SDP
    Rlx.MomentsOrd(3)
    Rlx.Parallel = False
    Rlx.InitSDP()
    # solve the SDP
    Rlx.Minimize()
    print(Rlx.Solution)
    # extract decomposition
    V = Rlx.Decompose()
    #for v in V[0]:
    #    print(remove_small_coefficient_terms(v))
    #    print("----"*30)
    # test the decomposition
    """sos = 0

    # Test the decomposition
    sos = 0
    for v in V:
        # for g0 = 1
        if v == 0:
            sos = expand(Rlx.ReduceExp(sum([p ** 2 for p in V[v]])))
        # for g1, the constraint
        else:
            sos = expand(Rlx.ReduceExp(
                sos + Rlx.Constraints[v - 1] * sum([p ** 2 for p in V[v]])))
            if v - 1 < len(Rlx.Constraints):
                sos = expand(Rlx.ReduceExp(
                    sos + Rlx.Constraints[v - 1] * sum([p ** 2 for p in V[v]])))
    sos = sos.subs(Rlx.RevSymDict)
    pln = Poly(sos).as_dict()
    pln = {ex: round(pln[ex], 5) for ex in pln}
    print(Poly(pln, (x, y, z)).as_expr())"""
