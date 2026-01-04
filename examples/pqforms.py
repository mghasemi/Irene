import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import *
from Irene.program import *
from Irene.geometric import *
from Irene import SDPRelaxations
import numpy as np
from math import lcm
from sympy import Poly, sympify, expand

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
                sm *= Y[idx] ** int(w[idx])
            # sm **= (1 / sum(w))
        return sm

    def PsdMean(q: int, p: int, Y: list, w: list):
        c = q
        if p != 0:
            c = lcm(q, p)
            res = Mean(q, Y, w) ** (c // q) - Mean(p, Y, w) ** (c // p)
        else:
            mod_pow = c/sum(w)
            res = Mean(q, Y, w) ** (c // q) - Mean(0, Y, [_ * mod_pow for _ in w])
        return res
    
    S = CommutativeSemigroup(['x', 'y', 'z', 'w'])#, 's', 't'])
    SA = SemigroupAlgebra(S)
    
    x = SA['x']
    y = SA['y']
    z = SA['z']
    w = SA['w']
    #s = SA['s']
    #t = SA['t']
    X = [x**4, y**4, z**4, w**4]#, s]
    optim = OptimizationProblem(SA)
    #optim.set_objective(x**6+y**6+z**6-5*x-4*y-z+8)
    #g = x**6+y**6+z**6+x**2 * y* z**2 - x**4 - y**4 - z**4 -y*z**3 - x*y**2 +2 + x**2
    #optim.set_objective(g)
    #optim.set_objective(5*x+6*y+x**3-y**2)
    #optim.add_constraints([-x*y-4*x**4-y**4+8])
    #####################################
    #f = x + z**3 +y**6 + z**6 + x**6
    #g1 = 1 - x**6 + y**6
    #####################################
    # f = PsdMean(6, 1, X, [1, 1, 1])
    #print(4*PsdMean(4, 0, X, [.5, .5, .5, .5]))
    # f = Mean(4, X, [.5, .5, .5, .5]) -x*y*z*w
    #f = x**4 + y**4 - x**2 * y**2 + x + y
    #f = PsdMean(4, 2, X, [1, 1, 1, 1])
    A = (x+w-y-x)/2
    B = (y+w-x-z)/2
    C = (z+w-x-y)/2
    D = (w-x-y-z)/2
    g = 2*(A**4 + B**4 + C**4 + D**4 - 4*A*B*C*D)
    R = x**2 * (x-w)**2 +y**2 * (y-w)**2+z**2 *(z-w)**2 +2*x*y*z* (x+y+z-2*w)
    f = PsdMean(1, 0, X, [1, 1, 1, 1])#+ PsdMean(4, 2, X, [1, 1, 1])
    f = x**4 + y**4 + z**4 + w**4 - 4*x*y*z*w
    print(g-R)
    optim.set_objective(f)
    
    gp = GPRelaxations(optim)
    gp.verbosity = 0
    rlx = SDPRelaxations.from_problem(optim)
    rlx.Parallel = False
    rlx.InitSDP()
    
    #gp.H = gp.auto_transform_matrix()
    
    #gp.H = np.array([[1, 0], [-1, 1]])
    
    #print(gp.H)
    
    print(gp.solve())
    
    #print(gp.solution)
    print('-' * 30)
    rlx.Minimize()
    print(rlx.Solution)
    
    V = rlx.Decompose()
    print(f"Number of decomposing Polynomials: {len(V[0])}")
    for v in V[0]:
        print(remove_small_coefficient_terms(v, threshold=1e-5))
        print("----"*30)
    """for v in V:
        # for g0 = 1
        if v == 0:
            sos = expand(rlx.ReduceExp(sum([p ** 2 for p in V[v]])))
        # for g1, the constraint
        else:
            sos = expand(rlx.ReduceExp(
                sos + rlx.Constraints[v - 1] * sum([p ** 2 for p in V[v]])))
    sos = sos.subs(rlx.RevSymDict)
    pln = Poly(sos).as_dict()
    pln = {ex: round(pln[ex], 5) for ex in pln}
    print(Poly(pln, (x, y, z)).as_expr())"""