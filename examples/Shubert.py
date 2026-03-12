"""
Minimize the Shubert function
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scipy.optimize
#from sage.all import *
from pyswarm import pso
from sympy import *
from Irene import *
import numpy as np

if __name__ == '__main__':
    x = Symbol('x')
    y = Symbol('y')
    s1 = Symbol('s1')
    c1 = Symbol('c1')
    s2 = Symbol('s2')
    c2 = Symbol('c2')
    f = sum([cos((j + 1) * x + j) for j in range(1, 6)]) * \
        sum([cos((j + 1) * y + j) for j in range(1, 6)])
    print(f)
    obj = N(expand(f, trig=True).subs(
        {sin(x): s1, cos(x): c1, sin(y): s2, cos(y): c2}))
    rels = [s1 ** 2 + c1 ** 2 - 1, s2 ** 2 + c2 ** 2 - 1]
    Rlx = SDPRelaxations([s1, c1, s2, c2], rels)
    Rlx.SetObjective(obj)
    Rlx.AddConstraint(1 - s1 ** 2 >= 0)
    Rlx.AddConstraint(1 - s2 ** 2 >= 0)
    Rlx.Parallel = False
    Rlx.InitSDP()
    # solve the SDP
    Rlx.Minimize()
    print(Rlx.Solution)
    g = lambda x: sum([np.cos((j + 1) * x[0] + j) for j in range(1, 6)]) * sum([np.cos((j + 1) * x[1] + j) for j in range(1, 6)])
    x0 = (-5, 5)
    cons = (
        {'type': 'ineq', 'fun': lambda x: 100 - x[0] ** 2},
        {'type': 'ineq', 'fun': lambda x: 100 - x[1] ** 2})
    sol1 = scipy.optimize.minimize(g, x0, method='COBYLA', constraints=cons)
    sol2 = scipy.optimize.minimize(g, x0, method='SLSQP', constraints=cons)
    print("solution according to 'COBYLA':")
    print(sol1)
    print("solution according to 'SLSQP':")
    print(sol2)

    # m1 = minimize_constrained(g, cons=[cn['fun'] for cn in cons], x0=x0)
    # m2 = minimize_constrained(g, cons=[cn['fun']
    #                                    for cn in cons], x0=x0, algorithm='l-bfgs-b')
    # print("Sage:")
    # print("minimize_constrained (default):", m1, g(m1))
    # print("minimize_constrained (l-bfgs-b):", m2, g(m2))
    print("PSO:")

    lb = [-10, -10]
    ub = [10, 10]
    cns = [cn['fun'] for cn in cons]
    print(pso(g, lb, ub, ieqcons=cns))