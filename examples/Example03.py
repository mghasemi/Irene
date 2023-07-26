"""
minimize x^2 * e^(-y) - e^y
subject to:
        1 - x^2 >= 0,
        y >= -1,
        y <= 2.
"""

from scipy.optimize import minimize
from pyswarm import pso
from sympy import *
from Irene import *
# define the symbolic variables and functions
x = Symbol('x')
e = Function('e')(x)
f = Function('f')(x)

# introduce the relation between symbols
rels = [e * f - 1]
# initiate the SDPRelaxations object
Rlx = SDPRelaxations([x, e, f], rels)
# monomial order
Rlx.SetMonoOrd('lex')
# set the objective
Rlx.SetObjective(x**2 - e)
# add constraints
Rlx.AddConstraint(1 - x**2 >= 0)
Rlx.AddConstraint(e <= exp(2))
Rlx.AddConstraint(e >= exp(-1))
# set the sdp solver
Rlx.SetSDPSolver('cvxopt')
# initiate the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print(Rlx.Solution)
# solve with scipy
fun = lambda x: x[0]**2 * exp(-x[1]) - exp(x[1])
cons = (
    {'type': 'ineq', 'fun': lambda x: 1 - x[0]**2},
    {'type': 'ineq', 'fun': lambda x: x[1] + 1},
    {'type': 'ineq', 'fun': lambda x: 2 - x[1]})
sol1 = minimize(fun, (0, 0), method='COBYLA', constraints=cons)
sol2 = minimize(fun, (0, 0), method='SLSQP', constraints=cons)
print("solution according to 'COBYLA':")
print(sol1)
print("solution according to 'SLSQP':")
print(sol2)
# solve with pso
lb = [-1, -1]
ub = [1, 2]
cns = [cons[0]['fun'], cons[1]['fun']]
print("PSO:")
print(pso(fun, lb, ub, ieqcons=cns))
