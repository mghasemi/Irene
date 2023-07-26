"""
minimize -(sin x - 1)^3 - (sin x - cos y)^4 - (cos y - 3)^2
subject to
        10 - (sin x - 1)^2 >= 0,
        10 - (sin x - cos y)^2 >= 0,
        10 - (cos y - 3)^2 >= 0.
"""

from scipy.optimize import minimize
from pyswarm import pso
from sympy import *
from Irene import SDPRelaxations
# define the symbolic variables and functions
x = Symbol('x')
f = Function('f')(x)
g = Function('g')(x)
h = Function('h')(x)
k = Function('k')(x)

print("Relaxation method:")
# introduce the relation between symbols
rels = [f**2 + g**2 - 1, h**2 + k**2 - 1]
# initiate the SDPRelaxations object
Rlx = SDPRelaxations([f, g, h, k], rels)
# monomial order
Rlx.SetMonoOrd('lex')
# set the objective
Rlx.SetObjective(-(f - 1)**3 - (f - k)**4 - (k - 3)**2)
# add constraints
Rlx.AddConstraint(10 - (f - 1)**2 >= 0)
Rlx.AddConstraint(10 - (f - k)**2 >= 0)
Rlx.AddConstraint(10 - (k - 3)**2 >= 0)
# set the sdp solver
Rlx.SetSDPSolver('cvxopt')
# initiate the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print(Rlx.Solution)
# solve with scipy
print("scipy.optimize:")
fun = lambda x: -(sin(x[0]) - 1)**3 - (sin(x[0]) -
                                       cos(x[1]))**4 - (cos(x[1]) - 3)**2
cons = (
    {'type': 'ineq', 'fun': lambda x: 10 - (sin(x[0]) - 1)**2},
    {'type': 'ineq', 'fun': lambda x: 10 - (sin(x[0]) - cos(x[1]))**2},
    {'type': 'ineq', 'fun': lambda x: 10 - (cos(x[1]) - 3)**2})
sol1 = minimize(fun, (0, 0), method='COBYLA', constraints=cons)
sol2 = minimize(fun, (0, 0), method='SLSQP', constraints=cons)
print(sol1)
print(sol2)
# solve with pso
print("PSO:")
lb = [-3.2, -3.2]
ub = [3.2, 3.2]
cons = [cons[0]['fun'], cons[1]['fun'], cons[2]['fun']]
print(pso(fun, lb, ub, ieqcons=cons))
