"""
Minimize the Rosenbrock function
"""

import scipy.optimize
from pyswarm import pso
from sympy import *
from Irene import SDPRelaxations

NumVars = 9
# define the symbolic variables and functions
x = [Symbol('x%d' % i) for i in range(NumVars)]

print("Relaxation method:")
# initiate the SDPRelaxations object
Rlx = SDPRelaxations(x)
# monomial order
Rlx.SetMonoOrd('lex')
rosen = sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i])
             ** 2 for i in range(NumVars - 1)])
# set the objective
Rlx.SetObjective(rosen)
# add constraints
for i in range(NumVars):
    Rlx.AddConstraint(9 - x[i] ** 2 >= 0)
# set the sdp solver
Rlx.SetSDPSolver('cvxopt')
# initiate the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print(Rlx.Solution)
# solve with scipy
print("scipy.optimize:")

fun = lambda x: sum([100 * (x[i + 1] - x[i] ** 2) ** 2 +
                     (1 - x[i]) ** 2 for i in range(NumVars - 1)])
cons = [
    {'type': 'ineq', 'fun': lambda x: 9 - x[i] ** 2} for i in range(NumVars)]
x0 = tuple([0 for _ in range(NumVars)])
sol1 = scipy.optimize.minimize(fun, x0, method='COBYLA', constraints=cons)
sol2 = scipy.optimize.minimize(fun, x0, method='SLSQP', constraints=cons)
print(sol1)
print(sol2)
# solve with pso
print("PSO:")

lb = [-3 for i in range(NumVars)]
ub = [3 for i in range(NumVars)]
cns = [cn['fun'] for cn in cons]
print(pso(fun, lb, ub, ieqcons=cns))
