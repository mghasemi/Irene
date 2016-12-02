"""
minimize (1.5 - x)^2 + 100 (y - x^2)^2
subject to
        9 - x^2 >= 0,
        9 - y^2 >= 0,
"""

from sympy import *
from Irene import SDPRelaxations
# define the symbolic variables and functions
x = Symbol('x')
y = Symbol('y')

print "Relaxation method:"
# initiate the SDPRelaxations object
Rlx = SDPRelaxations([x, y])
# monomial order
Rlx.SetMonoOrd('lex')
rosen = (1.5 - x)**2 + 100 * (y - x**2)**2
# set the objective
Rlx.SetObjective(rosen)
# add constraints
Rlx.AddConstraint(9 - x**2 >= 0)
Rlx.AddConstraint(9 - y**2 >= 0)
# set the sdp solver
Rlx.SetSDPSolver('cvxopt')
# initiate the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print Rlx.Solution
# solve with scipy
print "scipy.optimize:"
from scipy.optimize import minimize
fun = lambda x: (1.5 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
cons = (
    {'type': 'ineq', 'fun': lambda x: 9 - x[0]**2},
    {'type': 'ineq', 'fun': lambda x: 9 - x[1]**2})
sol1 = minimize(fun, (0, 0), method='COBYLA', constraints=cons)
sol2 = minimize(fun, (0, 0), method='SLSQP', constraints=cons)
print sol1
print sol2
# solve with pso
print "PSO:"
from pyswarm import pso
lb = [-3.2, -3.2]
ub = [3.2, 3.2]
cns = [cn['fun'] for cn in cons]
print pso(fun, lb, ub, ieqcons=cns)
from sage.all import *
m1 = minimize_constrained(fun, cons=[cn['fun'] for cn in cons], x0=[
                          0, 0])
m2 = minimize_constrained(fun, cons=[cn['fun'] for cn in cons], x0=[
                          0, 0], algorithm='l-bfgs-b')
print "Sage:"
print "minimize_constrained (default):", m1, fun(m1)
print "minimize_constrained (l-bfgs-b):", m2, fun(m2)
