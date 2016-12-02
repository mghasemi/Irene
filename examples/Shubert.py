"""
Minimize the Shubert function
"""
from sympy import *
from Irene import *
x = Symbol('x')
y = Symbol('y')
s1 = Symbol('s1')
c1 = Symbol('c1')
s2 = Symbol('s2')
c2 = Symbol('c2')
f = sum([cos((j + 1) * x + j) for j in range(1, 6)]) * \
    sum([cos((j + 1) * y + j) for j in range(1, 6)])
print f
obj = N(expand(f, trig=True).subs(
    {sin(x): s1, cos(x): c1, sin(y): s2, cos(y): c2}))
rels = [s1**2 + c1**2 - 1, s2**2 + c2**2 - 1]
Rlx = SDPRelaxations([s1, c1, s2, c2], rels)
Rlx.SetObjective(obj)
Rlx.AddConstraint(1 - s1**2 >= 0)
Rlx.AddConstraint(1 - s2**2 >= 0)
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print Rlx.Solution
g = lambda x: sum([cos((j + 1) * x[0] + j) for j in range(1, 6)]) * \
    sum([cos((j + 1) * x[1] + j) for j in range(1, 6)])
x0 = (-5, 5)
from scipy.optimize import minimize
cons = (
    {'type': 'ineq', 'fun': lambda x: 100 - x[0]**2},
    {'type': 'ineq', 'fun': lambda x: 100 - x[1]**2})
sol1 = minimize(g, x0, method='COBYLA', constraints=cons)
sol2 = minimize(g, x0, method='SLSQP', constraints=cons)
print "solution according to 'COBYLA':"
print sol1
print "solution according to 'SLSQP':"
print sol2

from sage.all import *
m1 = minimize_constrained(g, cons=[cn['fun'] for cn in cons], x0=x0)
m2 = minimize_constrained(g, cons=[cn['fun']
                                   for cn in cons], x0=x0, algorithm='l-bfgs-b')
print "Sage:"
print "minimize_constrained (default):", m1, g(m1)
print "minimize_constrained (l-bfgs-b):", m2, g(m2)
print "PSO:"
from pyswarm import pso
lb = [-10, -10]
ub = [10, 10]
cns = [cn['fun'] for cn in cons]
print pso(g, lb, ub, ieqcons=cns)

"""
Solution of a Semidefinite Program:
                Solver: CVXOPT
                Status: Optimal
   Initialization Time: 1129.02412415 seconds
              Run Time: 5.258507 seconds
Primal Objective Value: -18.0955649723
  Dual Objective Value: -18.0955648855
Feasible solution for moments of order 6
Scipy 'COBYLA':
     fun: -3.3261182321238367
   maxcv: 0.0
 message: 'Optimization terminated successfully.'
    nfev: 39
  status: 1
 success: True
       x: array([-3.96201407,  4.81176624])
Scipy 'SLSQP':
     fun: -0.856702387212005
     jac: array([-0.00159422,  0.00080796,  0.        ])
 message: 'Optimization terminated successfully.'
    nfev: 35
     nit: 7
    njev: 7
  status: 0
 success: True
       x: array([-4.92714381,  4.81186391])
Sage:
minimize_constrained (default): (-3.962032420336303, 4.811734682897321) -3.32611819422
minimize_constrained (l-bfgs-b): (-3.962032420336303, 4.811734682897321) -3.32611819422
PSO:
Stopping search: Swarm best objective change less than 1e-08
(array([-0.77822054,  4.8118272 ]), -18.095565036766224)

"""
