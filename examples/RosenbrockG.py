"""
Minimize the Rosenbrock function
"""

from sympy import *
from Irene import SDPRelaxations
NumVars = 9
# define the symbolic variables and functions
x = [Symbol('x%d' % i) for i in range(NumVars)]

print "Relaxation method:"
# initiate the SDPRelaxations object
Rlx = SDPRelaxations(x)
# monomial order
Rlx.SetMonoOrd('lex')
rosen = sum([100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])
             ** 2 for i in range(NumVars - 1)])
# set the objective
Rlx.SetObjective(rosen)
# add constraints
for i in range(NumVars):
    Rlx.AddConstraint(9 - x[i]**2 >= 0)
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
fun = lambda x: sum([100 * (x[i + 1] - x[i]**2)**2 +
                     (1 - x[i])**2 for i in range(NumVars - 1)])
cons = [
    {'type': 'ineq', 'fun': lambda x: 9 - x[i]**2} for i in range(NumVars)]
x0 = tuple([0 for _ in range(NumVars)])
sol1 = minimize(fun, x0, method='COBYLA', constraints=cons)
sol2 = minimize(fun, x0, method='SLSQP', constraints=cons)
print sol1
print sol2
# solve with pso
print "PSO:"
from pyswarm import pso
lb = [-3 for i in range(NumVars)]
ub = [3 for i in range(NumVars)]
cns = [cn['fun'] for cn in cons]
print pso(fun, lb, ub, ieqcons=cns)

"""
Relaxation method:
Solution of a Semidefinite Program:
                Solver: CVXOPT
                Status: Optimal
   Initialization Time: 750.234924078 seconds
              Run Time: 8.43369 seconds
Primal Objective Value: 1.67774267808e-08
  Dual Objective Value: 1.10015692778e-08
Feasible solution for moments of order 2

scipy.optimize:
     fun: 4.4963584556077389
   maxcv: 0.0
 message: 'Maximum number of function evaluations has been exceeded.'
    nfev: 1000
  status: 2
 success: False
       x: array([  8.64355944e-01,   7.47420978e-01,   5.59389194e-01,
         3.16212252e-01,   1.05034350e-01,   2.05923923e-02,
         9.44389237e-03,   1.12341021e-02,  -7.74530516e-05])
     fun: 1.3578865444308464e-07
     jac: array([ 0.00188377,  0.00581741, -0.00182463,  0.00776938, -0.00343305,
       -0.00186283,  0.0020364 ,  0.00881489, -0.0047164 ,  0.        ])
 message: 'Optimization terminated successfully.'
    nfev: 625
     nit: 54
    njev: 54
  status: 0
 success: True
       x: array([ 1.00000841,  1.00001216,  1.00000753,  1.00001129,  1.00000134,
        1.00000067,  1.00000502,  1.00000682,  0.99999006])
PSO:
Stopping search: maximum iterations reached --> 100
(array([-0.30495496,  0.10698904, -0.129344  ,  0.07972014,  0.027356  ,
        0.02170117, -0.0036854 ,  0.10778454,  0.04141022]), 12.067752160169965)
"""