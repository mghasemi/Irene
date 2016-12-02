"""
minimize x sinh y + (e^(y sin x))
subject to 
		-pi <= x <= pi
		-pi <= y <= pi
using Legendre approximation of the transcendental parts `e^(y sin x)` and sinh(y)
"""
from sympy import *
from Irene import *
from pyProximation import OrthSystem
# introduce symbols and functions
x = Symbol('x')
y = Symbol('y')
sh = Function('sh')(y)
ch = Function('ch')(y)
# transcendental term of objective
f = exp(y * sin(x))
g = sinh(y)
# Legendre polynomials via pyProximation
D_f = [(-pi, pi), (-pi, pi)]
D_g = [(-pi, pi)]
Orth_f = OrthSystem([x, y], D_f)
Orth_g = OrthSystem([y], D_g)
# set bases
B_f = Orth_f.PolyBasis(10)
B_g = Orth_g.PolyBasis(10)
# link B to S
Orth_f.Basis(B_f)
Orth_g.Basis(B_g)
# generate the orthonormal bases
Orth_f.FormBasis()
Orth_g.FormBasis()
# extract the coefficients of approximations
Coeffs_f = Orth_f.Series(f)
Coeffs_g = Orth_g.Series(g)
# form the approximations
f_app = sum([Orth_f.OrthBase[i] * Coeffs_f[i]
             for i in range(len(Orth_f.OrthBase))])
g_app = sum([Orth_g.OrthBase[i] * Coeffs_g[i]
             for i in range(len(Orth_g.OrthBase))])
# initiate the Relaxation object
Rlx = SDPRelaxations([x, y])
# set the objective
Rlx.SetObjective(x * g_app + f_app)
# add support constraints
Rlx.AddConstraint(pi**2 - x**2 >= 0)
Rlx.AddConstraint(pi**2 - y**2 >= 0)
# set the sdp solver
Rlx.SetSDPSolver('cvxopt')
# initialize the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print Rlx.Solution
# using scipy
from scipy.optimize import minimize
fun = lambda x: x[0] * sinh(x[1]) + exp(x[1] * sin(x[0]))
cons = (
    {'type': 'ineq', 'fun': lambda x: pi**2 - x[0]**2},
    {'type': 'ineq', 'fun': lambda x: pi**2 - x[1]**2}
)
sol1 = minimize(fun, (0, 0), method='COBYLA', constraints=cons)
sol2 = minimize(fun, (0, 0), method='SLSQP', constraints=cons)
print "solution according to 'COBYLA':"
print sol1
print "solution according to 'SLSQP':"
print sol2
# particle swarm optimization
from pyswarm import pso
lb = [-3.3, -3.3]
ub = [3.3, 3.3]
cns = [cons[0]['fun'], cons[1]['fun']]
print "PSO:"
print pso(fun, lb, ub, ieqcons=cns)
