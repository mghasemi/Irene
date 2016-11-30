"""
minimize x + e^(x sin x)
subject to 
		-pi <= x <= pi
using Legendre approximation of the transcendental part `e^(x sin x)`
"""
from sympy import *
from Irene import *
from pyProximation import OrthSystem
# introduce symbols and functions
x = Symbol('x')
e = Function('e')(x)
# transcendental term of objective
f = exp(x * sin(x))
# Legendre polynomials via pyProximation
D = [(-pi, pi)]
S = OrthSystem([x], D)
# set B = {1, x, x^2, ..., x^12}
B = S.PolyBasis(12)
# link B to S
S.Basis(B)
# generate the orthonormal basis
S.FormBasis()
# extract the coefficients of approximation
Coeffs = S.Series(f)
# form the approximation
f_app = sum([S.OrthBase[i] * Coeffs[i] for i in range(len(S.OrthBase))])
# initiate the Relaxation object
Rlx = SDPRelaxations([x])
# set the objective
Rlx.SetObjective(x + f_app)
# add support constraints
Rlx.AddConstraint(pi**2 - x**2 >= 0)
# set the sdp solver
Rlx.SetSDPSolver('dsdp')
# initialize the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print Rlx.Solution
# using scipy
from scipy.optimize import minimize
fun = lambda x: x[0] + exp(x[0] * sin(x[0]))
cons = (
    {'type': 'ineq', 'fun': lambda x: pi**2 - x[0]**2},
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
cns = [cons[0]['fun']]
print "PSO:"
print pso(fun, lb, ub, ieqcons=cns)