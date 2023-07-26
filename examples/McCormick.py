from sympy import *
from Irene import *
from pyProximation import OrthSystem, Measure
# introduce symbols and functions
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
t = Symbol('t')
# transcendental term of objective
f = sin(z)
# Legendre polynomials via pyProximation
D_f = [(-2, 2)]
w = lambda x: 1. / sqrt(4 - x**2)
M = Measure(D_f, w)
# link the measure to S
Orth_f = OrthSystem([z], D_f)
Orth_f.SetMeasure(M)
# set bases
B_f = Orth_f.PolyBasis(10)
# link B to S
Orth_f.Basis(B_f)
# generate the orthonormal bases
Orth_f.FormBasis()
# extract the coefficients of approximations
Coeffs_f = Orth_f.Series(f)
# form the approximations
f_app = sum([Orth_f.OrthBase[i] * Coeffs_f[i]
             for i in range(len(Orth_f.OrthBase))])
# objective function
obj = f_app.subs({z: x + y}) + (x - y)**2 - 1.5 * x + 2.5 * y + 1
print(obj)
# definition of 't'
rels = []  # [t**2 * (x**2 + y**2) - 1]
# initiate the Relaxation object
Rlx = SDPRelaxations([x, y], rels)
Rlx.Parallel = False
# set the objective
Rlx.SetObjective(obj)
# add support constraints
Rlx.AddConstraint(4 - (x**2 + y**2) >= 0)
# set the sdp solver
Rlx.SetSDPSolver('cvxopt')
# initialize the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
Rlx.Solution.ExtractSolution('lh', 1)
print(Rlx.Solution)
