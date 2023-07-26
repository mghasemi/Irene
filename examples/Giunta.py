"""
Minimize the Giunta function
"""
from scipy.optimize import minimize
from pyswarm import pso
from sympy import *
from Irene import *

x = Symbol('x')
y = Symbol('y')
s1 = Symbol('s1')
c1 = Symbol('c1')
s2 = Symbol('s2')
c2 = Symbol('c2')
f = .6 + (sin(x - 1) + (sin(x - 1)) ** 2 + .02 * sin(4 * (x - 1))) + \
    (sin(y - 1) + (sin(y - 1)) ** 2 + .02 * sin(4 * (y - 1)))
f = expand(f, trig=True)
f = N(f.subs({sin(x): s1, cos(x): c1, sin(y): s2, cos(y): c2}))
rels = [s1 ** 2 + c1 ** 2 - 1, s2 ** 2 + c2 ** 2 - 1]
Rlx = SDPRelaxations([s1, c1, s2, c2], rels)
Rlx.SetObjective(f)
Rlx.AddConstraint(1 - s1 ** 2 >= 0)
Rlx.AddConstraint(1 - s2 ** 2 >= 0)
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print(Rlx.Solution)
# solve with scipy
fun = lambda x: .6 + (sin((16. / 15.) * x[0] - 1) + (sin((16. / 15.) * x[0] - 1)) ** 2 + .02 * sin(
    4 * ((16. / 15.) * x[0] - 1))) + (sin((16. / 15.) * x[1] - 1) + (sin((16. / 15.) * x[1] - 1)) ** 2 + .02 * sin(
    4 * ((16. / 15.) * x[1] - 1)))
cons = [
    {'type': 'ineq', 'fun': lambda x: 1 - x[i] ** 2} for i in range(2)]
x0 = tuple([0 for _ in range(2)])
sol1 = minimize(fun, x0, method='COBYLA', constraints=cons)
sol2 = minimize(fun, x0, method='SLSQP', constraints=cons)
print("solution according to 'COBYLA':")
print(sol1)
print("solution according to 'SLSQP':")
print(sol2)
# solve with pso
print("PSO:")
lb = [-1 for i in range(2)]
ub = [1 for i in range(2)]
cns = [cn['fun'] for cn in cons]
print(pso(fun, lb, ub, ieqcons=cns))
