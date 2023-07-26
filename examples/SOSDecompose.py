# SOS decomposition
from sympy import *
from Irene import SDPRelaxations

# define the symbolic variables and functions
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

Rlx = SDPRelaxations([x, y, z])
Rlx.SetObjective(x ** 3 + x ** 2 * y ** 2 + z ** 2 * x * y - x * z)
Rlx.AddConstraint(9 - (x ** 2 + y ** 2 + z ** 2) >= 0)
# initiate the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print(Rlx.Solution)
# extract decomposition
V = Rlx.Decompose()
# test the decomposition
sos = 0
for v in V:
    # for g0 = 1
    if v == 0:
        sos = expand(Rlx.ReduceExp(sum([p ** 2 for p in V[v]])))
    # for g1, the constraint
    else:
        sos = expand(Rlx.ReduceExp(
            sos + Rlx.Constraints[v - 1] * sum([p ** 2 for p in V[v]])))
sos = sos.subs(Rlx.RevSymDict)
pln = Poly(sos).as_dict()
pln = {ex: round(pln[ex], 5) for ex in pln}
print(Poly(pln, (x, y, z)).as_expr())
