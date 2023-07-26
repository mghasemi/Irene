from sympy import *
from Irene import *

# introduce symbols and functions
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
s = Symbol('s')
# objective function
obj = -((pi ** 2 + 12 ** 2 * (x ** 2 + y ** 2)) + (pi ** 2 - 4 * 12 ** 2 * (x ** 2 + y ** 2))
        ) / ((pi ** 2 + 12 ** 2 * (x ** 2 + y ** 2)) * (2 + .5 * (x ** 2 + y ** 2)))
# numerator
top = numer(obj)
# denominator
bot = expand(denom(obj))
# initiate the Relaxation object
Rlx = SDPRelaxations([x, y])
# settings
Rlx.Probability = False
Rlx.Parallel = False
# set the objective
Rlx.SetObjective(top)
# Rlx.AddConstraint(4 - (x**2 + y**2) >= 0)
# moment constraint
Rlx.MomentConstraint(Mom(bot) == 1)
# set the sdp solver
Rlx.SetSDPSolver('cvxopt')
# initialize the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
Rlx.Solution.ExtractSolution('lh', 1)
print(Rlx.Solution)
