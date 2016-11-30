"""
minimize -(sin x - 1)^3 - (sin x - cosh y)^4 - (cosh y - 3)^2
subject to:
        10 - (sin x - 1)^2 >= 0,
        10 - (sin x - cosh y)^2 >= 0,
        10 - (cosh y - 3)^2 >= 0.
        L(sin x + cos x) >= .9
        L(cos^2 x -.25 cosh x) >= .6
"""

from sympy import *
from Irene import *
# define the symbolic variables and functions
x = Symbol('x')
f = Function('f')(x)
g = Function('g')(x)
h = Function('h')(x)
k = Function('k')(x)

# introduce the relation between symbols
rels = [g**2 + f**2 - 1, k**2 - h**2 - 1]
# initiate the SDPRelaxations object
Rlx = SDPRelaxations([f, g, h, k], rels)
# monomial order
Rlx.SetMonoOrd('lex')
# set the objective
Rlx.SetObjective(-(f - 1)**3 - (f - k)**4 - (k - 3)**2)
# add constraints
Rlx.AddConstraint(10 - (f - 1)**2 >= 0)
Rlx.AddConstraint(10 - (f - k)**2 >= 0)
Rlx.AddConstraint(10 - (k - 3)**2 >= 0)
# moment constraints
Rlx.MomentConstraint(Mom(f + g) >= .9)
Rlx.MomentConstraint(Mom(g**2 - .25 * k) - .6 >= 0)
# set the sdp solver
Rlx.SetSDPSolver('cvxopt')
# initiate the SDP
Rlx.InitSDP()
# solve the SDP
Rlx.Minimize()
print Rlx.Solution
print "Moment of %s = " % str(f), Rlx.Solution.TruncatedMmntSeq[f]
print "Moment of %s = " % str(g**2), Rlx.Solution.TruncatedMmntSeq[g**2]
print "Moment of %s = " % str(g), Rlx.Solution.TruncatedMmntSeq[g]
print "Moment of %s = " % str(k), Rlx.Solution.TruncatedMmntSeq[k]
print "________________________________"
print "Moment of %s = " % str(f + g), Rlx.Solution.TruncatedMmntSeq[f] + \
    Rlx.Solution.TruncatedMmntSeq[g]
print "Moment of %s = " % str(g**2 + .25 * k - .6), Rlx.Solution.TruncatedMmntSeq[g**2] + \
    .25 * Rlx.Solution.TruncatedMmntSeq[k] - .6
