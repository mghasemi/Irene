import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from gpkit import VectorVariable, Variable, Model
from gpkit.constraints.bounded import Bounded, ConstraintSet

from .grouprings import _degree
from .program import OptimizationProblem

class SONCRelaxations(object):
    r"""
    This class aims to provide a framework for polynomial optimization using the techniques
    introduced by Ghasemi, Lasserre, and Marshall, using Geometric Programming.
    """

    def __init__(self, prog: OptimizationProblem):
        self.prog = prog
        self.program_size = len(prog.constraints) + 1
        self.g = [-prog.objective] + prog.constraints
        self.Ord = self.prog.program_degree()
        self.error_bound = 1e-10
        self.solution = None
        self.f_sonc_g = None

    def form_gp(self):
        mu = VectorVariable(self.program_size, 'mu', '', "Lagrangian coefficients")
        self.prog.newton()
        # Define the objective function
        p = 0.
        alpha0 = self.prog.tuple2mono(self.prog.vertices[0])
        for g in self.g:
            print(self.prog.delta(-g, self.Ord))
        for i in range(1, self.program_size):
            g_i_plus_alpha0 = max(0, self.g[i][alpha0])
            p += mu[i] * g_i_plus_alpha0
        print(p)
