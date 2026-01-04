import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import *
from Irene.program import *
from Irene.geometric import *
import numpy as np

if __name__ == '__main__':
    
    S = CommutativeSemigroup(['x', 'y', 'z'])
    SA = SemigroupAlgebra(S)
    
    x = SA['x']
    y = SA['y']
    z = SA['z']
    optim = OptimizationProblem(SA)
    #optim.set_objective(x**6+y**6+z**6-5*x-4*y-z+8)
    #g = x**6+y**6+z**6+x**2 * y* z**2 - x**4 - y**4 - z**4 -y*z**3 - x*y**2 +2 + x**2
    #optim.set_objective(g)
    #optim.set_objective(5*x+6*y+x**3-y**2)
    #optim.add_constraints([-x*y-4*x**4-y**4+8])
    #####################################
    #f = x + z**3 +y**6 + z**6 + x**6
    #g1 = 1 - x**6 + y**6
    #####################################
    f = -y - 2*x**2
    g1 = y - x**4 * y + y**5 - x**6 - y**6
    g2 = y - 5*x**2 + x**4 * y - x**6 - y**6
    optim.set_objective(f)
    optim.add_constraints([g1, g2])
    
    gp = GPRelaxations(optim)
    
    gp.H = gp.auto_transform_matrix()
    
    gp.H = np.array([[1, 0], [-1, 1]])
    
    print(gp.H)
    
    gp.solve()
    
    print(gp.solution)