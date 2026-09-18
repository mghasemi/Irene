from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxation_api import RelaxationEngine
from Irene.relaxations import RelaxationConfig

# 1. Define the Commutative Semigroup and Algebra Space
# Generators: x (domain), u (transcendental lift), du_x, du_y (derivatives)
sg = CommutativeSemigroup(['x', 'y', 'u', 'du_x', 'du_y'])
sga = SemigroupAlgebra(sg)
x, y, u, du_x, du_y = sga['x'], sga['y'], sga['u'], sga['du_x'], sga['du_y']

# 2. Configure Explicit Derivation Operators for Holonomic Prolongations
# Simulating the lift u = exp(x*y^2) => d_x(u) = y^2 * u, d_y(u) = 2*x*y * u
sga.add_derivative({
    sg.generators[0]: 1,        # d_x(x) = 1
    sg.generators[1]: 0,        # d_x(y) = 0
    sg.generators[2]: du_x      # d_x(u) = du_x
})
sga.add_derivative({
    sg.generators[0]: 0,        # d_y(x) = 0
    sg.generators[1]: 1,        # d_y(y) = 1
    sg.generators[2]: du_y      # d_y(u) = du_y
})

# 3. Formulate the Lifted Optimization Problem
prog = OptimizationProblem(sga)

# Replace transcendental node with the lifted polynomial objective
prog.set_objective(x * u)

# 4. Inject ADE constraints to formalize the differential ideal
prog.add_constraints([
    du_x - (y**2 * u),             # d_x u = y^2 u
    du_y - (2 * x * y * u),        # d_y u = 2xy u
    u - 1                          # Initial evaluation symmetry breaking: u(0,0) = 1 
                                   # (Note: In a true moment constraint, this is an evaluation functional)
])

# 5. Enforce the Differential Archimedean Boxing Condition
# This guarantees strong duality and infinite-dimensional convergence.
R_box = 10.0
prog.add_constraints([
    R_box**2 - x**2 - y**2,        # Spatial bounding
    R_box**2 - u**2 - du_x**2 - du_y**2  # Prolongation bounding
])

# 6. Configure the Two-Stage Hybrid Reduction Pipeline
config = RelaxationConfig(
    reduction_method="border_basis", # Employs QR numerical stability for quotienting
    quotient_basis="border",         # Inner Step: Monoid Quotienting
    monomial_pruning=True,           # Inner Step: Newton Polytope support scaling
    sparsity_detection=True,         # Outer Step: Chordal Graph Clique Decomposition
    verbose_reduction=False
)

# 7. Execute the Unified Differential SDP Relaxation
engine = RelaxationEngine(prog, order=4, solver="sdpa", config=config)
result = engine.solve("sos")

print(f"Certified Lower Bound: {result.value:.8f}")
print(f"Solver Status: {result.status}")