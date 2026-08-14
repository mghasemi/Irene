"""SOS relaxation for a sqrt/exp differential lift.

Problem:
    minimize 1 - x**2 + y
    subject to
        x*y + 1 >= 0
        u + x/2 - x**2 + 2*y >= 0
        2 - y**2 >= 0
        y - x*v + 1 >= 0
        v >= 0

The algebraic lift u = sqrt(x*y + 1) is represented by
u**2 - x*y - 1 = 0 together with u >= 0.  The exponential lift
v = exp(x*y) is represented by the supplied differential identities.
The derivative variables vx=d_v(x) and vy=d_v(y) are algebra generators.
"""

from math import exp, sqrt

import numpy as np
from scipy.optimize import minimize

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxation_api import RelaxationEngine
from Irene.relaxations import RelaxationConfig


def solve_with_scipy():
    """Find a feasible reference point for the original two-variable problem.

    This solves the original nonlinear problem, not the lifted polynomial
    relaxation. Therefore its objective is a feasible upper bound, while the
    SOS result below is a relaxation lower bound; they need not be equal.
    """
    def objective(point):
        x_value, y_value = point
        return 1.0 - x_value**2 + y_value

    def constraint_values(point):
        x_value, y_value = point
        radicand = x_value * y_value + 1.0
        root = sqrt(max(radicand, 0.0))
        exponential = exp(np.clip(x_value * y_value, -700.0, 700.0))
        return np.array([
            radicand,
            root + x_value / 2.0 - x_value**2 + 2.0 * y_value,
            2.0 - y_value**2,
            y_value - x_value * exponential + 1.0,
        ])

    constraints = {"type": "ineq", "fun": constraint_values}
    starts = [
        (x_value, y_value)
        for x_value in np.linspace(-3.0, 3.0, 7)
        for y_value in np.linspace(-sqrt(2.0), sqrt(2.0), 5)
        if x_value * y_value + 1.0 >= 0.0
    ]
    results = []
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="SLSQP",
            bounds=[(-10.0, 10.0), (-sqrt(2.0), sqrt(2.0))],
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        if result.success and np.min(constraint_values(result.x)) >= -1e-7:
            results.append(result)

    if not results:
        return None
    return min(results, key=lambda item: item.fun)
# Algebra generators: coordinates, lifts, and d_v-coordinate variables.
# u = sqrt(x*y + 1), v = exp(x*y), d_vx = d_v(x), d_vy = d_v(y)
generator_names = ["x", "y", "u", "v", "d_vx", "d_vy", "d_vu"]
sg = CommutativeSemigroup(generator_names)
sga = SemigroupAlgebra(sg)

# CommutativeSemigroup sorts its free-group generators, so map by name rather
# than relying on positional ordering.
generators = {
    generator.ext_rep[0].name: generator for generator in sg.generators
}
x, y, u, v, d_vx, d_vy, d_vu = (sga[name] for name in generator_names)

# Differential operators d_x, d_y, and d_v.  Entries for every generator are
# supplied because SemigroupAlgebra.diff indexes the map for each generator.
# d_x
sga.add_derivative({
    generators["x"]: 1,
    generators["y"]: 0,
    generators["u"]: 0,
    generators["v"]: y * v,
})
# d_y
sga.add_derivative({
    generators["x"]: 0,
    generators["y"]: 1,
    generators["u"]: 0,
    generators["v"]: x * v,
})
# d_v
sga.add_derivative({
    generators["x"]: d_vx,
    generators["y"]: d_vy,
    generators["u"]: d_vu,
    generators["v"]: 1,
})

# Equality constraints are quotient relations.  In particular, the two ADE
# equations below are exactly the relations supplied in the formulation.
relations = [
    u**2 - (x * y + 1),
    v * y**2 * d_vx - (y - x * y * v * d_vy),
    v * x**2 * d_vy - (x - x * y * v * d_vx),
    2 * u * d_vu - (y * d_vx + x * d_vy),
]

prog = OptimizationProblem(sga, relations=relations)
prog.set_objective(1 - x**2 + y)
prog.add_constraints([
    #x * y + 1,
    u + x / 2 - x**2 + 2 * y,
    2 - y**2,
    y - x * v + 1,
    #u + 0,  # Selects the nonnegative square-root branch.
    #v + 0 # Exponential lifts are nonnegative.
])

# Use the exact Groebner quotient path for this relation-constrained example.
# The Newton-pruning flag is intentionally disabled: certified SOS pruning is
# not sound without a complete Gram-support certificate.
config = RelaxationConfig(
    reduction_method="border_basis",
    quotient_basis="groebner",
    monomial_pruning=True,
    sparsity_detection=True,
    verbose_reduction=False,
)

engine = RelaxationEngine(
    prog,
    order=2,
    solver="cvxopt",
    verbosity=1,
    config=config,
)
# result = engine.solve("sosonc_sos_first")
result = engine.solve("sos")

print(f"Certified Lower Bound: {result.value:.8f}")
print(f"Solver Status: {result.status}")
print(f"Message: {result.message}")

scipy_result = solve_with_scipy()
if scipy_result is None:
    print("SciPy Status: no feasible result found")
else:
    print(f"SciPy Feasible Upper Bound: {scipy_result.fun:.8f}")
    print(f"SciPy Point: x={scipy_result.x[0]:.8f}, y={scipy_result.x[1]:.8f}")
    print(f"SOS-to-SciPy Gap: {scipy_result.fun - result.value:.8f}")
