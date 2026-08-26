"""Diagnostic: SDP+ADE gap analysis for sqrt/exp differential lift."""

from math import exp, sqrt
import numpy as np
from scipy.optimize import minimize

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxation_api import RelaxationEngine
from Irene.relaxations import RelaxationConfig, SDPRelaxations

# ====================================================================
# 1. SciPy reference (same as original)
# ====================================================================
def solve_with_scipy():
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
        (xv, yv)
        for xv in np.linspace(-3.0, 3.0, 7)
        for yv in np.linspace(-sqrt(2.0), sqrt(2.0), 5)
        if xv * yv + 1.0 >= 0.0
    ]
    results = []
    for start in starts:
        result = minimize(objective, np.asarray(start, dtype=float), method="SLSQP",
                         bounds=[(-10.0, 10.0), (-sqrt(2.0), sqrt(2.0))],
                         constraints=constraints, options={"ftol": 1e-10, "maxiter": 1000})
        if result.success and np.min(constraint_values(result.x)) >= -1e-7:
            results.append(result)
    return min(results, key=lambda item: item.fun) if results else None

scipy_result = solve_with_scipy()
print("=" * 70)
print("SCI-PY REFERENCE")
print("=" * 70)
if scipy_result:
    print(f"  Optimal value: {scipy_result.fun:.8f}")
    print(f"  Point: x={scipy_result.x[0]:.6f}, y={scipy_result.x[1]:.6f}")

# ====================================================================
# 2. Build the problem (same as original)
# ====================================================================
generator_names = ["x", "y", "u", "v", "d_vx", "d_vy", "d_vu"]
sg = CommutativeSemigroup(generator_names)
sga = SemigroupAlgebra(sg)

generators = {g.ext_rep[0].name: g for g in sg.generators}
x, y, u, v, d_vx, d_vy, d_vu = (sga[name] for name in generator_names)

# Derivatives
sga.add_derivative({generators["x"]: 1, generators["y"]: 0, generators["u"]: 0, generators["v"]: y * v})
sga.add_derivative({generators["x"]: 0, generators["y"]: 1, generators["u"]: 0, generators["v"]: x * v})
sga.add_derivative({generators["x"]: d_vx, generators["y"]: d_vy, generators["u"]: d_vu, generators["v"]: 1})

relations = [
    u**2 - (x * y + 1),
    v * y**2 * d_vx - (y - x * y * v * d_vy),
    v * x**2 * d_vy - (x - x * y * v * d_vx),
    2 * u * d_vu - (y * d_vx + x * d_vy),
]

# ====================================================================
# 3. Inspect Groebner quotient basis at order=2
# ====================================================================
print("\n" + "=" * 70)
print("GROEBNER QUOTIENT DIAGNOSTIC")
print("=" * 70)

prog = OptimizationProblem(sga, relations=relations)
prog.set_objective(1 - x**2 + y)
prog.add_constraints([
    u + x / 2 - x**2 + 2 * y,
    2 - y**2,
    y - x * v + 1,
])

config = RelaxationConfig(
    reduction_method="border_basis",
    quotient_basis="groebner",
    monomial_pruning=True,
    sparsity_detection=True,
    verbose_reduction=False,
)

sdp_relax = SDPRelaxations.from_problem(prog, config=config)
sdp_relax.MomentsOrd(2)

# Inspect the quotient basis
print(f"\n  Number of generators: {sg.num_gens}")
print(f"  Generators: {[g.ext_rep[0].name for g in sg.generators]}")
print(f"  Relations count: {len(relations)}")

# Get monomial basis info
if hasattr(sdp_relax, 'Monomials'):
    mons = sdp_relax.Monomials
    print(f"\n  Monomial basis size (order=2): {len(mons) if hasattr(mons, '__len__') else 'N/A'}")

# Check the Groebner basis
if hasattr(sdp_relax, '_groebner_basis'):
    gb = sdp_relax._groebner_basis
    print(f"\n  Groebner basis size: {len(gb) if gb else 'None'}")
    for i, g in enumerate(gb or []):
        print(f"    GB[{i}]: degree={sg.degree(g)}")

# ====================================================================
# 4. Run SDP without sign constraints (baseline)
# ====================================================================
print("\n" + "=" * 70)
print("SDP VARIANT A: No u>=0, v>=0 (current)")
print("=" * 70)

sdp_relax.SetSDPSolver("cvxopt")
sdp_relax.InitSDP()
sdp_relax.Minimize()
sol = sdp_relax.Solution
if sol and hasattr(sol, 'Primal') and sol.Primal is not None:
    val_a = float(sol.Primal)
    print(f"  SDP lower bound: {val_a:.8f}")
else:
    val_a = None
    print("  No solution")

# ====================================================================
# 5. Run SDP WITH u>=0 and v>=0
# ====================================================================
print("\n" + "=" * 70)
print("SDP VARIANT B: With u>=0, v>=0")
print("=" * 70)

sg_b = CommutativeSemigroup(generator_names)
sga_b = SemigroupAlgebra(sg_b)
gens_b = {g.ext_rep[0].name: g for g in sg_b.generators}
x_b, y_b, u_b, v_b, d_vx_b, d_vy_b, d_vu_b = (sga_b[name] for name in generator_names)

sga_b.add_derivative({gens_b["x"]: 1, gens_b["y"]: 0, gens_b["u"]: 0, gens_b["v"]: y_b * v_b})
sga_b.add_derivative({gens_b["x"]: 0, gens_b["y"]: 1, gens_b["u"]: 0, gens_b["v"]: x_b * v_b})
sga_b.add_derivative({gens_b["x"]: d_vx_b, gens_b["y"]: d_vy_b, gens_b["u"]: d_vu_b, gens_b["v"]: 1})

prog_b = OptimizationProblem(sga_b, relations=relations)
prog_b.set_objective(1 - x_b**2 + y_b)
prog_b.add_constraints([
    u_b + x_b / 2 - x_b**2 + 2 * y_b,
    2 - y_b**2,
    y_b - x_b * v_b + 1,
    u_b + 0,   # u >= 0
    v_b + 0,   # v >= 0
])

engine_b = RelaxationEngine(prog_b, order=2, solver="cvxopt", verbosity=0, config=config)
result_b = engine_b.solve("sos")
print(f"  SDP lower bound: {result_b.value:.8f}")
print(f"  Status: {result_b.status}")

# ====================================================================
# 6. Run at order=1 for comparison
# ====================================================================
print("\n" + "=" * 70)
print("SDP VARIANT C: Order=1, no sign constraints")
print("=" * 70)

engine_c = RelaxationEngine(prog, order=1, solver="cvxopt", verbosity=0, config=config)
result_c = engine_c.solve("sos")
print(f"  SDP lower bound: {result_c.value:.8f}")
print(f"  Status: {result_c.status}")

# ====================================================================
# 7. Summary table
# ====================================================================
print("\n" + "=" * 70)
print("GAP ANALYSIS SUMMARY")
print("=" * 70)
if scipy_result and val_a is not None:
    print(f"\n  SciPy feasible upper bound:   {scipy_result.fun:.8f}")
    print(f"  SDP (order=2, no signs):     {val_a:.8f}  gap={scipy_result.fun - val_a:.4f}")
    print(f"  SDP (order=2, with signs):   {result_b.value:.8f}  gap={scipy_result.fun - result_b.value:.4f}")
    print(f"  SDP (order=1, no signs):     {result_c.value:.8f}  gap={scipy_result.fun - result_c.value:.4f}")
