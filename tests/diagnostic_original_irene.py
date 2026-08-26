"""Run sqrt/exp differential lift through ORIGINAL Irene (legacy SymPy API).

The old SDPRelaxations(prog) ignores prog.relations, so we must use the legacy
constructor: SDPRelaxations(gens=[sympy_symbols], relations=[sympy_exprs]).
"""

import sys, os
from math import exp, sqrt

# Ensure original Irene is on path BEFORE IreneRewrite
orig_irene_path = os.path.join(os.path.dirname(__file__), '..', 'Irene')
if orig_irene_path not in sys.path:
    sys.path.insert(0, orig_irene_path)

import numpy as np
from scipy.optimize import minimize
from sympy import symbols, groebner

# Import from ORIGINAL Irene (the one at ../Irene/Irene/)
sys.path.insert(0, os.path.join(orig_irene_path, 'Irene'))
from relaxations import SDPRelaxations

# ====================================================================
# 1. SciPy reference
# ====================================================================
def solve_with_scipy():
    def objective(point):
        xv, yv = point
        return 1.0 - xv**2 + yv

    def constraint_values(point):
        xv, yv = point
        radicand = xv * yv + 1.0
        root = sqrt(max(radicand, 0.0))
        exponential = exp(np.clip(xv * yv, -700.0, 700.0))
        return np.array([radicand, root + xv/2.0 - xv**2 + 2.0*yv,
                         2.0 - yv**2, yv - xv*exponential + 1.0])

    constraints = {"type": "ineq", "fun": constraint_values}
    starts = [(xv, yv) for xv in np.linspace(-3.0, 3.0, 7)
              for yv in np.linspace(-sqrt(2.0), sqrt(2.0), 5)
              if xv * yv + 1.0 >= 0.0]
    results = []
    for start in starts:
        result = minimize(objective, np.asarray(start, dtype=float), method="SLSQP",
                         bounds=[(-10., 10.), (-sqrt(2.), sqrt(2.))],
                         constraints=constraints, options={"ftol": 1e-10, "maxiter": 1000})
        if result.success and np.min(constraint_values(result.x)) >= -1e-7:
            results.append(result)
    return min(results, key=lambda r: r.fun) if results else None

scipy_result = solve_with_scipy()
print("=" * 70)
print("ORIGINAL IRENE (legacy SymPy API) — sqrt/exp differential lift")
print("=" * 70)
if scipy_result:
    print(f"SciPy feasible upper bound: {scipy_result.fun:.8f}")
    print(f"SciPy point: x={scipy_result.x[0]:.6f}, y={scipy_result.x[1]:.6f}")

# ====================================================================
# 2. Full ADE problem with original Irene (legacy API)
# ====================================================================
x, y, u, v, d_vx, d_vy, d_vu = symbols('x y u v d_vx d_vy d_vu')

relations_sympy = [
    u**2 - (x*y + 1),
    v * y**2 * d_vx - (y - x*y*v*d_vy),
    v * x**2 * d_vy - (x - x*y*v*d_vx),
    2*u*d_vu - (y*d_vx + x*d_vy),
]

print("\n" + "=" * 70)
print("VARIANT A: Full ADE, order=2")
print("=" * 70)

sdp_a = SDPRelaxations([x, y, u, v, d_vx, d_vy, d_vu], relations_sympy)
sdp_a.MomentsOrd(2)

gb_a = sdp_a.Groebner
print(f"Groebner basis size: {len(gb_a)}")
for i, g in enumerate(gb_a):
    print(f"  GB[{i}]: degree={g.total_degree()}, terms={len(g.as_expr().args) if hasattr(g,'as_expr') and g.as_expr().is_Add else 1}")

sdp_a.SetObjective(1 - x**2 + y)
sdp_a.AddConstraint(u + x/2 - x**2 + 2*y >= 0)
sdp_a.AddConstraint(2 - y**2 >= 0)
sdp_a.AddConstraint(y - x*v + 1 >= 0)

sdp_a.SetSDPSolver("cvxopt")
sdp_a.InitSDP()
sdp_a.Minimize()

val_a = sdp_a.Solution.Primal if sdp_a.Solution and sdp_a.Solution.Primal is not None else None
print(f"  SDP lower bound: {val_a:.8f}" if val_a is not None else "  No solution")

# ====================================================================
# 3. Only algebraic lift, no ADE (original Irene)
# ====================================================================
x2, y2, u2 = symbols('x y u')

relations_d = [u2**2 - (x2*y2 + 1)]

print("\n" + "=" * 70)
print("VARIANT D: Only u^2=xy+1, no ADE, order=2")
print("=" * 70)

sdp_d = SDPRelaxations([x2, y2, u2], relations_d)
sdp_d.MomentsOrd(2)
print(f"Groebner basis size: {len(sdp_d.Groebner)}")

sdp_d.SetObjective(1 - x2**2 + y2)
sdp_d.AddConstraint(u2 + x2/2 - x2**2 + 2*y2 >= 0)
sdp_d.AddConstraint(2 - y2**2 >= 0)

sdp_d.SetSDPSolver("cvxopt")
sdp_d.InitSDP()
sdp_d.Minimize()

val_d = sdp_d.Solution.Primal if sdp_d.Solution and sdp_d.Solution.Primal is not None else None
print(f"  SDP lower bound: {val_d:.8f}" if val_d is not None else "  No solution")

# ====================================================================
# 4. Full ADE, order=3 (original Irene) — may be slow with 7 generators
# ====================================================================
print("\n" + "=" * 70)
print("VARIANT H: Full ADE, order=3")
print("=" * 70)

try:
    sdp_h = SDPRelaxations([x, y, u, v, d_vx, d_vy, d_vu], relations_sympy)
    sdp_h.MomentsOrd(3)
    print(f"Groebner basis size: {len(sdp_h.Groebner)}")

    sdp_h.SetObjective(1 - x**2 + y)
    sdp_h.AddConstraint(u + x/2 - x**2 + 2*y >= 0)
    sdp_h.AddConstraint(2 - y**2 >= 0)
    sdp_h.AddConstraint(y - x*v + 1 >= 0)

    sdp_h.SetSDPSolver("cvxopt")
    sdp_h.InitSDP()
    sdp_h.Minimize()

    val_h = sdp_h.Solution.Primal if sdp_h.Solution and sdp_h.Solution.Primal is not None else None
    print(f"  SDP lower bound: {val_h:.8f}" if val_h is not None else "  No solution")
except Exception as e:
    val_h = None
    print(f"  Error: {e}")

# ====================================================================
# Summary
# ====================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY (Original Irene, legacy API)")
print("=" * 70)
if scipy_result:
    print(f"\n  SciPy feasible upper bound:   {scipy_result.fun:.8f}")
    if val_a is not None:
        print(f"  SDP (o=2, full ADE):       {val_a:.8f}  gap={scipy_result.fun - val_a:+.4f}")
    if val_d is not None:
        print(f"  SDP (o=2, no ADE):         {val_d:.8f}  gap={scipy_result.fun - val_d:+.4f}")
    if val_h is not None:
        print(f"  SDP (o=3, full ADE):       {val_h:.8f}  gap={scipy_result.fun - val_h:+.4f}")
