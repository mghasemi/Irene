#!/usr/bin/env python3
"""
Differential-Algebraic SDP Demo — Tractable Edition
===================================================

Demonstrates the full DSDP pipeline with a numerically well-conditioned problem:
  minimize x * u  subject to  u = exp(x)  (1D ADE lift)
  on [-R, R] with R=3.

This is small enough that order-2 converges cleanly while still exercising:
  - build_ade_relations() for derivative symbol adjunction
  - AddConstraint() for holonomic prolongation relations
  - Archimedean boxing + mean polynomial certificates
  - Border-basis quotienting via RelaxationConfig
"""

import os
os.environ["IRENE_QUOTIENT_BASIS"] = "border"

from sympy import symbols, expand
from Irene.dsdp import DSDPRelaxations
from Irene.relaxation_api import RelaxationEngine
from Irene.program import OptimizationProblem
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.relaxations import RelaxationConfig

# ==================================================================
# PART A: DSDP relaxation with 1D ADE lift (u = exp(x))
# ==================================================================
print("=" * 60)
print("PART A: DSDP Relaxation — 1D ADE Lift")
print("=" * 60)

x = symbols('x')
u = symbols('u')   # transcendental lift: u ≈ exp(x)

dsdp = DSDPRelaxations(
    [x, u],
    name="DSDP_1D_ADE",
    archimedean=True,
    box_size=3.0,       # smaller box → better conditioning
    verbosity=1,
)

# d_x(x) = 1, d_x(u) = du (where du ≈ exp(x) = u)
diff_map = {x: 1, u: symbols('du')}
dx_syms, dx_rels, _ = dsdp.build_ade_relations(diff_map, prefix="d", wrt="x")

print(f"  Derivative symbol for u: {dx_syms[u]}")
print(f"  Generators after adjunction: {dsdp.NumGenerators}")

# Inject ADE constraint: du - u = 0  (i.e. d_x(u) = u, which is exp'(x) = exp(x))
du = dx_syms[u]
ade_constraint = expand(du - u)
print(f"  ADE constraint: {ade_constraint}")

dsdp.AddConstraint(ade_constraint)
dsdp.SetObjective(x * u)

# Configure border-basis reduction
config = RelaxationConfig(
    reduction_method="border_basis",
    monomial_pruning=True,
    border_basis_degree=2,
    sparsity_detection=False,   # too small for sparsity to help in 1D
    verbose_reduction=True,
    quotient_basis="border",
)

dsdp.config = config
dsdp.NewtonPruning = True

result_a = dsdp.solve(order=2)

print(f"\n{'─'*60}")
if result_a is not None:
    print(f"  Certified Lower Bound: {result_a:.8f}")
else:
    sol = dsdp.Solution
    if sol is not None:
        msg = str(getattr(sol, "Message", "")) + " " + str(getattr(sol, "Status", ""))
        primal = getattr(sol, "Primal", None)
        print(f"  Solver Message: {msg}")
        if primal is not None:
            print(f"  Primal Value: {primal:.8f}")
    else:
        print("  No solution returned")
print(f"  Generators:   {dsdp.NumGenerators}")
print(f"  ADE Relations:{len(dsdp.FreeRelations)}")
print(f"  Diff KKT:     {dsdp.diff_constraints_count}")

# ==================================================================
# PART B: Standard SOS via RelaxationEngine (no differential structure)
# ==================================================================
print(f"\n{'='*60}")
print("PART B: Standard SOS — RelaxationEngine")
print("=" * 60)

sg = CommutativeSemigroup(['x', 'y'])
sga = SemigroupAlgebra(sg)
x_sa, y_sa = sga['x'], sga['y']

prog = OptimizationProblem(sga)
# Objective: x^2 + y^2 - 2*x (minimum at (1,0), value -1)
prog.set_objective(x_sa**2 + y_sa**2 - 2 * x_sa)
prog.add_constraints([4 - x_sa**2 - y_sa**2])  # Archimedean: x^2+y^2 <= 4

engine_config = RelaxationConfig(
    reduction_method="newton_polytope",
    monomial_pruning=True,
    sparsity_detection=False,
    verbose_reduction=True,
)

engine = RelaxationEngine(prog, order=2, solver="cvxopt", config=engine_config)
sos_result = engine.solve("sos")

print(f"\n  SOS Lower Bound: {sos_result.value:.8f}")
print(f"  Status:          {sos_result.status}")
print(f"  Runtime:         {sos_result.runtime:.3f}s")
print(f"  Expected min:    -1.0 (at x=1, y=0)")

# ==================================================================
# PART C: SONC relaxation via RelaxationEngine
# ==================================================================
print(f"\n{'='*60}")
print("PART C: SONC Relaxation — RelaxationEngine")
print("=" * 60)

sg2 = CommutativeSemigroup(['x', 'y'])
sga2 = SemigroupAlgebra(sg2)
x2, y2 = sga2['x'], sga2['y']

prog2 = OptimizationProblem(sga2)
# Posynomial-friendly objective: x + 1/x on [0.5, 2] (min at x=1, value=2)
prog2.set_objective(x2 + y2)
prog2.add_constraints([2 - x2])       # upper bound
prog2.add_constraints([x2 - 0.5])     # lower bound
prog2.add_constraints([2 - y2])
prog2.add_constraints([y2 - 0.5])

sonc_result = engine.solve("sonc")

print(f"\n  SONC Lower Bound: {sonc_result.value:.8f}")
print(f"  Status:           {sonc_result.status}")
print(f"  Runtime:          {sonc_result.runtime:.3f}s")

# ==================================================================
# Summary
# ==================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print("=" * 60)
print(f"  DSDP (1D ADE):   {'Converged' if result_a is not None else 'See details above'}")
print(f"  SOS (order 2):   {sos_result.value:.6f}  [{sos_result.status}]")
print(f"  SONC:            {sonc_result.value:.6f}  [{sonc_result.status}]")
print("=" * 60)
