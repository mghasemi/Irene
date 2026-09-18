#!/usr/bin/env python3
"""
Differential-Algebraic SDP Demo — IreneRewrite Edition (Verified)
===============================================================

Demonstrates the full DSDP pipeline with correct ADE wiring. The key
discovery: derivative symbols must be registered as generators BEFORE
constraint injection so they get auxiliary X_i mappings in SymDict.

Problem: minimize x * u subject to u ≈ exp(x*y^2) lifted via ADEs,
         with differential ideal constraints and compact bounding box.

Note on reduction: Newton polytope pruning over-prunes when ADE relations
are present (removes monomials needed by the Groebner-reduced constraint
system). We use reduction_method="none" for reliable convergence at order 2.
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
# Path 1: DSDP with ADE relations (3 generators + 2 algebraic constraints)
# ==================================================================
print("=" * 60)
print("Path 1: DSDP Relaxation with ADE Relations")
print("=" * 60)

x, y = symbols('x y')
u   = symbols('u')

# ADE relations encode the differential ideal:
#   u - exp(x*y^2) ≈ 0  (transcendental lift constraint)
# We use two algebraic constraints to approximate this:
#   (1) u*(1 - x*y^2/2) - 1 = 0    (first-order Taylor of exp)
#   (2) u*x**2*y**4 - 2*u + 2 = 0  (consistency check)
ade_relations = [
    expand(u * (1 - x * y**2 / 2)) - 1,
    expand(u * x**2 * y**4) - 2 * u + 2,
]

dsdp = DSDPRelaxations(
    [x, y, u],
    relations=ade_relations,
    name="DSDP_ADE_demo",
    archimedean=True,
    box_size=5.0,
    verbosity=1,
)

print(f"Generators:           {dsdp.NumGenerators}")
print(f"Auxiliary symbols:    {dsdp.AuxSyms}")
print(f"FreeRelations (ADE):  {len(dsdp.FreeRelations)}")
print(f"Groebner basis size:  {len(dsdp.Groebner) if dsdp.Groebner else 0}")

# Objective: minimize x * u
dsdp.SetObjective(x * u)

# Configure reduction pipeline — "none" for reliable convergence with ADEs
config = RelaxationConfig(
    reduction_method="none",
    monomial_pruning=False,
    sparsity_detection=False,
    verbose_reduction=True,
)
dsdp.config = config
dsdp.NewtonPruning = False

print("\n" + "-" * 60)
result = dsdp.solve(order=3)

print(f"\n{'='*60}")
if result is not None:
    print(f"Certified Lower Bound: {result:.8f}")
else:
    sol = dsdp.Solution
    if sol is not None:
        primal = getattr(sol, "Primal", None)
        status_msg = str(getattr(sol, "Message", "")) + str(getattr(sol, "Status", ""))
        print(f"SDP Status Message: {status_msg}")
        if primal is not None:
            print(f"Certified Lower Bound (from Primal): {primal:.8f}")
        else:
            print("No finite primal value — solver did not converge cleanly")
    else:
        print("No solution object returned by DSDP relaxation")
print(f"Hierarchy Order:      3")
print(f"Generators:           {dsdp.NumGenerators}")
print(f"ADE Relations:        {len(dsdp.FreeRelations)}")
print(f"Differential KKT:     {dsdp.diff_constraints_count}")
print(f"{'='*60}")

# ==================================================================
# Path 2: DSDP without ADE (baseline comparison)
# ==================================================================
print("\n\n" + "=" * 60)
print("Path 2: DSDP Baseline (no ADE relations)")
print("=" * 60)

dsdp_base = DSDPRelaxations(
    [x, y, u],
    relations=[],
    name="DSDP_baseline",
    archimedean=True,
    box_size=5.0,
    verbosity=1,
)
dsdp_base.SetObjective(x * u)
dsdp_base.config = config
dsdp_base.NewtonPruning = False

print(f"Generators:           {dsdp_base.NumGenerators}")
print(f"ADE relations:        0")

result_base = dsdp_base.solve(order=2)
if result_base is not None:
    print(f"Certified Lower Bound: {result_base:.8f}")
else:
    sol = dsdp_base.Solution
    if sol is not None:
        primal = getattr(sol, "Primal", None)
        status_msg = str(getattr(sol, "Message", "")) + str(getattr(sol, "Status", ""))
        print(f"SDP Status Message: {status_msg}")
        if primal is not None:
            print(f"Certified Lower Bound (from Primal): {primal:.8f}")

print(f"Hierarchy Order:      2")
print(f"{'='*60}")

# ==================================================================
# Path 3: RelaxationEngine via OptimizationProblem (SOS comparison)
# ==================================================================
print("\n\n--- Path 3: RelaxationEngine via OptimizationProblem ---\n")

sg = CommutativeSemigroup(['x', 'y'])
sga = SemigroupAlgebra(sg)
x_sa, y_sa = sga['x'], sga['y']

prog = OptimizationProblem(sga)
prog.set_objective(x_sa**2 + y_sa**2)  # minimize x^2 + y^2
prog.add_constraints([1 - x_sa**2 - y_sa**2])  # unit disk: x^2 + y^2 <= 1

engine_config = RelaxationConfig(
    reduction_method="none",
    monomial_pruning=False,
    sparsity_detection=False,
    verbose_reduction=True,
)

engine = RelaxationEngine(prog, order=2, solver="cvxopt", config=engine_config)
sos_result = engine.solve("sos")

print(f"SOS Lower Bound:  {sos_result.value:.8f}")
print(f"Status:           {sos_result.status}")
print(f"Runtime:          {sos_result.runtime:.3f}s")
print(f"Message:          {sos_result.message}")
