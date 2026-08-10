#!/usr/bin/env python3
"""Diagnose which P3 config component causes -inf on unconstrained problems."""

import time
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxation_api import RelaxationEngine
from Irene.relaxations import RelaxationConfig


def build_motzkin():
    sg = CommutativeSemigroup(["x", "y"])
    sa = SemigroupAlgebra(sg)
    x, y = sa["x"], sa["y"]
    prog = OptimizationProblem(sa)
    f = x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2
    prog.set_objective(f)
    return prog


configs = [
    ("Baseline (none)", RelaxationConfig(reduction_method="none", monomial_pruning=False, sparsity_detection=False)),
    ("Newton pruning only", RelaxationConfig(reduction_method="none", monomial_pruning=True, sparsity_detection=False)),
    ("reduction=newton_polytope", RelaxationConfig(reduction_method="newton_polytope", monomial_pruning=False, sparsity_detection=False)),
    ("Sparsity only", RelaxationConfig(reduction_method="none", monomial_pruning=False, sparsity_detection=True)),
    ("Newton + pruning", RelaxationConfig(reduction_method="newton_polytope", monomial_pruning=True, sparsity_detection=False)),
    ("P3 full (broken)", RelaxationConfig(reduction_method="newton_polytope", monomial_pruning=True, sparsity_detection=True)),
]

prog = build_motzkin()

print(f"{'Config':<30} {'Order1 val':>12} {'Order2 val':>12} {'Order3 val':>12}")
print("-" * 70)

for label, cfg in configs:
    vals = []
    for order in [1, 2, 3]:
        engine = RelaxationEngine(prog, order=order, solver="clarabel",
                                  verbosity=0, config=cfg)
        res = engine.solve("sos")
        v = res.value
        if abs(v) > 1e6:
            vals.append("-inf" if v < 0 else "+inf")
        elif abs(v) < 1e-8:
            vals.append(f"{v:.2e}")
        else:
            vals.append(f"{v:.4f}")

    print(f"{label:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")
