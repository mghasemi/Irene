#!/usr/bin/env python3
"""Test Newton polytope pruning in SDPRelaxations."""

import sys
sys.path.insert(0, '/home/mehdi/Code/Python/IreneRewrite')

from Irene.relaxations import SDPRelaxations
from sympy import symbols

x, y = symbols('x y')

# Test 1: Sparse polynomial (Motzkin-like) - pruning should help significantly
print("=" * 60)
print("Test 1: Sparse Motzkin-like problem")
print("=" * 60)

rlx = SDPRelaxations([x, y])
rlx.SetObjective(x**4 * y**4 - x**2 * y**2 - 1)
rlx.MomentsOrd(1)
rlx.RelaxationDeg()

# Without pruning
SDPRelaxations.NewtonPruning = False
rlx.ReducedBases = {}  # clear cache
basis_full = rlx.ReducedMonomialBase(rlx.MmntOrd)
print(f"Basis size WITHOUT pruning: {len(basis_full)}")

# With pruning
SDPRelaxations.NewtonPruning = True
rlx.ReducedBases = {}  # clear cache
basis_pruned = rlx.ReducedMonomialBase(rlx.MmntOrd)
print(f"Basis size WITH pruning:    {len(basis_pruned)}")

reduction = (1 - len(basis_pruned) / len(basis_full)) * 100
print(f"Reduction: {reduction:.1f}%")

# Verify correctness: pruned basis should be a subset of full basis
pruned_set = set(basis_pruned)
full_set = set(basis_full)
assert pruned_set.issubset(full_set), "PRUNING ERROR: pruned basis contains elements not in full basis!"
print("✓ Pruned basis is a valid subset of full basis")

# Test 2: Dense polynomial - pruning should have minimal effect
print()
print("=" * 60)
print("Test 2: Dense polynomial (all monomials present)")
print("=" * 60)

rlx2 = SDPRelaxations([x, y])
rlx2.SetObjective(x**4 + x**3*y + x**2*y**2 + x*y**3 + y**4 - 1)
rlx2.AddConstraint(x**2 + y**2 >= 1)
rlx2.MomentsOrd(1)
rlx2.RelaxationDeg()

SDPRelaxations.NewtonPruning = False
rlx2.ReducedBases = {}
basis_full2 = rlx2.ReducedMonomialBase(rlx2.MmntOrd)
print(f"Basis size WITHOUT pruning: {len(basis_full2)}")

SDPRelaxations.NewtonPruning = True
rlx2.ReducedBases = {}
basis_pruned2 = rlx2.ReducedMonomialBase(rlx2.MmntOrd)
print(f"Basis size WITH pruning:    {len(basis_pruned2)}")

reduction2 = (1 - len(basis_pruned2) / len(basis_full2)) * 100
print(f"Reduction: {reduction2:.1f}%")

pruned_set2 = set(basis_pruned2)
full_set2 = set(basis_full2)
assert pruned_set2.issubset(full_set2), "PRUNING ERROR: pruned basis contains elements not in full basis!"
print("✓ Pruned basis is a valid subset of full basis")

# Test 3: Verify ExponentsVec also respects pruning
print()
print("=" * 60)
print("Test 3: ExponentsVec consistency with pruning")
print("=" * 60)

rlx3 = SDPRelaxations([x, y])
rlx3.SetObjective(x**4 - x*y + y**2)
rlx3.MomentsOrd(1)
rlx3.RelaxationDeg()

SDPRelaxations.NewtonPruning = False
rlx3.ReducedBases = {}
exp_full = rlx3.ExponentsVec(rlx3.MmntOrd)
print(f"Exponent vector size WITHOUT pruning: {len(exp_full)}")

SDPRelaxations.NewtonPruning = True
rlx3.ReducedBases = {}
exp_pruned = rlx3.ExponentsVec(rlx3.MmntOrd)
print(f"Exponent vector size WITH pruning:    {len(exp_pruned)}")

assert len(exp_pruned) <= len(exp_full), "Pruning should never increase basis size!"
print("✓ Exponent vector respects pruning")

# Reset to default
SDPRelaxations.NewtonPruning = False
print()
print("=" * 60)
print("All tests passed!")
print("=" * 60)
