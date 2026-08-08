"""
Border Basis Algorithm — Reference Implementation (Greuel-Pfister style)

The border basis method works as follows:
1. Fix a monomial basis B = {x^alpha : |alpha| <= d} for the quotient space K[x]/I
2. Compute the border dB = {b*x_i : b in B, i=1..n} \ B
3. For each f in dB, find coefficients c_beta such that f = sum c_beta x^beta (mod I)

Key insight: build a relation matrix from shifted generators. Each row is the
coefficient vector of x^gamma * g_i over all monomials of degree <= d+1.
Pivot columns represent eliminable monomials; non-pivots form the basis B.

CRITICAL: Process columns from HIGHEST to LOWEST degree so that higher-degree
monomials are eliminated first, keeping lower-degree ones in the basis. This
matches the mathematical convention where we reduce leading terms.

Reference: Greuel & Pfister, "A Border Basis Algorithm", 2002
"""

import numpy as np
from itertools import product


def monomial_exponents(nvars, max_deg):
    """All exponent tuples of total degree <= max_deg, sorted by degree descending."""
    exps = [exp for exp in product(range(max_deg + 1), repeat=nvars) if sum(exp) <= max_deg]
    # Sort by total degree DESCENDING so higher-degree monomials are processed first
    exps.sort(key=lambda e: -sum(e))
    return exps


def poly_mul_monomial(p, shift_exp):
    """Multiply polynomial p (dict) by monomial x^shift_exp."""
    return {tuple(a + b for a, b in zip(exp, shift_exp)): c
            for exp, c in p.items()}


def compute_border_basis(nvars, generators_dict, degree):
    """
    Compute border basis using the Greuel-Pfister algorithm.

    Args:
        nvars: number of variables
        generators_dict: list of {exp_tuple: coeff} representing generators
        degree: max degree for monomial basis

    Returns:
        basis: list of exponent tuples (standard monomials)
        border: list of exponent tuples (border elements)
        mult_tables: dict mapping each border exp to coefficient vector over basis
    """
    # Step 1: Get all monomials up to degree d+1, sorted by degree DESCENDING
    all_exps_d = [e for e in monomial_exponents(nvars, degree) if sum(e) <= degree]
    all_exps_d1 = monomial_exponents(nvars, degree + 1)

    exp_to_idx_d1 = {exp: i for i, exp in enumerate(all_exps_d1)}
    n_total = len(all_exps_d1)

    # Step 2: Build relation matrix from shifted generators
    relations = []

    for gen in generators_dict:
        if not gen:
            continue
        max_shift_deg = degree + 1 - max(sum(exp) for exp in gen.keys())
        if max_shift_deg < 0:
            continue

        shift_exps = [e for e in all_exps_d1 if sum(e) <= max_shift_deg]
        for gamma in shift_exps:
            shifted = poly_mul_monomial(gen, gamma)
            if not shifted:
                continue

            row = np.zeros(n_total)
            for exp, coeff in shifted.items():
                if exp in exp_to_idx_d1:
                    row[exp_to_idx_d1[exp]] += float(coeff)

            if np.any(row != 0):
                relations.append(row)

    if not relations:
        return sorted(all_exps_d), [], {}

    R = np.array(relations, dtype=float)

    # Step 3: Find pivot columns via greedy selection (iterate from highest degree first)
    U_svd, S, Vt = np.linalg.svd(R, full_matrices=False)
    tol = 1e-10 * max(R.shape) * S[0] if len(S) > 0 else 1e-10

    pivot_cols = set()
    for col_idx in range(n_total):
        col = R[:, col_idx]
        if np.linalg.norm(col) < tol:
            continue
        if not pivot_cols:
            pivot_cols.add(col_idx)
        else:
            # Check if this column is in the span of previous pivots
            pivot_matrix = R[:, list(pivot_cols)]
            proj = pivot_matrix @ (np.linalg.pinv(pivot_matrix) @ col)
            residual = np.linalg.norm(col - proj)
            if residual > tol:
                pivot_cols.add(col_idx)

    # Standard monomials: those in degree <= d whose column index is NOT a pivot
    basis_set = set()
    for exp in all_exps_d:
        idx = exp_to_idx_d1.get(exp)
        if idx is not None and idx not in pivot_cols:
            basis_set.add(exp)

    # Safety: ensure we have at least the constant term
    if (0,) * nvars not in basis_set and len(basis_set) == 0:
        basis_set = set(all_exps_d)

    basis = sorted(basis_set)

    # Step 4: Compute the border dB
    basis_frozen = frozenset(basis)
    border_set = set()

    for exp in basis:
        for i in range(nvars):
            new_exp = list(exp)
            new_exp[i] += 1
            new_tuple = tuple(new_exp)
            if sum(new_tuple) <= degree + 1 and new_tuple not in basis_frozen:
                border_set.add(new_tuple)

    border = sorted(border_set)

    # Step 5: Compute multiplication tables via augmented system solve
    # For each border element f, we need to find c such that:
    #   e_f - sum_j c_j * e_{beta_j} is in RowSpace(R)
    # i.e., e_f = V_row @ lambda + B_mat @ c  for some lambda
    # where V_row spans the row space of R (subset of R^{n_total}).
    # SVD: R = U @ diag(S) @ Vt, so RowSpace(R) is spanned by Vt[:rank].T
    # which has shape (n_total, rank).

    mult_tables = {}

    if border and len(basis) > 0:
        n_basis = len(basis)
        rank = int(np.sum(S > tol)) if len(S) > 0 else 0

        # Build basis indicator columns B_mat (n_total x n_basis matrix)
        B_mat = np.zeros((n_total, n_basis))
        for j, b_exp in enumerate(basis):
            if b_exp in exp_to_idx_d1:
                idx = exp_to_idx_d1[b_exp]
                B_mat[idx, j] = 1.0

        # Row space basis of R lives in R^{n_total}: Vt[:rank].T has shape (n_total, rank)
        if rank > 0:
            V_row = Vt[:rank].T  # n_total x rank, orthonormal basis for RowSpace(R) ⊂ R^{n_total}
            A_aug = np.hstack([V_row, B_mat])  # n_total x (rank + n_basis)
        else:
            A_aug = B_mat

        for border_exp in border:
            target = np.zeros(n_total)
            if border_exp in exp_to_idx_d1:
                target[exp_to_idx_d1[border_exp]] = 1.0

            # Solve A_aug @ [lambda; c] ≈ target via least squares
            sol, _, _, _ = np.linalg.lstsq(A_aug, target, rcond=None)

            # The last n_basis entries of sol are the coefficients c_j
            coeffs = sol[-n_basis:] if rank > 0 else sol

            mult_tables[border_exp] = coeffs

    return basis, border, mult_tables


# ============================================================
# Test against known mathematical results
# ============================================================

def run_tests():
    passed = 0
    failed = 0

    # ---- Test 1: I = <x^2 - 2> in Q[x], degree=2 ----
    print("=" * 70)
    print("Test 1: I = <x^2 - 2> in Q[x], degree=2")
    print("Expected: Basis = {1, x} (dim=2), Border contains x^2")
    print("x^2 mod I should be ~2*1 (since x^2 = 2)")
    print("=" * 70)

    gen1 = [{(2,): 1.0, (0,): -2.0}]
    basis, border, mt = compute_border_basis(1, gen1, degree=2)
    print(f"Basis: {basis}")
    print(f"Border: {border}")

    try:
        assert len(basis) == 2, f"Basis size should be 2, got {len(basis)}"
        assert (0,) in set(basis), "Constant term 1 should be in basis"
        assert (1,) in set(basis), "x should be in basis"
        # x^2 is a border element: multiplying x by x gives x^2 which is not in basis
        assert (2,) in border, "x^2 should be in border"

        # Check mult table: x^2 = 2 mod I
        if (2,) in mt:
            coeffs = mt[(2,)]
            one_idx = basis.index((0,))
            assert abs(coeffs[one_idx] - 2.0) < 0.1, f"x^2 coeff of 1 should be ~2, got {coeffs[one_idx]}"
            print(f"  x^2 mod I: coeff of 1 = {coeffs[one_idx]:.4f} (expected ~2.0)")

        passed += 1
        print("PASS")
    except AssertionError as e:
        failed += 1
        print(f"FAIL: {e}")

    # ---- Test 2: I = <x^2, y^2> in Q[x,y], degree=2 ----
    # In Q[x,y]/<x^2, y^2>, the basis is {1, x, y, xy} — dim=4
    print()
    print("=" * 70)
    print("Test 2: I = <x^2, y^2> in Q[x,y], degree=2")
    print("Expected: Basis = {1, x, y, xy} (dim=4)")
    print("=" * 70)

    gen2 = [{(2, 0): 1.0}, {(0, 2): 1.0}]
    basis2, border2, mt2 = compute_border_basis(2, gen2, degree=2)
    print(f"Basis: {basis2}")
    print(f"Border: {border2}")

    try:
        assert len(basis2) == 4, f"Basis size should be 4, got {len(basis2)}"
        assert (0, 0) in set(basis2), "1 should be in basis"
        assert (1, 0) in set(basis2), "x should be in basis"
        assert (0, 1) in set(basis2), "y should be in basis"
        assert (1, 1) in set(basis2), "xy should be in basis"
        assert (2, 0) not in set(basis2), "x^2 should NOT be in basis"
        assert (0, 2) not in set(basis2), "y^2 should NOT be in basis"

        # Check: x^2 = 0 mod I
        if (2, 0) in mt2:
            coeffs_x2 = mt2[(2, 0)]
            max_coeff = max(abs(c) for c in coeffs_x2)
            assert max_coeff < 0.1, f"x^2 should be ~0, got max coeff {max_coeff}"

        passed += 1
        print("PASS")
    except AssertionError as e:
        failed += 1
        print(f"FAIL: {e}")

    # ---- Test 3: I = <xy - 1> in Q[x,y], degree=2 ----
    print()
    print("=" * 70)
    print("Test 3: I = <xy - 1> in Q[x,y], degree=2")
    print("Expected: xy should NOT be in basis (since xy = 1 mod I)")
    print("=" * 70)

    gen3 = [{(1, 1): 1.0, (0, 0): -1.0}]
    basis3, border3, mt3 = compute_border_basis(2, gen3, degree=2)
    print(f"Basis: {basis3}")
    print(f"Border: {border3}")

    try:
        assert (1, 1) not in set(basis3), "xy should NOT be in basis (equals 1 mod I)"
        passed += 1
        print("PASS")
    except AssertionError as e:
        failed += 1
        print(f"FAIL: {e}")

    # ---- Test 4: I = <x^2-y, y^2-x> in Q[x,y], degree=3 ----
    print()
    print("=" * 70)
    print("Test 4: I = <x^2-y, y^2-x> in Q[x,y], degree=3")
    print("Expected: dim >= 3 (basis includes {1, x, y})")
    print("=" * 70)

    gen4 = [
        {(2, 0): 1.0, (0, 1): -1.0},
        {(0, 2): 1.0, (1, 0): -1.0}
    ]
    basis4, border4, mt4 = compute_border_basis(2, gen4, degree=3)
    print(f"Basis: {basis4}")
    print(f"Border: {border4}")
    print(f"Basis size: {len(basis4)}")

    try:
        assert len(basis4) >= 3, f"Basis should have dim >= 3, got {len(basis4)}"
        passed += 1
        print("PASS")
    except AssertionError as e:
        failed += 1
        print(f"FAIL: {e}")

    # ---- Test 5: No generators (free algebra) ----
    print()
    print("=" * 70)
    print("Test 5: I = <0> in Q[x], degree=2 (no relations)")
    print("Expected: Basis = {1, x, x^2} (dim=3)")
    print("=" * 70)

    basis5, border5, mt5 = compute_border_basis(1, [], degree=2)
    print(f"Basis: {basis5}")

    try:
        assert len(basis5) == 3, f"Free algebra dim should be 3 at deg 2, got {len(basis5)}"
        passed += 1
        print("PASS")
    except AssertionError as e:
        failed += 1
        print(f"FAIL: {e}")

    # ---- Test 6: Motzkin polynomial ideal context ----
    # I = <x^2 + y^2 - 3> — single quadratic constraint
    print()
    print("=" * 70)
    print("Test 6: I = <x^2+y^2-3> in Q[x,y], degree=2")
    print("Expected: dim=5 (basis {1, x, y, xy, x^2 or y^2})")
    print("=" * 70)

    gen6 = [{(2, 0): 1.0, (0, 2): 1.0, (0, 0): -3.0}]
    basis6, border6, mt6 = compute_border_basis(2, gen6, degree=2)
    print(f"Basis: {basis6}")
    print(f"Border: {border6}")

    try:
        # With one quadratic relation in 2 vars at deg 2:
        # Total monomials at deg<=2: 1+x+y+x^2+xy+y^2 = 6
        # One relation eliminates 1, so dim=5
        assert len(basis6) == 5, f"Expected dim 5, got {len(basis6)}"

        # Verify: x^2 + y^2 = 3 mod I (if both are border elements)
        if (2, 0) in mt6 and (0, 2) in mt6:
            coeffs_x2 = mt6[(2, 0)]
            coeffs_y2 = mt6[(0, 2)]
            # x^2 + y^2 should equal 3*1 in the quotient
            sum_coeffs = coeffs_x2 + coeffs_y2
            one_idx = basis6.index((0, 0))
            assert abs(sum_coeffs[one_idx] - 3.0) < 0.5, \
                f"x^2+y^2 should equal ~3*1, got {sum_coeffs[one_idx]}"
            print(f"  x^2 + y^2 mod I: coeff of 1 = {sum_coeffs[one_idx]:.4f} (expected ~3.0)")

        passed += 1
        print("PASS")
    except AssertionError as e:
        failed += 1
        print(f"FAIL: {e}")

    # ---- Summary ----
    print()
    print("=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} tests passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - algorithm needs correction")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
