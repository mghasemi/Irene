"""Reference border-basis checks based on classical quotient-space examples."""

from itertools import product

import numpy as np


def monomial_exponents(nvars, max_deg):
    """All exponent tuples of total degree <= max_deg, sorted by degree descending."""
    exps = [exp for exp in product(range(max_deg + 1), repeat=nvars) if sum(exp) <= max_deg]
    exps.sort(key=lambda e: -sum(e))
    return exps


def poly_mul_monomial(poly_dict, shift_exp):
    """Multiply polynomial dictionary by monomial x^shift_exp."""
    return {tuple(a + b for a, b in zip(exp, shift_exp)): c for exp, c in poly_dict.items()}


def compute_border_basis(nvars, generators_dict, degree):
    """Compute a border basis from shifted-generator relations up to degree+1."""
    all_exps_d = [e for e in monomial_exponents(nvars, degree) if sum(e) <= degree]
    all_exps_d1 = monomial_exponents(nvars, degree + 1)

    exp_to_idx_d1 = {exp: i for i, exp in enumerate(all_exps_d1)}
    n_total = len(all_exps_d1)

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

    rel_matrix = np.array(relations, dtype=float)
    _u_svd, svals, vt = np.linalg.svd(rel_matrix, full_matrices=False)
    tol = 1e-10 * max(rel_matrix.shape) * svals[0] if len(svals) > 0 else 1e-10

    pivot_cols = set()
    for col_idx in range(n_total):
        col = rel_matrix[:, col_idx]
        if np.linalg.norm(col) < tol:
            continue
        if not pivot_cols:
            pivot_cols.add(col_idx)
            continue

        pivot_matrix = rel_matrix[:, list(pivot_cols)]
        proj = pivot_matrix @ (np.linalg.pinv(pivot_matrix) @ col)
        residual = np.linalg.norm(col - proj)
        if residual > tol:
            pivot_cols.add(col_idx)

    basis_set = set()
    for exp in all_exps_d:
        idx = exp_to_idx_d1.get(exp)
        if idx is not None and idx not in pivot_cols:
            basis_set.add(exp)

    if (0,) * nvars not in basis_set and len(basis_set) == 0:
        basis_set = set(all_exps_d)

    basis = sorted(basis_set)

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
    mult_tables = {}

    if border and len(basis) > 0:
        n_basis = len(basis)
        rank = int(np.sum(svals > tol)) if len(svals) > 0 else 0

        basis_matrix = np.zeros((n_total, n_basis))
        for j, b_exp in enumerate(basis):
            if b_exp in exp_to_idx_d1:
                basis_matrix[exp_to_idx_d1[b_exp], j] = 1.0

        if rank > 0:
            row_basis = vt[:rank].T
            augmented = np.hstack([row_basis, basis_matrix])
        else:
            augmented = basis_matrix

        for border_exp in border:
            target = np.zeros(n_total)
            if border_exp in exp_to_idx_d1:
                target[exp_to_idx_d1[border_exp]] = 1.0

            sol, _, _, _ = np.linalg.lstsq(augmented, target, rcond=None)
            coeffs = sol[-n_basis:] if rank > 0 else sol
            mult_tables[border_exp] = coeffs

    return basis, border, mult_tables


def test_single_variable_quadratic_relation():
    gen = [{(2,): 1.0, (0,): -2.0}]
    basis, border, mt = compute_border_basis(1, gen, degree=2)

    assert len(basis) == 2
    assert (0,) in set(basis)
    assert (1,) in set(basis)
    assert (2,) in border

    if (2,) in mt:
        one_idx = basis.index((0,))
        assert abs(mt[(2,)][one_idx] - 2.0) < 0.1


def test_two_variable_square_ideal_basis():
    gen = [{(2, 0): 1.0}, {(0, 2): 1.0}]
    basis, _border, mt = compute_border_basis(2, gen, degree=2)

    assert len(basis) == 4
    assert (0, 0) in set(basis)
    assert (1, 0) in set(basis)
    assert (0, 1) in set(basis)
    assert (1, 1) in set(basis)
    assert (2, 0) not in set(basis)
    assert (0, 2) not in set(basis)

    if (2, 0) in mt:
        assert max(abs(c) for c in mt[(2, 0)]) < 0.1


def test_xy_minus_one_excludes_xy_from_basis():
    gen = [{(1, 1): 1.0, (0, 0): -1.0}]
    basis, _border, _mt = compute_border_basis(2, gen, degree=2)

    assert (1, 1) not in set(basis)


def test_coupled_quadratic_relations_have_dimension_at_least_three():
    gen = [
        {(2, 0): 1.0, (0, 1): -1.0},
        {(0, 2): 1.0, (1, 0): -1.0},
    ]
    basis, _border, _mt = compute_border_basis(2, gen, degree=3)

    assert len(basis) >= 3


def test_free_algebra_has_expected_degree_two_dimension():
    basis, _border, _mt = compute_border_basis(1, [], degree=2)

    assert len(basis) == 3


def test_single_quadratic_constraint_dimension_five():
    gen = [{(2, 0): 1.0, (0, 2): 1.0, (0, 0): -3.0}]
    basis, _border, mt = compute_border_basis(2, gen, degree=2)

    assert len(basis) == 5

    if (2, 0) in mt and (0, 2) in mt:
        sum_coeffs = mt[(2, 0)] + mt[(0, 2)]
        one_idx = basis.index((0, 0))
        assert abs(sum_coeffs[one_idx] - 3.0) < 0.5
