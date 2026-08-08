"""Validation tests for the corrected BorderBasis implementation.

Runs 6 mathematical test cases against known results from Greuel-Pfister theory.
All tests must pass before P3.1 can be marked complete.
"""
import sys
sys.path.insert(0, '/home/mehdi/Code/Python/IreneRewrite')

from Irene.border_basis import BorderBasis
from Irene.symbolic_engine import engine


def test_1_x2_minus_2():
    """I = <x^2 - 2> in Q[x], degree=2. Expected: basis={1,x}, x^2 -> 2*1."""
    print("=" * 70)
    print("Test 1: I = <x^2 - 2>, degree=2")
    x = engine.Symbol('x')
    bb = BorderBasis([x], [x**2 - 2], degree=2)

    assert len(bb.basis) == 2, f"Basis size should be 2, got {len(bb.basis)}"
    assert (0,) in set(bb.basis), "1 should be in basis"
    assert (1,) in set(bb.basis), "x should be in basis"
    assert (2,) in bb.border, "x^2 should be in border"

    # x^2 = 2 mod I
    coeffs = bb.mult_tables[(2,)]
    one_idx = bb.basis.index((0,))
    assert abs(coeffs[one_idx] - 2.0) < 0.1, f"x^2 coeff of 1 should be ~2, got {coeffs[one_idx]}"

    # Verify reduce() works correctly
    reduced = bb.reduce(x**2 + x)
    poly_dict = engine.Poly(reduced, x).as_dict()
    assert abs(poly_dict.get((0,), 0) - 2.0) < 0.1, f"reduce(x^2+x): const should be ~2+0=2"
    assert abs(poly_dict.get((1,), 0) - 1.0) < 0.1, f"reduce(x^2+x): x coeff should be ~1"

    print(f"  Basis: {bb.basis}, Border: {bb.border}")
    print(f"  x^2 mod I -> coeff of 1 = {coeffs[one_idx]:.4f} (expected ~2.0)")
    print("  PASS")


def test_2_x2_y2():
    """I = <x^2, y^2> in Q[x,y], degree=2. Expected: basis={1,x,y,xy}, dim=4."""
    print()
    print("=" * 70)
    print("Test 2: I = <x^2, y^2>, degree=2")
    x, y = engine.Symbol('x'), engine.Symbol('y')
    bb = BorderBasis([x, y], [x**2, y**2], degree=2)

    assert len(bb.basis) == 4, f"Basis size should be 4, got {len(bb.basis)}"
    basis_set = set(bb.basis)
    assert (0, 0) in basis_set, "1 should be in basis"
    assert (1, 0) in basis_set, "x should be in basis"
    assert (0, 1) in basis_set, "y should be in basis"
    assert (1, 1) in basis_set, "xy should be in basis"
    assert (2, 0) not in basis_set, "x^2 should NOT be in basis"
    assert (0, 2) not in basis_set, "y^2 should NOT be in basis"

    # x^2 = 0 mod I
    if (2, 0) in bb.mult_tables:
        max_coeff = max(abs(c) for c in bb.mult_tables[(2, 0)])
        assert max_coeff < 0.1, f"x^2 should be ~0, got max coeff {max_coeff}"

    print(f"  Basis: {bb.basis}, Border: {bb.border}")
    print("  PASS")


def test_3_xy_minus_1():
    """I = <xy - 1> in Q[x,y], degree=2. Expected: xy NOT in basis."""
    print()
    print("=" * 70)
    print("Test 3: I = <xy - 1>, degree=2")
    x, y = engine.Symbol('x'), engine.Symbol('y')
    bb = BorderBasis([x, y], [x*y - 1], degree=2)

    assert (1, 1) not in set(bb.basis), "xy should NOT be in basis (equals 1 mod I)"

    # Verify: xy -> 1*const via reduce
    reduced = bb.reduce(x * y)
    poly_dict = engine.Poly(reduced, x, y).as_dict()
    const_val = poly_dict.get((0, 0), 0)
    assert abs(const_val - 1.0) < 0.1, f"reduce(xy) should be ~1, got {const_val}"

    print(f"  Basis: {bb.basis}, Border: {bb.border}")
    print(f"  reduce(xy) -> const = {const_val:.4f} (expected ~1.0)")
    print("  PASS")


def test_4_x2_minus_y():
    """I = <x^2-y, y^2-x> in Q[x,y], degree=3. Expected: dim >= 3."""
    print()
    print("=" * 70)
    print("Test 4: I = <x^2-y, y^2-x>, degree=3")
    x, y = engine.Symbol('x'), engine.Symbol('y')
    bb = BorderBasis([x, y], [x**2 - y, y**2 - x], degree=3)

    assert len(bb.basis) >= 3, f"Basis should have dim >= 3, got {len(bb.basis)}"

    print(f"  Basis: {bb.basis} (size={len(bb.basis)})")
    print("  PASS")


def test_5_free_algebra():
    """I = <0> in Q[x], degree=2. Expected: basis={1,x,x^2}, dim=3."""
    print()
    print("=" * 70)
    print("Test 5: I = <0>, degree=2 (free algebra)")
    x = engine.Symbol('x')
    bb = BorderBasis([x], [], degree=2)

    assert len(bb.basis) == 3, f"Free algebra dim should be 3 at deg 2, got {len(bb.basis)}"

    print(f"  Basis: {bb.basis}")
    print("  PASS")


def test_6_motzkin_constraint():
    """I = <x^2+y^2-3> in Q[x,y], degree=2. Expected: dim=5."""
    print()
    print("=" * 70)
    print("Test 6: I = <x^2+y^2-3>, degree=2")
    x, y = engine.Symbol('x'), engine.Symbol('y')
    bb = BorderBasis([x, y], [x**2 + y**2 - 3], degree=2)

    assert len(bb.basis) == 5, f"Expected dim 5, got {len(bb.basis)}"

    # Verify: x^2 + y^2 = 3 mod I (if both are border elements)
    if (2, 0) in bb.mult_tables and (0, 2) in bb.mult_tables:
        coeffs_x2 = bb.mult_tables[(2, 0)]
        coeffs_y2 = bb.mult_tables[(0, 2)]
        sum_coeffs = coeffs_x2 + coeffs_y2
        one_idx = bb.basis.index((0, 0))
        assert abs(sum_coeffs[one_idx] - 3.0) < 0.5, \
            f"x^2+y^2 should equal ~3*1, got {sum_coeffs[one_idx]}"

    print(f"  Basis: {bb.basis}, Border: {bb.border}")
    if (2, 0) in bb.mult_tables and (0, 2) in bb.mult_tables:
        one_idx = bb.basis.index((0, 0))
        print(f"  x^2+y^2 mod I -> coeff of 1 = {sum_coeffs[one_idx]:.4f} (expected ~3.0)")
    print("  PASS")


def test_7_conditioning():
    """Verify conditioning diagnostics work."""
    print()
    print("=" * 70)
    print("Test 7: Conditioning diagnostic")
    x = engine.Symbol('x')
    bb = BorderBasis([x], [x**2 - 2], degree=2)

    diag = bb.conditioning_diagnostic()
    assert 'condition_number' in diag, "Missing condition_number"
    assert 'basis_conditioning' in diag, "Missing basis_conditioning"
    assert 'is_well_conditioned' in diag, "Missing is_well_conditioned"

    print(f"  Condition number: {diag['condition_number']:.2e}")
    print(f"  Basis conditioning: {diag['basis_conditioning']:.2e}")
    print(f"  Well conditioned: {diag['is_well_conditioned']}")
    print("  PASS")


def test_8_moment_matrix_structure():
    """Verify moment matrix structure output."""
    print()
    print("=" * 70)
    print("Test 8: Moment matrix structure")
    x, y = engine.Symbol('x'), engine.Symbol('y')
    bb = BorderBasis([x, y], [x**2, y**2], degree=2)

    struct = bb.moment_matrix_structure()
    assert struct['basis_size'] == 4, f"Basis size should be 4"
    assert 'block_structure' in struct, "Missing block_structure"

    print(f"  Basis: {struct['basis_size']}, Border: {struct['border_size']}")
    print("  PASS")


if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_1_x2_minus_2,
        test_2_x2_y2,
        test_3_xy_minus_1,
        test_4_x2_minus_y,
        test_5_free_algebra,
        test_6_motzkin_constraint,
        test_7_conditioning,
        test_8_moment_matrix_structure,
    ]

    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")

    print()
    print("=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} tests passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED — BorderBasis implementation validated")
    else:
        print("SOME TESTS FAILED — algorithm needs correction")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
