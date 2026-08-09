"""Tests for the BorderBasis module.

Validates basis computation, border construction, multiplication tables,
and polynomial reduction modulo ideals using the border basis framework.
"""

import sys
from pathlib import Path

# Ensure parent directory is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Irene.border_basis import BorderBasis
from sympy import symbols


def test_basis_ideal_x2_y2():
    """Ideal <x^2, y^2> with degree=2.

    Quotient K[x,y]/<x^2,y^2> has basis {1, x, y, xy}.
    """
    x, y = symbols("x y")
    bb = BorderBasis([x, y], [x**2, y**2], degree=2)

    expected_basis = {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert set(bb.basis) == expected_basis, f"Basis mismatch: {bb.basis}"
    assert len(bb.border) == 4


def test_basis_ideal_x3_y3():
    """Ideal <x^3, y^3> with degree=3.

    Quotient has basis {1, x, y, x^2, xy, y^2, x^2y, xy^2} (8 elements).
    """
    x, y = symbols("x y")
    bb = BorderBasis([x, y], [x**3, y**3], degree=3)

    expected_basis = {(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2)}
    assert set(bb.basis) == expected_basis, f"Basis mismatch: {bb.basis}"


def test_free_algebra():
    """No generators -- all monomials of degree \\leqslant d form the basis."""
    x, y = symbols("x y")
    bb = BorderBasis([x, y], [], degree=2)

    expected_basis = {(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)}
    assert set(bb.basis) == expected_basis


def test_reduce_in_ideal():
    """x^2 \\in <x^2, y^2> should reduce to 0."""
    x, y = symbols("x y")
    bb = BorderBasis([x, y], [x**2, y**2], degree=2)

    reduced = bb.reduce(x**2)
    assert reduced == 0


def test_reduce_basis_element():
    """xy \\notin <x^2, y^2> should reduce to xy (it's in the basis)."""
    x, y = symbols("x y")
    bb = BorderBasis([x, y], [x**2, y**2], degree=2)

    reduced = bb.reduce(x * y)
    # reduce() returns SymPy expression with float coeffs; use .equals() for structural comparison
    assert hasattr(reduced, "equals") and reduced.equals(x * y), f"Expected xy, got {reduced}"


def test_reduce_higher_power():
    """x^3 mod <x^2, y^2> should be 0 (x^3 = x\\cdotx^2 \\in I)."""
    x, y = symbols("x y")
    bb = BorderBasis([x, y], [x**2, y**2], degree=2)

    reduced = bb.reduce(x**3)
    assert reduced == 0


def test_reduce_circle_ideal():
    """x^3 mod <x^2+y^2-1> should be x (since x^2 \\equiv 1-y^2)."""
    x, y = symbols("x y")
    bb = BorderBasis([x, y], [x**2 + y**2 - 1], degree=2)

    reduced = bb.reduce(x**3)
    assert hasattr(reduced, "equals") and reduced.equals(x), f"Expected x, got {reduced}"


def test_reduce_xy_minus_one():
    """xy mod <xy-1> should be 1."""
    x, y = symbols("x y")
    bb = BorderBasis([x, y], [x * y - 1], degree=2)

    reduced = bb.reduce(x * y)
    assert float(reduced) == 1.0, f"Expected 1.0, got {reduced}"


def test_mult_table_xy_minus_one():
    """Multiplication table for xy mod <xy-1> should give coefficient 1 on basis element (0,0)."""
    x, y = symbols("x y")
    bb = BorderBasis([x, y], [x * y - 1], degree=2)

    # xy is a border element; its table should reduce to 1\\cdot(0,0)
    assert (1, 1) in bb.mult_tables
    coeffs = bb.mult_tables[(1, 1)]
    # The basis includes (0,0); check that one coefficient is ~1.0
    assert any(abs(c - 1.0) < 1e-10 for c in coeffs), f"Expected coeff ~1.0, got {coeffs}"


def test_univariate():
    """Univariate ideal <x^3 - 1> with degree=2."""
    x = symbols("x")
    bb = BorderBasis([x], [x**3 - 1], degree=2)

    # Basis should be {1, x, x^2} (degree \\leqslant 2, no reduction at this level)
    expected_basis = {(0,), (1,), (2,)}
    assert set(bb.basis) == expected_basis


if __name__ == "__main__":
    tests = [
        test_basis_ideal_x2_y2,
        test_basis_ideal_x3_y3,
        test_free_algebra,
        test_reduce_in_ideal,
        test_reduce_basis_element,
        test_reduce_higher_power,
        test_reduce_circle_ideal,
        test_reduce_xy_minus_one,
        test_mult_table_xy_minus_one,
        test_univariate,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"[OK] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
