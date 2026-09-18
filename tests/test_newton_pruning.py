"""Newton polytope pruning tests for SDPRelaxations."""

from sympy import symbols

from Irene.relaxations import SDPRelaxations


x, y = symbols("x y")


def _reduced_basis(relaxation, use_pruning):
    SDPRelaxations.NewtonPruning = use_pruning
    relaxation.ReducedBases = {}
    return relaxation.ReducedMonomialBase(relaxation.MmntOrd)


def _exponents_vec(relaxation, use_pruning):
    SDPRelaxations.NewtonPruning = use_pruning
    relaxation.ReducedBases = {}
    return relaxation.ExponentsVec(relaxation.MmntOrd)


def test_sparse_problem_pruned_basis_is_subset():
    rlx = SDPRelaxations([x, y])
    rlx.SetObjective(x**4 * y**4 - x**2 * y**2 - 1)
    rlx.MomentsOrd(1)
    rlx.RelaxationDeg()

    basis_full = _reduced_basis(rlx, use_pruning=False)
    basis_pruned = _reduced_basis(rlx, use_pruning=True)

    assert set(basis_pruned).issubset(set(basis_full))
    assert len(basis_pruned) <= len(basis_full)


def test_dense_problem_pruned_basis_is_subset():
    rlx = SDPRelaxations([x, y])
    rlx.SetObjective(x**4 + x**3 * y + x**2 * y**2 + x * y**3 + y**4 - 1)
    rlx.AddConstraint(x**2 + y**2 >= 1)
    rlx.MomentsOrd(1)
    rlx.RelaxationDeg()

    basis_full = _reduced_basis(rlx, use_pruning=False)
    basis_pruned = _reduced_basis(rlx, use_pruning=True)

    assert set(basis_pruned).issubset(set(basis_full))
    assert len(basis_pruned) <= len(basis_full)


def test_exponents_vec_does_not_grow_under_pruning():
    rlx = SDPRelaxations([x, y])
    rlx.SetObjective(x**4 - x * y + y**2)
    rlx.MomentsOrd(1)
    rlx.RelaxationDeg()

    exp_full = _exponents_vec(rlx, use_pruning=False)
    exp_pruned = _exponents_vec(rlx, use_pruning=True)

    assert len(exp_pruned) <= len(exp_full)


def test_pruning_keeps_required_sos_monomials():
    """Pruning must preserve the complete certified SOS basis."""
    rlx = SDPRelaxations([x, y])
    rlx.SetObjective(x**4 + y**4)
    rlx.MomentsOrd(2)
    rlx.RelaxationDeg()

    basis_full = set(_reduced_basis(rlx, use_pruning=False))
    basis_pruned = set(_reduced_basis(rlx, use_pruning=True))
    assert basis_pruned == basis_full


def test_degree_six_pruning_preserves_moment_basis():
    """Degree-six SOS problems must not lose Gram monomials under pruning."""
    rlx = SDPRelaxations([x, y, symbols('z')])
    rlx.SetObjective(x**4 * y**2 + x**2 * y**4 - 3*x**2*y**2 + 1)
    rlx.MomentsOrd(3)
    rlx.RelaxationDeg()

    basis_full = set(_reduced_basis(rlx, use_pruning=False))
    basis_pruned = set(_reduced_basis(rlx, use_pruning=True))
    assert basis_pruned == basis_full


def teardown_module(module):
    # Restore default global behavior for other tests.
    SDPRelaxations.NewtonPruning = False
