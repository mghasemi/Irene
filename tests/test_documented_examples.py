"""Regression tests pinning the examples documented in ``doc/``.

The repository docs quote concrete numbers (basis sizes, verbose reduction
diagnostics, SDP objective values).  These tests re-run the documented calls
and assert exactly those quoted values so CI catches drift between the
documentation and actual behavior:

- ``doc/newton_polytope.rst`` "Example Usage": hull-only Newton pruning of
  Motzkin at degree bound 6 gives a full basis of 28 reduced to 10
  (both for semigroup-algebra input and plain SymPy input).
- ``doc/relaxation_api.rst`` reduction-pipeline walkthrough: the P3
  configuration on Motzkin at order 3 prints the measured verbose line
  ``[SDPRlx] Newton pruning: 28 -> 28 (100.00% retained)`` and returns the
  same objective value as the unpruned baseline.

Every number asserted here was produced by running the corresponding code,
not taken from prose.
"""

import json
import math
from pathlib import Path

import yaml
from sympy import symbols

from benchmarks.p3_vs_baseline import build_motzkin
from benchmarks.run_gallery import build_problem
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.newton_polytope import prune_basis_from_polys, prune_basis_from_problem
from Irene.relaxation_api import RelaxationEngine
from Irene.relaxations import RelaxationConfig
from Irene.sosonc import SOSONCRelaxations

REPO_ROOT = Path(__file__).resolve().parents[1]


def _motzkin_ring_element():
    """Motzkin polynomial built through the documented construction path."""
    sg = CommutativeSemigroup(["x", "y"])
    sga = SemigroupAlgebra(sg)
    x, y = sga["x"], sga["y"]
    return x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2


# ---------------------------------------------------------------------------
# doc/newton_polytope.rst -- "Example Usage" (hull-only prune, 28 -> 10)
# ---------------------------------------------------------------------------

def test_newton_doc_example_semigroup_input():
    """The documented ring-element example prunes the degree-6 basis 28 -> 10."""
    motzkin = _motzkin_ring_element()
    pruner = prune_basis_from_polys([motzkin], num_vars=2, max_degree=6)

    assert pruner.full_basis_size == 28
    assert pruner.pruned_basis_size == 10


def test_newton_doc_example_sympy_input():
    """Plain SymPy input gives the identical counts (documented equivalence)."""
    x, y = symbols("x y")
    f = x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1
    pruner = prune_basis_from_polys([f], num_vars=2, max_degree=6)

    assert pruner.full_basis_size == 28
    assert pruner.pruned_basis_size == 10


def test_newton_doc_example_exponents():
    """The ten retained exponents are exactly the documented set."""
    x, y = symbols("x y")
    f = x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1
    pruner = prune_basis_from_polys([f], num_vars=2, max_degree=6)

    expected = {
        (0, 0), (1, 1), (1, 2), (2, 1), (2, 2),
        (2, 3), (2, 4), (3, 2), (3, 3), (4, 2),
    }
    assert set(map(tuple, pruner.compute_pruned_basis())) == expected


def test_newton_doc_example_problem_input():
    """prune_basis_from_problem accepts an OptimizationProblem directly."""
    prog = build_motzkin()
    pruner = prune_basis_from_problem(prog, max_degree=6)

    assert pruner.full_basis_size == 28
    assert pruner.pruned_basis_size == 10


# ---------------------------------------------------------------------------
# doc/relaxation_api.rst -- measured pipeline walkthrough (P3 config, order 3)
# ---------------------------------------------------------------------------

def test_relaxation_api_doc_walkthrough(capsys):
    """The documented P3-config run prints the measured line and matches baseline.

    Mirrors the configuration in doc/relaxation_api.rst: Motzkin at order 3,
    reduction_method="newton_polytope", monomial_pruning=True,
    sparsity_detection=True, verbose_reduction=True. The verbose output must
    contain the measured moment-basis diagnostic (28 -> 28, certified-safe
    retention).

    The objective value is asserted against an unpruned baseline run in the
    SAME environment rather than a hardcoded number: this order-3 Motzkin SDP
    is ill-conditioned (moment-matrix condition ~1e14), so the absolute
    optimum drifts by O(1) across solver/BLAS/python environments, while the
    structural claim -- that the reduction pipeline is a designed no-op on
    this dense instance -- holds in every environment.  The documented value
    (-526.08114071) was measured on the maintainer's machine (Python 3.11,
    Clarabel 0.11.1).
    """
    prog = build_motzkin()
    config = RelaxationConfig(
        reduction_method="newton_polytope",
        monomial_pruning=True,
        sparsity_detection=True,
        verbose_reduction=True,
    )

    engine = RelaxationEngine(prog, order=3, solver="clarabel", verbosity=0,
                              config=config)
    result = engine.solve("sos")

    captured = capsys.readouterr().out
    assert "[SDPRlx] Newton pruning: 28 -> 28 (100.00% retained)" in captured
    assert result.status == "optimal"

    # Same environment, reduction pipeline disabled: must return the identical
    # SDP value (fresh problem instance so no state leaks between engines).
    baseline = RelaxationEngine(
        build_motzkin(), order=3, solver="clarabel", verbosity=0,
        config=RelaxationConfig(sparsity_detection=True),
    ).solve("sos")
    assert baseline.status == "optimal"
    assert abs(result.value - baseline.value) <= 1e-6 * max(1.0, abs(baseline.value)), (
        f"P3 ({result.value}) deviates from unpruned baseline ({baseline.value}): "
        "the reduction pipeline must be a designed no-op on this dense instance"
    )


def test_benchmark_json_p3_matches_baseline():
    """benchmarks/results/p3_vs_baseline.json: P3 equals baseline in every cell.

    The docs state that on these instances the relaxed SDP is identical to the
    unpruned baseline at orders 1--3; pin that structural invariant on the
    committed benchmark artifact (values, not timings -- timings are
    machine-dependent).
    """
    path = REPO_ROOT / "benchmarks" / "results" / "p3_vs_baseline.json"
    data = json.loads(path.read_text())

    for problem, entry in data.items():
        assert {e["order"] for e in entry["orders"]} == {1, 2, 3}, problem
        for cell in entry["orders"]:
            base, p3 = cell["baseline"], cell["p3_optimized"]
            assert base["value"] is not None and p3["value"] is not None, (
                f"{problem} order {cell['order']}: a non-finite run was committed"
            )
            assert abs(base["value"] - p3["value"]) < 1e-6, (
                f"{problem} order {cell['order']}: P3 deviates from baseline "
                f"({p3['value']} vs {base['value']}) -- the reduction pipeline "
                "must be a designed no-op on these dense instances"
            )


# ---------------------------------------------------------------------------
# benchmarks/gallery.yaml -- CI smoke subset (JOSS reproducibility gate)
# ---------------------------------------------------------------------------

def test_gallery_smoke_subset():
    """Two small gallery problems run end-to-end at orders 1-2, solver=clarabel.

    Per-push smoke subset of the full catalog in ``benchmarks/gallery.yaml``
    (the heavy full-catalog sweep lives in the benchmark workflow step).
    Problems picked by catalog order: Motzkin (first separating example) and
    constrained_1d (smallest constrained problem, degree 2, exact at order 1).

    Pinned constants were measured on commit a507e3d in the repo ``.venv``
    (Python 3.11.15, cvxpy 1.9.2) with solver "clarabel" and match the
    committed CI artifacts ``benchmarks/results/gallery_20260808_070931Z.json``
    and ``gallery_20260808_201748Z.json`` (both solver=clarabel, tol=1e-4) to
    displayed precision; those artifacts cover motzkin sos r1-r3 and
    constrained_1d sos r1. The Motzkin cell is a weak lower bound by design:
    the catalog marks its SOS relaxations "infeasible_or_weak", and see
    ``test_relaxation_api_doc_walkthrough`` for the documented O(1)
    cross-environment drift of this ill-conditioned SDP.
    """
    with open(REPO_ROOT / "benchmarks" / "gallery.yaml") as f:
        by_id = {p["id"]: p for p in yaml.safe_load(f)["gallery"]}

    pinned = {
        ("motzkin", 1): -526.081140705874,
        ("motzkin", 2): -526.081140705874,
        ("constrained_1d", 1): 0.9999999877800618,
        ("constrained_1d", 2): 0.9999999968011417,
    }

    for pid, orders in (("motzkin", (1, 2)), ("constrained_1d", (1, 2))):
        prog = build_problem(by_id[pid])
        for r in orders:
            engine = SOSONCRelaxations(
                prog, verbosity=0, relaxation_order=r, solver="clarabel"
            )
            res = engine.globalMinSOS()

            expected = pinned[(pid, r)]
            assert res.status == "optimal", (
                f"{pid} order {r}: status={res.status!r}, "
                f"error_code={getattr(res, 'error_code', None)}"
            )
            assert res.val is not None and math.isfinite(res.val), (
                f"{pid} order {r}: non-finite bound {res.val!r}"
            )
            assert math.isclose(res.val, expected, rel_tol=1e-6), (
                f"{pid} order {r}: val={res.val}, pinned={expected} "
                "(measured on commit a507e3d, solver=clarabel)"
            )
