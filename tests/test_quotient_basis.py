"""Tests for the quotient-basis reduction option (Groebner vs BorderBasis).

The user-selectable ``RelaxationConfig.quotient_basis`` option chooses the
quotient-ring reduction engine used by ``ReduceExp`` and
``ReducedMonomialBase``:

- ``'groebner'`` (default): classical SymPy Groebner-basis reduction — the
  behavior of original Irene.
- ``'border'``: IreneRewrite's BorderBasis quotient-algebra reduction using
  numerically computed multiplication tables.

Also covered: the ``IRENE_QUOTIENT_BASIS`` environment variable.
"""
import os
import subprocess
import sys

import pytest
from sympy import symbols

from Irene.relaxations import RelaxationConfig, SDPRelaxations, _default_config


def _sdp_with_relation(quotient_basis="groebner"):
    """SDPRelaxations on min x^2 + y^2 s.t. relation x^2 + y^2 - 1 = 0."""
    x, y = symbols("x y")
    cfg = RelaxationConfig(quotient_basis=quotient_basis)
    rlx = SDPRelaxations([x, y], relations=[x**2 + y**2 - 1], config=cfg)
    return rlx, x, y


class TestConfigValidation:
    def test_default_is_groebner(self):
        assert RelaxationConfig().quotient_basis == "groebner"

    def test_accepts_border(self):
        assert RelaxationConfig(quotient_basis="border").quotient_basis == "border"

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            RelaxationConfig(quotient_basis="magic")

    def test_default_config_honours_env(self):
        os.environ["IRENE_QUOTIENT_BASIS"] = "border"
        try:
            assert _default_config().quotient_basis == "border"
        finally:
            os.environ.pop("IRENE_QUOTIENT_BASIS", None)
        assert _default_config().quotient_basis == "groebner"


class TestReductionEquivalence:
    def test_reduce_expression_equivalent(self):
        """x^2+y^2 reduces to 1 modulo <x^2+y^2-1> in both modes."""
        for qb in ("groebner", "border"):
            rlx, x, y = _sdp_with_relation(qb)
            rlx.SetObjective(x**2 + y**2)
            red = rlx.RedObjective
            if qb == "groebner":
                assert abs(float(red) - 1.0) < 1e-12
            else:
                # border basis returns float coefficients
                assert abs(float(red) - 1.0) < 1e-8

    def test_border_reduces_higher_degree_ideal_multiples(self):
        """Border reduction preserves the ideal for terms above its border."""
        x = symbols("x")
        rlx = SDPRelaxations(
            [x],
            relations=[x**2 - 2],
            config=RelaxationConfig(
                quotient_basis="border",
                border_basis_degree=2,
            ),
        )

        assert rlx.ReduceExp(x**4 - 4) == 0
        assert rlx.ReduceExp(x**5 - 4 * x) == 0

    def test_relation_free_problem_same_basis(self):
        """No relations: both modes give the full monomial basis."""
        x, y = symbols("x y")
        for qb in ("groebner", "border"):
            rlx = SDPRelaxations([x, y], relations=[],
                                 config=RelaxationConfig(quotient_basis=qb))
            rlx.SetObjective(x + y)
            basis = rlx.ReducedMonomialBase(2)
            # basis lives in AuxSym space (X1, X2)
            X1, X2 = rlx.AuxSyms
            assert len(basis) == 6  # {1, X1, X2, X1^2, X1X2, X2^2}
            assert all(m in (1, X1, X2, X1**2, X1 * X2, X2**2) for m in basis)

    def test_quotient_basis_matches_reduction_method(self):
        """quotient_basis='border' dispatches ReducedMonomialBase to border."""
        x, y = symbols("x y")
        rlx = SDPRelaxations([x, y], relations=[x**2 + y**2 - 1],
                             config=RelaxationConfig(quotient_basis="border"))
        rlx.SetObjective(x**2 + y**2)
        basis = rlx.ReducedMonomialBase(2)
        # quotient by <x^2+y^2-1> (lex, LM=x^2): standard monomials of
        # degree <= 2 are {1, x, y, xy, y^2} -- 5 elements
        assert len(basis) == 5
        assert basis.count(1) == 1  # constant term not duplicated

    def test_border_falls_back_for_positive_dimensional_ideal(self):
        """Invalid truncated border bases must fall back to Groebner reduction."""
        x, y, u, v, dvx, dvy, dvu = symbols("x y u v dvx dvy dvu")
        relations = [
            u**2 - (x * y + 1),
            v * y**2 * dvx - (y - x * y * v * dvy),
            v * x**2 * dvy - (x - x * y * v * dvx),
            2 * u * dvu - (y * dvx + x * dvy),
        ]
        rlx = SDPRelaxations(
            [x, y, u, v, dvx, dvy, dvu],
            relations=relations,
            config=RelaxationConfig(
                reduction_method="border_basis",
                quotient_basis="border",
                border_basis_degree=2,
            ),
        )

        assert rlx._get_border_basis(2) is None
        assert rlx.ReduceExp(relations[1]) == 0


class TestEndToEnd:
    def test_sos_bound_relation_problem(self):
        """min x^2+y^2 s.t. x^2+y^2=1: true min 1, both modes give ~1."""
        for qb in ("groebner", "border"):
            rlx, x, y = _sdp_with_relation(qb)
            rlx.SetObjective(x**2 + y**2)
            rlx.MomentsOrd(1)
            rlx.InitSDP()
            lb = rlx.Minimize()
            assert abs(float(lb) - 1.0) < 1e-3, f"{qb}: lb={lb}"

    def test_sos_bound_unconstrained_quartic(self):
        """quartic_1d without relations: -1/4 at order 2 in both modes."""
        x = symbols("x")
        for qb in ("groebner", "border"):
            rlx = SDPRelaxations([x], relations=[],
                                 config=RelaxationConfig(quotient_basis=qb))
            rlx.SetObjective(x**4 - x**2)
            rlx.MomentsOrd(2)
            rlx.InitSDP()
            lb = rlx.Minimize()
            assert abs(float(lb) + 0.25) < 1e-3, f"{qb}: lb={lb}"


class TestEnvVarIntegration:
    def test_env_var_selects_border(self):
        code = (
            "import os\n"
            "from Irene.relaxations import SDPRelaxations, RelaxationConfig\n"
            "from sympy import symbols\n"
            "x = symbols('x')\n"
            "rlx = SDPRelaxations([x], relations=[])\n"
            "print(rlx.config.quotient_basis)\n"
        )
        env = dict(os.environ, IRENE_QUOTIENT_BASIS="border")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env,
            cwd="/home/mehdi/Code/Python/IreneRewrite",
        )
        assert out.returncode == 0, out.stderr
        assert "border" in out.stdout.strip()

    def test_env_var_invalid_falls_back(self):
        code = (
            "from Irene.relaxations import SDPRelaxations\n"
            "from sympy import symbols\n"
            "x = symbols('x')\n"
            "rlx = SDPRelaxations([x], relations=[])\n"
            "print(rlx.config.quotient_basis)\n"
        )
        env = dict(os.environ, IRENE_QUOTIENT_BASIS="bogus")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env,
            cwd="/home/mehdi/Code/Python/IreneRewrite",
        )
        assert out.returncode == 0, out.stderr
        assert "groebner" in out.stdout.strip()
