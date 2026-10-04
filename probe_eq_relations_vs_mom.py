"""Probe: equality constraint via relations= vs Mom(g)==0 moment constraint.

Model: min x + y  s.t.  x*y = 1   (true optimum = 2 at (1,1), AM-GM).

The relation route quotients the algebra by <xy-1> (pointwise identity).
The Mom route imposes only the *moment* condition E[xy] = 1, which is
satisfied by measures off the hyperbola, e.g.
    (1/3) d_{(t,1/t)} + (2/3) d_{(-2/t,-t/2)}  ->  E[xy] = 1, E[x+y] = -1/t,
so as t -> 0+ the moment relaxation is driven far below the true optimum 2.
"""
import warnings
from sympy import symbols

from Irene.relaxations import SDPRelaxations, Mom

x, y = symbols("x y")


def run(tag, use_relations, use_mom, order):
    try:
        rlx = (SDPRelaxations([x, y], relations=[x * y - 1]) if use_relations
               else SDPRelaxations([x, y]))
        rlx.SetObjective(x + y)
        if use_mom:
            rlx.MomentConstraint(Mom(x * y - 1) == 0)
        rlx.MomentsOrd(order)
        rlx.InitSDP()
        rlx.Minimize()
        out = rlx.Info.get("min")
        basis = rlx.ReducedMonomialBase(2 * order)
        print(f"{tag:36s} f_min={out!r:24s} status={rlx.Info.get('status')!r:12s} "
              f"basis({2*order})={len(basis):3d} nLoc={len(rlx.Constraints)} "
              f"nMom={len(rlx.MomConst)}")
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"{tag:36s} EXCEPTION {type(exc).__name__}: {exc}")
        return None


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("TRUE optimum = 2.0\n")
    for order in (1, 2):
        print(f"--- moment order {order} ---")
        run("A: relations=[xy-1]", True, False, order)
        run("B: MomentConstraint(Mom(xy-1)==0)", False, True, order)
        run("C: no constraint at all", False, False, order)