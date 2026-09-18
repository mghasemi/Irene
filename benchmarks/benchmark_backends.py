#!/usr/bin/env python3
"""
Comprehensive Irene vs IreneRewrite Backend Benchmark
=====================================================

Runs the SAME feature set through three configurations:

  --mode irene                Original Irene 1.2.5 (SymPy only)
  --mode irene_rewrite        IreneRewrite 2.0 (SymEngine primary + SymPy fallback)
  --mode irene_rewrite_sympy  IreneRewrite 2.0 forced to pure SymPy backend
                              (IRENE_SYMBOLIC_BACKEND=sympy)

Feature sections covered:
  sos_sonc_sosonc  — SDP/SONC/SOS+SONC relaxations on 4 gallery problems
  gp               — geometric programming relaxation (GPExample problem)
  dsdp_mean        — DSDP mean-polynomial relaxation (Choi-Lam M_{1,0})
  dsdp_kkt         — differential KKT relaxation (small ADE problem)
  ade_relations    — build_ade_relations() derivative-symbol construction
  border_basis     — BorderBasis on test ideals
  sparsity         — correlative sparsity detection
  newton_polytope  — Newton polytope monomial pruning
  symbolic_micro   — expand / Poly / groebner / Matrix micro-benchmarks

Each section records elapsed wall time and a result summary so the three modes
can be compared apples-to-apples. Solver stdout is suppressed.

Usage:
  /home/mehdi/Code/Python/Irene/.venv/bin/python3 \\      # original
      benchmarks/benchmark_backends.py --mode irene --output benchmarks/results/backend_irene.json
  /home/mehdi/Code/Python/IreneRewrite/.venv/bin/python3 \\  # rewrite symengine
      benchmarks/benchmark_backends.py --mode irene_rewrite --output benchmarks/results/backend_rewrite_se.json
  /home/mehdi/Code/Python/IreneRewrite/.venv/bin/python3 \\  # rewrite sympy
      benchmarks/benchmark_backends.py --mode irene_rewrite_sympy --output benchmarks/results/backend_rewrite_sp.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODE = None


@contextmanager
def suppress_stdout():
    """Suppress solver diagnostics that corrupt JSON/console output."""
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        devnull.close()


def setup_mode(mode: str) -> str:
    """Set sys.path (and env var for sympy mode) and return package root."""
    global MODE
    MODE = mode
    if mode == "irene":
        root = "/home/mehdi/Code/Python/Irene"
        sys.path.insert(0, root)
    elif mode in ("irene_rewrite", "irene_rewrite_sympy"):
        root = "/home/mehdi/Code/Python/IreneRewrite"
        sys.path.insert(0, root)
        if mode == "irene_rewrite_sympy":
            os.environ["IRENE_SYMBOLIC_BACKEND"] = "sympy"
        else:
            os.environ["IRENE_SYMBOLIC_BACKEND"] = "symengine"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    os.chdir(root)
    return root


def timed(fn):
    """Decorator capturing elapsed time."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            status = "ok"
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            status = "error"
        elapsed = round(time.perf_counter() - t0, 4)
        if isinstance(result, dict):
            result["elapsed_s"] = elapsed
            result["status"] = status
        else:
            result = {"value": result, "elapsed_s": elapsed, "status": status}
        return result
    return wrapper


# =============================================================================
# 1. SOS / SONC / SOS+SONC relaxations
# =============================================================================

RELAX_PROBLEMS = [
    {
        "id": "quartic_1d",
        "variables": ["x"],
        "objective": "x**4 - x**2",
        "constraints": [],
        "true_min": -0.25,
        "orders": [2],
    },
    {
        "id": "motzkin",
        "variables": ["x", "y"],
        "objective": "x**4*y**2 + x**2*y**4 + 1 - 3*x**2*y**2",
        "constraints": [],
        "true_min": 0.0,
        "orders": [1],
    },
    {
        "id": "sphere_4",
        "variables": ["x", "y"],
        "objective": "x**4 + y**4",
        "constraints": [("x**2 + y**2 - 1", "eq")],
        "true_min": 0.5,
        "orders": [2],
    },
    {
        "id": "schick",
        "variables": ["x", "y"],
        "objective": "0.5*(1 + 2*x*y + x**2*y)**2 + x**4*y**2 + x**2*y**4 + 1 - 3*x**2*y**2",
        "constraints": [],
        "true_min": 0.0,
        "orders": [1],
    },
]


def build_relax_problem(pdef):
    from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
    variables = pdef["variables"]
    sg = CommutativeSemigroup(variables)
    sga = SemigroupAlgebra(sg)
    sym_dict = {v: sga[v] for v in variables}
    objective = eval(pdef["objective"], {"__builtins__": {}}, sym_dict)
    prog = type(sga)(sg) if False else __import__("Irene.program", fromlist=["OptimizationProblem"]).OptimizationProblem(sga)
    prog.set_objective(objective)
    for c_expr, c_type in pdef.get("constraints", []):
        cexpr = eval(c_expr, {"__builtins__": {}}, sym_dict)
        prog.add_constraints([cexpr])
    return prog


@timed
def section_relaxations():
    from Irene.sosonc import SOSONCRelaxations
    out = {}
    for pdef in RELAX_PROBLEMS:
        pid = pdef["id"]
        prog = build_relax_problem(pdef)
        entry = {"true_min": pdef["true_min"], "methods": {}}
        for r in pdef["orders"]:
            engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=r)
            for method, call in [
                ("sos", lambda: engine.globalMinSOS()),
                ("sonc", lambda: engine.globalMinSONC()),
                ("sosonc", lambda: engine.globalMinSOSPSONC(first="sos")),
            ]:
                t0 = time.perf_counter()
                try:
                    with suppress_stdout():
                        res = call()
                    val = float(res.val) if hasattr(res, "val") else float(res)
                    status = getattr(res, "status", "unknown")
                    entry["methods"][f"{method}_r{r}"] = {
                        "value": round(val, 8),
                        "status": status,
                        "elapsed_s": round(time.perf_counter() - t0, 4),
                    }
                except Exception as exc:
                    entry["methods"][f"{method}_r{r}"] = {
                        "value": None,
                        "status": "exception",
                        "error": str(exc)[:200],
                        "elapsed_s": round(time.perf_counter() - t0, 4),
                    }
        out[pid] = entry
    return out


# =============================================================================
# 2. GP relaxation (GPExample problem)
# =============================================================================

@timed
def section_gp():
    from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
    from Irene.program import OptimizationProblem
    from Irene.geometric import GPRelaxations
    import numpy as np

    S = CommutativeSemigroup(["x", "y", "z"])
    SA = SemigroupAlgebra(S)
    x, y, z = SA["x"], SA["y"], SA["z"]
    optim = OptimizationProblem(SA)
    f = -y - 2 * x**2
    g1 = y - x**4 * y + y**5 - x**6 - y**6
    g2 = y - 5 * x**2 + x**4 * y - x**6 - y**6
    optim.set_objective(f)
    optim.add_constraints([g1, g2])
    gp = GPRelaxations(optim)
    gp.H = gp.auto_transform_matrix()
    gp.H = np.array([[1, 0], [-1, 1]])
    t0 = time.perf_counter()
    with suppress_stdout():
        sol = gp.solve()
    return {
        "objective": str(f),
        "num_constraints": 2,
        "solve_output": str(sol)[:120],
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }


# =============================================================================
# 3. DSDP mean relaxation (Choi-Lam)
# =============================================================================

@timed
def section_dsdp_mean():
    from sympy import symbols
    from Irene.dsdp import DSDPMeanRelaxation
    x, y, z, w = symbols("x y z w")
    dsdp = DSDPMeanRelaxation(
        gens=[x, y, z, w], weights=[1.0, 1.0, 1.0, 1.0], q=1, p=0, verbosity=0)
    dsdp.SetObjective(x**4 + y**4 + z**4 + w**4 - 4 * x * y * z * w)
    with suppress_stdout():
        lb = dsdp.solve(order=2)
    return {"problem": "choi_lam_M10", "lower_bound": float(lb)}


# =============================================================================
# 4. DSDP KKT relaxation
# =============================================================================

@timed
def section_dsdp_kkt():
    from sympy import symbols
    from Irene.dsdp import DSDPKKTRelaxation
    x, y = symbols("x y")
    dsdp = DSDPKKTRelaxation(
        gens=[x, y], relations=[], diff_map={x: 1, y: -y}, verbosity=0)
    dsdp.SetObjective(x**2 + y**2)
    dsdp.AddConstraint(1 - x**2 - y**2)
    with suppress_stdout():
        lb = dsdp.solve_kkt(order=1)
    return {"problem": "exp_decay_kkt", "lower_bound": float(lb)}


# =============================================================================
# 5. ADE relations (build_ade_relations)
# =============================================================================

@timed
def section_ade_relations():
    from sympy import symbols
    from Irene.dsdp import DSDPRelaxations
    x, u = symbols("x u")
    dsdp = DSDPRelaxations([x, u], relations=[])
    dm = {x: 1, u: 1 + u**2}
    t0 = time.perf_counter()
    dsyms, rels, gens = dsdp.build_ade_relations(dm)
    build_ms = (time.perf_counter() - t0) * 1000
    # Multi-derivation variant
    y, L, v = symbols("y L v")
    dsdp2 = DSDPRelaxations([y, L, v], relations=[])
    dsyms2, rels2, gens2 = dsdp2.build_ade_relations({y: 1, L: v, v: -v**2}, wrt="y")
    return {
        "single_deriv_symbols": list(map(str, dsyms.values())),
        "multi_deriv_symbols": list(map(str, dsyms2.values())),
        "relations": [str(r) for r in rels],
        "gens": [str(g) for g in gens],
        "build_ms": round(build_ms, 3),
    }


# =============================================================================
# 6. Border basis
# =============================================================================

@timed
def section_border_basis():
    from sympy import symbols
    x, y = symbols("x y")
    ideals = [
        ("circle_xy", [x**2 + y**2 - 1, x * y - 1]),
        ("monomial", [x**2, y**2]),
    ]
    out = {}
    for name, gens in ideals:
        t0 = time.perf_counter()
        try:
            if MODE == "irene":
                from Irene.border_basis import BorderBasis as OrigBB
                bb = OrigBB(polynomials=gens, variables=[x, y], max_degree=4)
                bb.compute()
                dim = bb.dimension()
                nf = bb.normal_form(x * y)
                out[name] = {
                    "api": "original(BorderBasis.polynomials/max_degree)",
                    "dimension": dim,
                    "normal_form_xy": str(dict(nf))[:80],
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                }
            else:
                from Irene.border_basis import BorderBasis as NewBB
                bb = NewBB(variables=[x, y], generators=gens, degree=4)
                reduced = bb.reduce(x * y)
                cond = bb.conditioning_diagnostic()
                out[name] = {
                    "api": "rewrite(BorderBasis.variables/generators/degree)",
                    "reduced_xy": str(reduced)[:80],
                    "conditioning": cond,
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                }
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}",
                         "elapsed_s": round(time.perf_counter() - t0, 4)}
    return out


# =============================================================================
# 7. Correlative sparsity
# =============================================================================

@timed
def section_sparsity():
    from sympy import symbols
    x, y, z = symbols("x y z")
    polys = [x**2 + y**2 - 1, z**2 + z]
    t0 = time.perf_counter()
    if MODE == "irene":
        from Irene.correlative_sparsity import analyze_correlative_sparsity
        cs = analyze_correlative_sparsity(polys, variables=[x, y, z])
        return {
            "api": "original(analyze_correlative_sparsity)",
            "is_sparse": cs.is_sparse(),
            "reduction_ratio": cs.total_reduction_ratio(degree=2),
            "summary": cs.summary(),
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }
    else:
        from Irene.sparsity import detect_sparsity_from_polys
        cs = detect_sparsity_from_polys(polys, num_vars=3)
        # NOTE: rewrite CorrelativeSparsity exposes `is_sparse` as a bool
        # attribute (populated by detect_*), not a method like the original.
        is_sparse = cs.is_sparse if isinstance(cs.is_sparse, bool) else cs.is_sparse()
        return {
            "api": "rewrite(detect_sparsity_from_polys)",
            "is_sparse": is_sparse,
            "reduction_factor": cs.reduction_factor(deg=2),
            "summary": cs.summary(),
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }


# =============================================================================
# 8. Newton polytope pruning
# =============================================================================

@timed
def section_newton():
    from sympy import symbols
    x, y = symbols("x y")
    polys = [x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2]
    t0 = time.perf_counter()
    if MODE == "irene":
        from Irene.newton_polytope import prune_by_newton_polytope
        pruner = prune_by_newton_polytope(polys, variables=[x, y], relaxation_degree=2)
        pruner.prune()
        return {
            "api": "original(prune_by_newton_polytope)",
            "total_reduction_ratio": pruner.total_reduction_ratio(),
            "summary": pruner.summary(),
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }
    else:
        from Irene.newton_polytope import prune_basis_from_polys
        pruner = prune_basis_from_polys(polys, num_vars=2, max_degree=4)
        info = pruner.moment_matrix_dimension_reduction()
        return {
            "api": "rewrite(prune_basis_from_polys)",
            "reduction_info": info,
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }


# =============================================================================
# 10. Quotient-basis option: Groebner vs BorderBasis reduction engine
# =============================================================================

QUOTIENT_PROBLEMS = [
    {
        "id": "quartic_1d",
        "variables": ["x"],
        "relations": [],
        "objective": "x**4 - x**2",
        "order": 2,
        "true_min": -0.25,
        "note": "no relations -> border mode falls back to full monomial basis",
    },
    {
        "id": "circle_relations",
        "variables": ["x", "y"],
        "relations": ["x**2 + y**2 - 1"],
        "objective": "x**2 + y**2",
        "order": 1,
        "true_min": 1.0,
        "note": "quotient by <x^2+y^2-1>; standard monomials {1,x,y,xy,y^2}",
    },
]


@timed
def section_quotient_basis():
    if MODE == "irene":
        return {"note": "original Irene has no border-basis option (Groebner only)"}
    from sympy import symbols
    from Irene.relaxations import SDPRelaxations, RelaxationConfig

    out = {}
    for pdef in QUOTIENT_PROBLEMS:
        pid = pdef["id"]
        gens = symbols(pdef["variables"])
        relations = [eval(r, {"__builtins__": {}}, dict(zip(pdef["variables"], gens)))
                     for r in pdef["relations"]]
        obj = eval(pdef["objective"], {"__builtins__": {}},
                   dict(zip(pdef["variables"], gens)))
        out[pid] = {"true_min": pdef["true_min"], "note": pdef["note"], "modes": {}}
        for qb in ("groebner", "border"):
            t0 = time.perf_counter()
            try:
                rlx = SDPRelaxations(list(gens), relations=relations,
                                     config=RelaxationConfig(quotient_basis=qb))
                rlx.SetObjective(obj)
                rlx.MomentsOrd(pdef["order"])
                with suppress_stdout():
                    rlx.InitSDP()
                    lb = rlx.Minimize()
                basis_size = len(rlx.ReducedMonomialBase(2 * pdef["order"]))
                out[pid]["modes"][qb] = {
                    "lower_bound": round(float(lb), 8),
                    "basis_size": basis_size,
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                }
            except Exception as exc:
                out[pid]["modes"][qb] = {
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                }
    return out


# =============================================================================
# 9. Symbolic engine micro-benchmarks
# =============================================================================

def _micro_impl():
    """Return (symbols, ops) where ops is dict of name -> callable."""
    if MODE == "irene":
        import sympy as sp
        x, y = sp.symbols("x y")
        ops = {
            "expand_deg8": lambda: sp.expand((x + y) ** 8),
            "poly_deg6": lambda: sp.Poly((x + y) ** 6, x, y),
            "groebner": lambda: sp.groebner([x**2 + y**2 - 1, x - y], x, y),
            "matrix_mul": lambda: (sp.Matrix([[x**2, 1], [0, x]]) * sp.Matrix([[x, y], [1, 0]])),
            "zeros_50": lambda: sp.zeros(50, 50),
        }
        return x, y, ops, "sympy-direct"
    else:
        from Irene.symbolic_engine import engine
        x, y = engine.symbols("x y")
        ops = {
            "expand_deg8": lambda: engine.expand((x + y) ** 8),
            "poly_deg6": lambda: engine.Poly((x + y) ** 6, x, y),
            "groebner": lambda: engine.groebner([x**2 + y**2 - 1, x - y], x, y),
            "matrix_mul": lambda: engine.Matrix([[x**2, 1], [0, x]]) * engine.Matrix([[x, y], [1, 0]]),
            "zeros_50": lambda: engine.zeros(50, 50),
        }
        return x, y, ops, f"engine-{engine.get_backend()}"


@timed
def section_symbolic_micro():
    x, y, ops, backend_label = _micro_impl()
    out = {"backend": backend_label, "ops": {}}
    # warm-up
    for fn in ops.values():
        try:
            fn()
        except Exception:
            pass
    for name, fn in ops.items():
        t0 = time.perf_counter()
        try:
            fn()
            out["ops"][name] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000, 3)}
        except Exception as exc:
            out["ops"][name] = {"error": str(exc)[:150],
                                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3)}
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Irene vs IreneRewrite backend benchmark")
    parser.add_argument("--mode", required=True,
                        choices=["irene", "irene_rewrite", "irene_rewrite_sympy"])
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--sections", default=None,
                        help="Comma-separated subset of sections (default: all)")
    args = parser.parse_args()

    # Resolve output path to absolute BEFORE setup_mode() chdirs the process
    output_path = None
    if args.output:
        output_path = os.path.abspath(args.output)

    root = setup_mode(args.mode)

    sections = {
        "sos_sonc_sosonc": section_relaxations,
        "gp": section_gp,
        "dsdp_mean": section_dsdp_mean,
        "dsdp_kkt": section_dsdp_kkt,
        "ade_relations": section_ade_relations,
        "border_basis": section_border_basis,
        "sparsity": section_sparsity,
        "newton_polytope": section_newton,
        "symbolic_micro": section_symbolic_micro,
        "quotient_basis": section_quotient_basis,
    }

    if args.sections:
        wanted = {s.strip() for s in args.sections.split(",")}
        sections = {k: v for k, v in sections.items() if k in wanted}

    # Report backend selection (informative for rewrite modes)
    try:
        from Irene.symbolic_engine import engine as _engine
        backend_report = _engine.get_backend()
    except ImportError:
        backend_report = "sympy-direct (no symbolic_engine module)"

    output = {
        "mode": args.mode,
        "package_root": root,
        "backend": backend_report,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sections": {},
    }

    for name, fn in sections.items():
        print(f"[{name}] running...", file=sys.stderr)
        output["sections"][name] = fn()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=1, default=str)
        print(f"wrote {output_path}", file=sys.stderr)
    else:
        print(json.dumps(output, indent=1, default=str))


if __name__ == "__main__":
    main()
