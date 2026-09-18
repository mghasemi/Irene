#!/usr/bin/env python3
"""
Irene vs IreneRewrite Cross-Version Comparison Benchmark
========================================================

Runs the same set of polynomial optimization problems through both Irene
and IreneRewrite, comparing relaxation bounds (SOS, SONC, SOS+SONC),
wall-clock timing, and numerical results against Scipy optimization.

Usage:
  # Run with IreneRewrite:
  /home/mehdi/Code/Python/IreneRewrite/.venv/bin/python3 benchmarks/compare_irene_vs_rewrite.py --mode irene_rewrite

  # Run with original Irene:
  /home/mehdi/Code/Python/Irene/.venv/bin/python3 benchmarks/compare_irene_vs_rewrite.py --mode irene

Output: JSON on stdout with structured timing and numerical results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

# ── Path setup ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODE = None  # set by main()


def setup_paths(mode: str):
    """Add the correct Irene package to sys.path based on mode."""
    if mode == "irene_rewrite":
        irene_root = "/home/mehdi/Code/Python/IreneRewrite"
    elif mode == "irene":
        irene_root = "/home/mehdi/Code/Python/Irene"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    sys.path.insert(0, irene_root)
    os.chdir(irene_root)
    return irene_root


# ═══════════════════════════════════════════════════════════════════
# Problem definitions — shared between both versions
# Each problem: (id, name, variables, degree, objective_str, constraints, true_min)
# ═══════════════════════════════════════════════════════════════════

PROBLEMS = [
    {
        "id": "quad_1d",
        "name": "1D Quadratic",
        "variables": ["x"],
        "degree": 2,
        "objective": "x**2",
        "constraints": [],
        "true_min": 0.0,
        "expected_sos_order": 1,
        "description": "x^2 — trivially SOS",
    },
    {
        "id": "quartic_1d",
        "name": "1D Quartic",
        "variables": ["x"],
        "degree": 4,
        "objective": "x**4 - x**2",
        "constraints": [],
        "true_min": -0.25,
        "expected_sos_order": 2,
        "description": "x^4 - x^2, min = -1/4 at x=±1/√2",
    },
    {
        "id": "motzkin",
        "name": "Motzkin",
        "variables": ["x", "y"],
        "degree": 6,
        "objective": "x**4*y**2 + x**2*y**4 + 1 - 3*x**2*y**2",
        "constraints": [],
        "true_min": 0.0,
        "expected_sos_order": None,  # SOS fails — not SOS
        "description": "Nonnegative, not SOS. SONC at order 3 certifies nonnegativity.",
    },
    {
        "id": "choi_lam",
        "name": "Choi-Lam",
        "variables": ["x", "y"],
        "degree": 6,
        "objective": "x**4*y**2 + x**2*y**4 + x**2*y**2*(x**2 + y**2 - 1)",
        "constraints": [],
        "true_min": 0.0,
        "expected_sos_order": None,  # SOS fails
        "description": "Nonnegative, not SOS. SONC at order 3 certifies nonnegativity.",
    },
    {
        "id": "robinson",
        "name": "Robinson",
        "variables": ["x", "y"],
        "degree": 6,
        "objective": "x**4*y**2 + x**2*y**4 + x**4 + y**4 - x**2 - y**2",
        "constraints": [],
        "true_min": 0.0,
        "expected_sos_order": None,
        "description": "Nonnegative, not SOS. Robinson polynomial.",
    },
    {
        "id": "constrained_1d",
        "name": "1D Constrained",
        "variables": ["x", "y"],
        "degree": 2,
        "objective": "x**2 + y**2",
        "constraints": [("x**2 + y**2 - 1", "eq")],
        "true_min": 1.0,
        "expected_sos_order": 1,
        "description": "min x^2+y^2 s.t. x^2+y^2=1. Min = 1.",
    },
    {
        "id": "sphere_4",
        "name": "Sphere Degree-4",
        "variables": ["x", "y"],
        "degree": 4,
        "objective": "x**4 + y**4",
        "constraints": [("x**2 + y**2 - 1", "eq")],
        "true_min": 0.5,
        "expected_sos_order": 2,
        "description": "min x^4+y^4 s.t. x^2+y^2=1. Min = 1/2.",
    },
    {
        "id": "schick",
        "name": "Schick SOS+SONC",
        "variables": ["x", "y"],
        "degree": 6,
        "objective": "0.5*(1 + 2*x*y + x**2*y)**2 + x**4*y**2 + x**2*y**4 + 1 - 3*x**2*y**2",
        "constraints": [],
        "true_min": 0.0,
        "expected_sos_order": None,  # SOS fails, SOS+SONC works
        "description": "Schick separating example: SOS+SONC certifies nonnegativity.",
    },
    {
        "id": "dense_bivar_8",
        "name": "Dense Bivariate Deg-8",
        "variables": ["x", "y"],
        "degree": 8,
        "objective": "(x + y)**8",
        "constraints": [],
        "true_min": 0.0,
        "expected_sos_order": 4,
        "description": "(x+y)^8 — even power, trivially nonnegative. Stress test.",
    },
    {
        "id": "sparse_trinomial",
        "name": "Sparse Trinomial",
        "variables": ["x", "y"],
        "degree": 6,
        "objective": "x**6 + y**6 + 1 - 3*x**2*y**2",
        "constraints": [],
        "true_min": -0.5,  # approximate
        "expected_sos_order": 3,
        "description": "x^6+y^6+1-3x^2y^2 — tests Newton polytope pruning.",
    },
]


# ═══════════════════════════════════════════════════════════════════
# Core benchmark runner
# ═══════════════════════════════════════════════════════════════════


def safe_float(v) -> float | None:
    """Convert to float, returning None for inf/nan/non-convertible."""
    try:
        fv = float(v)
        if math.isinf(fv) or math.isnan(fv):
            return None
        return round(fv, 10)
    except (TypeError, ValueError):
        return None


# ── Suppress solver/telemetry stdout noise ──
def _suppress_stdout():
    """Context manager to suppress stdout during solver calls."""
    import os as _os, sys as _sys
    return open(_os.devnull, 'w')

# ── Actual suppress helper ──
class _SuppressStdout:
    def __enter__(self):
        import sys as _sys
        self._old = _sys.stdout
        _sys.stdout = open('/dev/null', 'w')
        return self
    def __exit__(self, *args):
        import sys as _sys
        _sys.stdout.close()
        _sys.stdout = self._old


def build_problem(prob_def: dict, sga):
    """Build an OptimizationProblem from a problem definition dict."""
    from Irene.program import OptimizationProblem

    variables = prob_def["variables"]
    sym_dict = {v: sga[v] for v in variables}

    obj_expr = eval(prob_def["objective"], {"__builtins__": {}}, sym_dict)
    prog = OptimizationProblem(sga)
    prog.set_objective(obj_expr)

    for c_expr, c_type in prob_def.get("constraints", []):
        c_parsed = eval(c_expr, {"__builtins__": {}}, sym_dict)
        # Note: Irene API only accepts inequality constraints natively.
        # Equality constraints g(x)=0 treated as single inequality g(x)<=0.
        # This is a known limitation; for sphere problems we rely on SOS.
        prog.add_constraints([c_parsed])

    return prog


def run_irene_rewrite(prob_def: dict, prog):
    """Run using IreneRewrite's RelaxationEngine unified API."""
    from Irene.relaxation_api import RelaxationEngine

    results = {"sos": {}, "sonc": {}, "sosonc": {}}
    degree = prob_def["degree"]
    max_order = max(1, degree // 2)

    for r in range(1, max_order + 1):
        # SOS
        t0 = time.perf_counter()
        try:
            engine = RelaxationEngine(prog, order=r, verbosity=0)
            res = engine.solve("sos")
            elapsed = time.perf_counter() - t0
            results["sos"][f"r{r}"] = {
                "value": safe_float(res.value),
                "status": res.status,
                "error_code": res.error_code,
                "elapsed_s": round(elapsed, 4),
                "runtime_s": round(res.runtime, 4) if res.runtime else None,
                "init_time_s": round(res.init_time, 4) if res.init_time else None,
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results["sos"][f"r{r}"] = {
                "value": None,
                "status": "exception",
                "error_code": -1,
                "elapsed_s": round(elapsed, 4),
                "error": str(e)[:200],
            }

        # SONC
        t0 = time.perf_counter()
        try:
            engine = RelaxationEngine(prog, order=r, verbosity=0)
            res = engine.solve("sonc")
            elapsed = time.perf_counter() - t0
            results["sonc"][f"r{r}"] = {
                "value": safe_float(res.value),
                "status": res.status,
                "error_code": res.error_code,
                "elapsed_s": round(elapsed, 4),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results["sonc"][f"r{r}"] = {
                "value": None,
                "status": "exception",
                "error_code": -1,
                "elapsed_s": round(elapsed, 4),
                "error": str(e)[:200],
            }

        # SOS+SONC
        t0 = time.perf_counter()
        try:
            engine = RelaxationEngine(prog, order=r, verbosity=0)
            res = engine.solve("sosonc_sos_first")
            elapsed = time.perf_counter() - t0
            results["sosonc"][f"r{r}"] = {
                "value": safe_float(res.value),
                "status": res.status,
                "error_code": res.error_code,
                "elapsed_s": round(elapsed, 4),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results["sosonc"][f"r{r}"] = {
                "value": None,
                "status": "exception",
                "error_code": -1,
                "elapsed_s": round(elapsed, 4),
                "error": str(e)[:200],
            }

    return results


def run_original_irene(prob_def: dict, prog):
    """Run using original Irene's SOSONCRelaxations class."""
    from Irene.sosonc import SOSONCRelaxations

    results = {"sos": {}, "sonc": {}, "sosonc": {}}
    degree = prob_def["degree"]
    max_order = max(1, degree // 2)

    for r in range(1, max_order + 1):
        # SOS
        t0 = time.perf_counter()
        try:
            engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=r)
            res = engine.globalMinSOS()
            elapsed = time.perf_counter() - t0
            results["sos"][f"r{r}"] = {
                "value": safe_float(res.val) if hasattr(res, 'val') else safe_float(res),
                "status": res.status if hasattr(res, 'status') else "unknown",
                "error_code": res.error_code if hasattr(res, 'error_code') else 0,
                "elapsed_s": round(elapsed, 4),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results["sos"][f"r{r}"] = {
                "value": None,
                "status": "exception",
                "error_code": -1,
                "elapsed_s": round(elapsed, 4),
                "error": str(e)[:200],
            }

        # SONC
        t0 = time.perf_counter()
        try:
            engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=r)
            res = engine.globalMinSONC()
            elapsed = time.perf_counter() - t0
            results["sonc"][f"r{r}"] = {
                "value": safe_float(res.val) if hasattr(res, 'val') else safe_float(res),
                "status": res.status if hasattr(res, 'status') else "unknown",
                "error_code": res.error_code if hasattr(res, 'error_code') else 0,
                "elapsed_s": round(elapsed, 4),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results["sonc"][f"r{r}"] = {
                "value": None,
                "status": "exception",
                "error_code": -1,
                "elapsed_s": round(elapsed, 4),
                "error": str(e)[:200],
            }

        # SOS+SONC
        t0 = time.perf_counter()
        try:
            engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=r)
            res = engine.globalMinSOSPSONC(first='sos')
            elapsed = time.perf_counter() - t0
            results["sosonc"][f"r{r}"] = {
                "value": safe_float(res.val) if hasattr(res, 'val') else safe_float(res),
                "status": res.status if hasattr(res, 'status') else "unknown",
                "error_code": res.error_code if hasattr(res, 'error_code') else 0,
                "elapsed_s": round(elapsed, 4),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results["sosonc"][f"r{r}"] = {
                "value": None,
                "status": "exception",
                "error_code": -1,
                "elapsed_s": round(elapsed, 4),
                "error": str(e)[:200],
            }

    return results


def run_scipy_optimization(prob_def: dict) -> dict:
    """Run scipy.optimize.minimize with multiple random starts."""
    import numpy as np
    from scipy.optimize import minimize
    from sympy import symbols, lambdify

    variables = prob_def["variables"]
    n = len(variables)
    sym_vars = symbols(variables)

    # Build objective function
    obj_sym = eval(prob_def["objective"], {"__builtins__": {}}, dict(zip(variables, sym_vars)))
    obj_fn = lambdify(sym_vars, obj_sym, "numpy")

    # Build constraint functions
    constraints = []
    for c_expr, c_type in prob_def.get("constraints", []):
        c_sym = eval(c_expr, {"__builtins__": {}}, dict(zip(variables, sym_vars)))
        c_fn = lambdify(sym_vars, c_sym, "numpy")
        if c_type == "eq":
            constraints.append({"type": "eq", "fun": lambda x, f=c_fn: f(*x)})

    best_val = float("inf")
    best_x = None
    best_success = False
    all_vals = []
    num_starts = max(10, 20 * n)

    rng = np.random.RandomState(42)

    for _ in range(num_starts):
        x0 = rng.uniform(-3, 3, size=n)
        try:
            res = minimize(obj_fn, x0, method="L-BFGS-B", constraints=constraints,
                           bounds=None, options={"maxiter": 5000})
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = float(res.fun)
                    best_x = [round(float(xi), 8) for xi in res.x]
                    best_success = res.success
            all_vals.append(float(res.fun))
        except Exception:
            continue

    all_vals_sorted = sorted(all_vals)[:5] if all_vals else []

    return {
        "scipy_best": round(best_val, 10) if best_val != float("inf") else None,
        "scipy_best_x": best_x,
        "scipy_success": best_success,
        "scipy_top5": all_vals_sorted,
        "scipy_num_starts": num_starts,
        "scipy_num_converged": len(all_vals),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Irene vs IreneRewrite comparison benchmark")
    parser.add_argument("--mode", required=True, choices=["irene", "irene_rewrite"],
                        help="Which Irene variant to benchmark")
    parser.add_argument("--scipy", action="store_true",
                        help="Also run Scipy optimization comparisons")
    parser.add_argument("--quick", action="store_true",
                        help="Only run quick problems (skip stress tests)")
    parser.add_argument("--problem", type=str, default=None,
                        help="Run only a specific problem by id")
    args = parser.parse_args()

    global MODE
    MODE = args.mode
    irene_root = setup_paths(MODE)

    # Import after path setup
    from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra

    # Filter problems
    problems = PROBLEMS
    if args.quick:
        quick_ids = {"quad_1d", "quartic_1d", "constrained_1d", "sphere_4", "motzkin"}
        problems = [p for p in problems if p["id"] in quick_ids]
    if args.problem:
        problems = [p for p in problems if p["id"] == args.problem]
        if not problems:
            print(json.dumps({"error": f"Problem '{args.problem}' not found"}))
            sys.exit(1)

    # Run benchmarks
    output = {
        "mode": MODE,
        "irene_root": irene_root,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_problems": len(problems),
        "results": [],
    }

    grand_total_t0 = time.perf_counter()

    for idx, prob_def in enumerate(problems):
        pid = prob_def["id"]
        name = prob_def["name"]
        print(f"\n[{idx+1}/{len(problems)}] {pid}: {name}", file=sys.stderr)

        entry = {
            "id": pid,
            "name": name,
            "degree": prob_def["degree"],
            "variables": prob_def["variables"],
            "true_min": prob_def["true_min"],
            "description": prob_def["description"],
        }

        # Build problem
        t0 = time.perf_counter()
        try:
            sg = CommutativeSemigroup(prob_def["variables"])
            sga = SemigroupAlgebra(sg)
            prog = build_problem(prob_def, sga)
            entry["build_time_s"] = round(time.perf_counter() - t0, 4)
        except Exception as e:
            entry["build_error"] = str(e)[:300]
            entry["build_time_s"] = round(time.perf_counter() - t0, 4)
            output["results"].append(entry)
            print(f"  BUILD ERROR: {e}", file=sys.stderr)
            continue

        # Run relaxations
        t0 = time.perf_counter()
        try:
            if MODE == "irene_rewrite":
                with _SuppressStdout():
                    relax_results = run_irene_rewrite(prob_def, prog)
            else:
                with _SuppressStdout():
                    relax_results = run_original_irene(prob_def, prog)
            entry["relaxation_time_s"] = round(time.perf_counter() - t0, 4)
            entry["relaxations"] = relax_results
        except Exception as e:
            entry["relaxation_error"] = str(e)[:300]
            entry["relaxation_time_s"] = round(time.perf_counter() - t0, 4)
            output["results"].append(entry)
            print(f"  RELAX ERROR: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue

        # Run Scipy comparison
        if args.scipy:
            t0 = time.perf_counter()
            try:
                scipy_res = run_scipy_optimization(prob_def)
                entry["scipy"] = scipy_res
                entry["scipy_time_s"] = round(time.perf_counter() - t0, 4)
            except Exception as e:
                entry["scipy_error"] = str(e)[:300]
                entry["scipy_time_s"] = round(time.perf_counter() - t0, 4)

        output["results"].append(entry)

        # Quick summary
        best_val = None
        for method in ["sos", "sonc", "sosonc"]:
            for order_key, r in relax_results.get(method, {}).items():
                v = r.get("value")
                if v is not None and (best_val is None or v < best_val):
                    best_val = v
        gap = abs(best_val - prob_def["true_min"]) if best_val is not None else None
        print(f"  Best bound: {best_val} (gap: {gap})", file=sys.stderr)

    output["total_time_s"] = round(time.perf_counter() - grand_total_t0, 4)

    # Print JSON to stdout
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
