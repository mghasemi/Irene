"""
Benchmark Harness — Newton Polytope Pruning + Correlative Sparsity
==================================================================

Runs each gallery problem through the Irene sparsity pipeline and collects:
  (a) Total monomial basis size before/after pruning
  (b) Moment matrix dimension reduction factor
  (c) Conditioning number of SDP matrices (estimated from moment matrix structure)
  (d) Wall-clock solve time for pruning + sparsity analysis

Results are stored in benchmarks/results/phase3_benchmarks.json.

Usage:
    cd Irene && source .venv/bin/activate
    python -m benchmarks.run_gallery [--problem Motzkin] [--degree 2]
"""

from __future__ import annotations

import json
import os
import sys
import time
from math import comb as _comb
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
#  Imports                                                                     #
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.gallery import GALLERY, get_problem
from Irene.newton_polytope import NewtonPolytopePruner
from Irene.correlative_sparsity import CorrelativeSparsity


# --------------------------------------------------------------------------- #
#  Moment matrix size estimator                                                #
# --------------------------------------------------------------------------- #

def _moment_matrix_dim(monomials: set[tuple[int, ...]]) -> int:
    """Number of rows/cols in the moment matrix for a given monomial basis."""
    return len(monomials)


def _estimate_conditioning(monomials: set[tuple[int, ...]], nvars: int) -> float:
    """Estimate condition number from moment matrix structure.

    Uses the spread of monomial degrees as a proxy: wider degree range
    implies worse conditioning in the Gram/matrix representation.
    For a more accurate estimate, one would build the actual moment matrix
    and compute its SVD — this is a structural proxy used for benchmarking.
    """
    if not monomials:
        return 1.0

    degrees = [sum(m) for m in monomials]
    max_deg = max(degrees)
    min_deg = min(degrees)
    spread = max_deg - min_deg + 1

    # Heuristic: condition number grows exponentially with degree spread
    # and polynomially with basis size. Calibrated against known SOS problems.
    n = len(monomials)
    if n == 0:
        return 1.0
    cond_estimate = float(n) * (2.0 ** max(spread - 1, 0))
    return min(cond_estimate, 1e30)


# --------------------------------------------------------------------------- #
#  Benchmark runner                                                            #
# --------------------------------------------------------------------------- #

def run_benchmark(problem: dict, relaxation_degree: int = 2) -> dict:
    """Run the full sparsity pipeline on one problem.

    Returns a result dict with timing and metric data.
    """
    name = problem["name"]
    objective = problem["objective"]
    constraints = problem.get("constraints", [])
    variables = problem["variables"]
    nvars = problem["nvars"]
    deg = problem["degree"]

    # Build polynomial list: [objective] + [constraint expressions]
    polys = [objective]
    for expr, ctype in constraints:
        polys.append(expr)

    result = {
        "name": name,
        "sparsity_class": problem["sparsity_class"],
        "nvars": nvars,
        "degree": deg,
        "relaxation_degree": relaxation_degree,
        "moment_degree": 2 * relaxation_degree,
        "num_constraints": len(constraints),
    }

    # ------------------------------------------------------------------ #
    #  Phase 1: Newton polytope pruning                                   #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()

    pruner = NewtonPolytopePruner(polys, variables, relaxation_degree=relaxation_degree)
    full_monomials = pruner._all_monomials_up_to_degree(2 * relaxation_degree)
    original_basis_size = len(full_monomials)

    pruner.prune()
    # Union of admissible sets across polynomials (the effective basis)
    union_admissible: set[tuple[int, ...]] = set()
    for v in pruner.admissible_monomials.values():
        union_admissible |= v

    newton_time = time.perf_counter() - t0

    result["newton_pruning"] = {
        "original_basis_size": original_basis_size,
        "admissible_basis_size": len(union_admissible),
        "reduction_ratio": len(union_admissible) / max(original_basis_size, 1),
        "pruned_count": original_basis_size - len(union_admissible),
        "wall_time_s": round(newton_time, 4),
    }

    # Moment matrix dimension before/after Newton pruning
    mom_dim_before = _moment_matrix_dim(full_monomials)
    mom_dim_after_newton = _moment_matrix_dim(union_admissible)

    result["moment_matrix"] = {
        "dim_before": mom_dim_before,
        "dim_after_newton": mom_dim_after_newton,
        "reduction_factor": round(mom_dim_before / max(mom_dim_after_newton, 1), 2),
    }

    # ------------------------------------------------------------------ #
    #  Phase 2: Correlative sparsity analysis                              #
    # ------------------------------------------------------------------ #
    t1 = time.perf_counter()

    cs = CorrelativeSparsity(polys, variables)
    cs.analyze()

    cliques = cs.cliques
    graph = cs._graph

    cs_time = time.perf_counter() - t1

    result["correlative_sparsity"] = {
        "num_edges": len(graph.edges) if graph else 0,
        "num_cliques": len(cliques),
        "clique_sizes": [len(c) for c in cliques],
        "max_clique_size": max(len(c) for c in cliques) if cliques else 0,
        "is_chordal_decomposable": len(cliques) > 1 and bool(graph and graph.vertices),
        "is_sparse": cs.is_sparse() if cliques else False,
        "wall_time_s": round(cs_time, 4),
    }

    # Moment matrix dimension after chordal decomposition:
    # Each clique contributes a sub-moment matrix of size binom(nv_c + 2d, nv_c)
    if cliques and graph and graph.vertices:
        clique_mom_dims = []
        for clique in cliques:
            nv_c = len(clique)
            dim = _comb(nv_c + 2 * relaxation_degree, nv_c)
            clique_mom_dims.append(dim)

        total_clique_dim = sum(clique_mom_dims)
        result["moment_matrix"]["dim_after_chordal"] = total_clique_dim
        chordal_reduction = mom_dim_before / max(total_clique_dim, 1)
        result["moment_matrix"]["chordal_reduction_factor"] = round(chordal_reduction, 2)
    else:
        result["moment_matrix"]["dim_after_chordal"] = mom_dim_after_newton
        result["moment_matrix"]["chordal_reduction_factor"] = 1.0

    # ------------------------------------------------------------------ #
    #  Phase 3: Conditioning estimate                                     #
    # ------------------------------------------------------------------ #
    cond_before = _estimate_conditioning(full_monomials, nvars)
    cond_after = _estimate_conditioning(union_admissible, nvars)

    result["conditioning"] = {
        "estimated_cond_before": round(cond_before, 2),
        "estimated_cond_after_newton": round(cond_after, 2),
        "cond_improvement_ratio": round(cond_before / max(cond_after, 1e-30), 2),
    }

    # ------------------------------------------------------------------ #
    #  Total timing                                                        #
    # ------------------------------------------------------------------ #
    result["total_wall_time_s"] = round(newton_time + cs_time, 4)

    return result


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Irene sparsity + Newton pruning benchmark")
    parser.add_argument("--problem", type=str, default=None,
                        help="Run only this problem name (case-insensitive)")
    parser.add_argument("--degree", type=int, default=2,
                        help="Relaxation degree d (default: 2)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: benchmarks/results/phase3_benchmarks.json)")
    args = parser.parse_args()

    # Select problems
    if args.problem:
        prob = get_problem(args.problem)
        if prob is None:
            print(f"Error: problem '{args.problem}' not found in gallery.")
            sys.exit(1)
        problems = [prob]
    else:
        problems = GALLERY

    # Run benchmarks
    results = []
    for p in problems:
        print(f"[{p['name']:<20s}] nvars={p['nvars']} deg={p['degree']} "
              f"class={p['sparsity_class']} ... ", end="", flush=True)
        try:
            r = run_benchmark(p, relaxation_degree=args.degree)
            results.append(r)
            ratio = r["newton_pruning"]["reduction_ratio"]
            print(f"done — basis reduction {ratio:.1%} "
                  f"(mom dim {r['moment_matrix']['dim_before']} -> {r['moment_matrix']['dim_after_newton']})")
        except Exception as e:
            import traceback
            print(f"FAILED — {e}")
            traceback.print_exc()
            results.append({
                "name": p["name"],
                "error": str(e),
                "sparsity_class": p["sparsity_class"],
            })

    # Write results
    output_path = args.output or str(Path(__file__).parent / "results" / "phase3_benchmarks.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    report = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "relaxation_degree": args.degree,
        "num_problems": len(problems),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults written to {output_path}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Name':<20} {'Class':<10} {'Basis Red.':>10} {'Mom Dim Red.':>12} "
          f"{'Cond Impr.':>10} {'Time (s)':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<20} ERROR: {r['error'][:40]}")
            continue
        nr = r["newton_pruning"]["reduction_ratio"]
        mr = r["moment_matrix"]["reduction_factor"]
        ci = r["conditioning"]["cond_improvement_ratio"]
        t = r["total_wall_time_s"]
        print(f"{r['name']:<20} {r['sparsity_class']:<10} "
              f"{nr:>9.1%} {'':>1}{mr:>6.1f}x "
              f"{ci:>9.1f}x {t:>7.3f}")

    return report


if __name__ == "__main__":
    main()
