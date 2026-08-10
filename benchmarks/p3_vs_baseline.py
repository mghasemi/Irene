#!/usr/bin/env python3
"""Phase 3 vs Baseline comparison — degree-6 bivariate stress test.

Runs Motzkin, Choi-Lam, and Robinson at orders 1-3 with:
  (a) baseline config (no reduction pipeline)
  (b) P3 optimized config (Newton pruning + border basis + sparsity)

Measures: matrix dimension, generation time, solve time, final bound.
"""

import time
import json
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxation_api import RelaxationEngine
from Irene.relaxations import RelaxationConfig


def build_motzkin():
    sg = CommutativeSemigroup(["x", "y"])
    sa = SemigroupAlgebra(sg)
    x, y = sa["x"], sa["y"]
    prog = OptimizationProblem(sa)
    f = x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2
    prog.set_objective(f)
    return prog


def build_choi_lam():
    sg = CommutativeSemigroup(["x", "y"])
    sa = SemigroupAlgebra(sg)
    x, y = sa["x"], sa["y"]
    prog = OptimizationProblem(sa)
    f = x**4 * y**2 + x**2 * y**4 + x**2 * y**2 * (x**2 + y**2 - 1)
    prog.set_objective(f)
    return prog


def build_robinson():
    sg = CommutativeSemigroup(["x", "y"])
    sa = SemigroupAlgebra(sg)
    x, y = sa["x"], sa["y"]
    prog = OptimizationProblem(sa)
    f = x**4 * y**2 + x**2 * y**4 + x**4 + y**4 - x**2 - y**2
    prog.set_objective(f)
    return prog


def run_comparison(prog, name, orders=[1, 2, 3]):
    """Run baseline vs P3 optimized for each order."""
    results = {"name": name, "orders": []}

    # Baseline config: no reductions
    baseline_config = RelaxationConfig(
        reduction_method="none",
        monomial_pruning=False,
        sparsity_detection=False,
    )

    # P3 optimized config: Newton polytope pruning + border basis + sparsity
    p3_config = RelaxationConfig(
        reduction_method="newton_polytope",
        monomial_pruning=True,
        sparsity_detection=True,
        verbose_reduction=False,
    )

    for order in orders:
        entry = {"order": order}

        # --- Baseline ---
        engine_base = RelaxationEngine(prog, order=order, solver="clarabel",
                                       verbosity=0, config=baseline_config)
        t0 = time.time()
        res_base = engine_base.solve("sos")
        t_base = time.time() - t0

        entry["baseline"] = {
            "value": round(res_base.value, 8),
            "status": res_base.status,
            "runtime_s": round(t_base, 4),
            "init_time_s": round(res_base.init_time or 0, 4),
            "matrix_dim": res_base.solver_info.get("matrix_dim", None),
        }

        # --- P3 Optimized ---
        engine_p3 = RelaxationEngine(prog, order=order, solver="clarabel",
                                     verbosity=0, config=p3_config)
        t0 = time.time()
        res_p3 = engine_p3.solve("sos")
        t_p3 = time.time() - t0

        entry["p3_optimized"] = {
            "value": round(res_p3.value, 8),
            "status": res_p3.status,
            "runtime_s": round(t_p3, 4),
            "init_time_s": round(res_p3.init_time or 0, 4),
            "matrix_dim": res_p3.solver_info.get("matrix_dim", None),
        }

        # Compute speedup
        if t_base > 0:
            entry["speedup"] = round(t_base / max(t_p3, 1e-9), 2)
        else:
            entry["speedup"] = "N/A"

        results["orders"].append(entry)

    return results


def main():
    problems = [
        (build_motzkin(), "Motzkin"),
        (build_choi_lam(), "Choi-Lam"),
        (build_robinson(), "Robinson"),
    ]

    all_results = {}
    for prog, name in problems:
        print(f"\n{'='*60}")
        print(f"Running {name}...")
        print(f"{'='*60}")
        result = run_comparison(prog, name, orders=[1, 2, 3])
        all_results[name] = result

        for order_entry in result["orders"]:
            o = order_entry["order"]
            base = order_entry["baseline"]
            p3 = order_entry["p3_optimized"]
            print(f"  Order {o}:")
            print(f"    Baseline: val={base['value']:.6e} time={base['runtime_s']:.3f}s "
                  f"(init={base['init_time_s']:.3f}s)")
            print(f"    P3 Opt:   val={p3['value']:.6e} time={p3['runtime_s']:.3f}s "
                  f"(init={p3['init_time_s']:.3f}s)")
            print(f"    Speedup:  {order_entry['speedup']}x")

    # Save results
    out_path = "/home/mehdi/Code/Python/IreneRewrite/benchmarks/results/p3_vs_baseline.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
