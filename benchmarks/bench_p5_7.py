#!/usr/bin/env python3
"""P5.7 Benchmark — Measure SDP init time with precomputed entry dicts

Compares sInitSDP wall-clock time on Motzkin and larger problems to validate
the P5.7 optimization (precomputing _poly().as_dict() per moment matrix entry
instead of re-running it for every Calpha call).

Usage:
    python benchmarks/bench_p5_7.py [--problem motzkin|choi_lam|dense_deg8]
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxations import SDPRelaxations


def build_motzkin():
    sg = CommutativeSemigroup(['x', 'y'])
    sga = SemigroupAlgebra(sg)
    prog = OptimizationProblem(sga)
    obj = sga['x']**4 * sga['y']**2 + sga['x']**2 * sga['y']**4 + sga.one - 3*sga['x']**2*sga['y']**2
    prog.set_objective(obj)
    return prog


def build_choi_lam():
    sg = CommutativeSemigroup(['x', 'y'])
    sga = SemigroupAlgebra(sg)
    prog = OptimizationProblem(sga)
    obj = sga['x']**4 * sga['y']**2 + sga['x']**2 * sga['y']**4 \
          + sga['x']**2 * sga['y']**2 * (sga['x']**2 + sga['y']**2 - 1)
    prog.set_objective(obj)
    return prog


def build_dense_deg8():
    sg = CommutativeSemigroup(['x', 'y'])
    sga = SemigroupAlgebra(sg)
    prog = OptimizationProblem(sga)
    obj = (sga['x'] + sga['y'])**8
    prog.set_objective(obj)
    return prog


PROBLEMS = {
    'motzkin': ('Motzkin poly (deg 6)', build_motzkin),
    'choi_lam': ('Choi-Lam poly (deg 6)', build_choi_lam),
    'dense_deg8': ('Dense bivariate deg 8', build_dense_deg8),
}


def bench_init(problem_id, order=3):
    name, builder = PROBLEMS[problem_id]
    prog = builder()

    # Use SDPRelaxations directly — this is where sInitSDP lives
    engine = SDPRelaxations.from_problem(prog)
    engine.MomentsOrd(order)
    engine.SetSDPSolver('cvxopt')

    t0 = time.perf_counter()
    engine.InitSDP()  # calls sInitSDP internally
    elapsed = time.perf_counter() - t0

    basis_2d = len(engine.ReducedMonomialBase(2 * order))
    basis_d = len(engine.ReducedMonomialBase(order))
    num_constraints = len(engine.CnsDegs) if hasattr(engine, 'CnsDegs') else 0

    return {
        'problem': problem_id,
        'name': name,
        'order': order,
        'basis_2d': basis_2d,
        'basis_d': basis_d,
        'num_constraints': num_constraints,
        'init_time_s': round(elapsed, 4),
    }


def main():
    parser = argparse.ArgumentParser(description='P5.7 Init Time Benchmark')
    parser.add_argument('--problem', default='all', choices=['motzkin', 'choi_lam', 'dense_deg8', 'all'])
    parser.add_argument('--order', type=int, default=3)
    args = parser.parse_args()

    ids = list(PROBLEMS.keys()) if args.problem == 'all' else [args.problem]

    print(f"\n{'='*60}")
    print("P5.7 SDP Init Time Benchmark")
    print(f"Relaxation order: {args.order}")
    print(f"{'='*60}\n")

    results = []
    for pid in ids:
        r = bench_init(pid, args.order)
        results.append(r)
        print(f"[{pid}] {r['name']}")
        print(f"  Basis(2d): {r['basis_2d']}, Basis(d): {r['basis_d']}")
        print(f"  Init time: {r['init_time_s']:.4f}s\n")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'results', f'p5_7_bench.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
