#!/usr/bin/env python3
"""Benchmark Gallery Runner — P4.2/P4.3

Loads benchmarks/gallery.yaml, constructs each problem via Irene's API,
runs SOS/SONC/SOSONC relaxations at specified orders, and records structured
JSON results for regression tracking and performance benchmarking.

Usage:
    python run_gallery.py [--solver clarabel|scs|mosek] [--tolerance 1e-4]
                          [--timeout 300] [--filter TAG]
                          [--output-dir ./benchmarks/results/]
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

# ── Add IreneRewrite parent to path ───────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml

try:
    from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
    from Irene.program import OptimizationProblem
    from Irene.sosonc import SOSONCRelaxations
except ImportError as e:
    print(f"FATAL: Cannot import Irene modules: {e}", file=sys.stderr)
    sys.exit(1)


# ====================================================================
# Problem construction helpers
# ====================================================================

def build_problem(problem_def):
    """Construct an OptimizationProblem from a gallery YAML entry."""
    variables = problem_def['variables']
    sg = CommutativeSemigroup(variables)
    sga = SemigroupAlgebra(sg)

    # Build symbol dict for expression evaluation
    sym_dict = {v: sga[v] for v in variables}

    # Parse objective
    obj_expr = problem_def['objective']
    try:
        objective = eval(obj_expr, {"__builtins__": {}}, sym_dict)
    except Exception as e:
        raise ValueError(f"Failed to parse objective '{obj_expr}': {e}")

    prog = OptimizationProblem(sga)
    prog.set_objective(objective)

    # Parse constraints if present
    # Note: Original Irene's add_constraints() only accepts inequality constraints.
    # Equality constraints g(x)=0 are encoded as pair of inequalities g(x)<=0 and -g(x)<=0,
    # but the original API does not support this directly. For now we treat all as ineq.
    if 'constraints' in problem_def:
        for c in problem_def['constraints']:
            try:
                cexpr = eval(c['expr'], {"__builtins__": {}}, sym_dict)
            except Exception as e:
                raise ValueError(f"Failed to parse constraint '{c['expr']}': {e}")
            prog.add_constraints([cexpr])

    return prog


# ====================================================================
# Relaxation runner
# ====================================================================

def run_relaxations(prog, problem_def, solver='clarabel', tolerance=1e-4, timeout=300):
    """Run SOS/SONC relaxations and collect results."""
    category = problem_def.get('category', 'unconstrained')
    degree = problem_def.get('degree', 2)

    # Determine relaxation orders to test based on degree
    max_order = max(1, degree // 2)
    orders = list(range(1, max_order + 1))

    results = {
        'sos': {},
        'sonc': {},
        'sosonc': {},
    }

    for r in orders:
        engine = SOSONCRelaxations(prog, verbosity=0, relaxation_order=r)

        # --- SOS ---
        t0 = time.perf_counter()
        try:
            sos_result = engine.globalMinSOS()
            elapsed = time.perf_counter() - t0
            results['sos'][f'r{r}'] = {
                'value': _safe_float(sos_result.val),
                'status': sos_result.status,
                'error_code': sos_result.error_code,
                'elapsed_s': round(elapsed, 4),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results['sos'][f'r{r}'] = {
                'value': None,
                'status': 'exception',
                'error_code': -1,
                'elapsed_s': round(elapsed, 4),
                'error': str(e)[:200],
            }

        # --- SONC (skip for constrained problems if not supported) ---
        t0 = time.perf_counter()
        try:
            sonc_result = engine.globalMinSONC()
            elapsed = time.perf_counter() - t0
            results['sonc'][f'r{r}'] = {
                'value': _safe_float(sonc_result.val),
                'status': sonc_result.status,
                'error_code': sonc_result.error_code,
                'elapsed_s': round(elapsed, 4),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results['sonc'][f'r{r}'] = {
                'value': None,
                'status': 'exception',
                'error_code': -1,
                'elapsed_s': round(elapsed, 4),
                'error': str(e)[:200],
            }

        # --- SOS+SONC combined ---
        t0 = time.perf_counter()
        try:
            sosonc_result = engine.globalMinSOSPSONC(first='sos')
            elapsed = time.perf_counter() - t0
            results['sosonc'][f'r{r}'] = {
                'value': _safe_float(sosonc_result.val),
                'status': sosonc_result.status,
                'error_code': sosonc_result.error_code,
                'elapsed_s': round(elapsed, 4),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results['sosonc'][f'r{r}'] = {
                'value': None,
                'status': 'exception',
                'error_code': -1,
                'elapsed_s': round(elapsed, 4),
                'error': str(e)[:200],
            }

    return results


def _safe_float(v):
    """Convert to float; return None for inf/nan."""
    try:
        fv = float(v)
        if math.isinf(fv) or math.isnan(fv):
            return None
        return round(fv, 10)
    except (TypeError, ValueError):
        return None


# ====================================================================
# Validation against expected values
# ====================================================================

def validate_result(problem_def, relaxation_results):
    """Compare computed bounds against known true minimum."""
    true_min = problem_def.get('true_min')
    if true_min is None:
        return {'valid': True, 'note': 'No reference value to compare'}

    # Find the best (lowest) finite bound across all methods and orders
    best_bound = None
    best_method = None
    for method in ['sos', 'sonc', 'sosonc']:
        for order_key, res in relaxation_results[method].items():
            val = res.get('value')
            if val is not None:
                if best_bound is None or val < best_bound:
                    best_bound = val
                    best_method = f"{method}_{order_key}"

    if best_bound is None:
        return {'valid': False, 'note': 'No finite bound computed', 'true_min': true_min}

    gap = abs(best_bound - true_min)
    tolerance = 1e-2  # relaxed tolerance for benchmark gallery
    valid = gap <= tolerance

    return {
        'valid': valid,
        'best_bound': best_bound,
        'best_method': best_method,
        'true_min': true_min,
        'gap': round(gap, 10),
        'within_tolerance': gap <= tolerance,
    }


# ====================================================================
# Main runner
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Irene Benchmark Gallery Runner')
    parser.add_argument('--solver', default='clarabel',
                        help='SDP solver backend (default: clarabel)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Solution tolerance (default: 1e-4)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Per-problem timeout in seconds')
    parser.add_argument('--filter', dest='tag_filter', default=None,
                        help='Only run problems with this tag')
    parser.add_argument('--output-dir', default='./benchmarks/results/',
                        help='Output directory for JSON results')
    args = parser.parse_args()

    # Load gallery
    gallery_path = os.path.join(os.path.dirname(__file__), 'gallery.yaml')
    with open(gallery_path) as f:
        gallery_data = yaml.safe_load(f)

    problems = gallery_data['gallery']

    # Filter by tag if requested
    if args.tag_filter:
        problems = [p for p in problems if args.tag_filter in p.get('tags', [])]
        print(f"Filtered to {len(problems)} problem(s) with tag '{args.tag_filter}'")
    else:
        print(f"Running full gallery: {len(problems)} problems")

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    results_file = os.path.join(args.output_dir, f'gallery_{timestamp}.json')

    summary = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'solver': args.solver,
        'tolerance': args.tolerance,
        'total_problems': len(problems),
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'results': [],
    }

    total_t0 = time.perf_counter()

    for idx, prob_def in enumerate(problems):
        pid = prob_def['id']
        pname = prob_def['name']
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(problems)}] {pid}: {pname}")
        print(f"  Category: {prob_def.get('category', 'N/A')} | Degree: {prob_def.get('degree', '?')}")

        t0 = time.perf_counter()
        try:
            prog = build_problem(prob_def)
            build_time = time.perf_counter() - t0
            print(f"  Build time: {build_time:.3f}s")

            relax_results = run_relaxations(
                prog, prob_def,
                solver=args.solver,
                tolerance=args.tolerance,
                timeout=args.timeout,
            )

            elapsed = time.perf_counter() - t0
            validation = validate_result(prob_def, relax_results)

            entry = {
                'id': pid,
                'name': pname,
                'category': prob_def.get('category'),
                'degree': prob_def.get('degree'),
                'build_time_s': round(build_time, 4),
                'total_elapsed_s': round(elapsed, 4),
                'relaxations': relax_results,
                'validation': validation,
            }

            if validation['valid']:
                summary['passed'] += 1
                status_str = '✓ PASS'
            else:
                summary['failed'] += 1
                status_str = '✗ FAIL'

            print(f"  Status: {status_str} | Total time: {elapsed:.3f}s")
            if validation.get('best_bound') is not None:
                print(f"    Best bound: {validation['best_bound']} "
                      f"(via {validation['best_method']})")
                print(f"    True min:   {validation['true_min']} | Gap: {validation['gap']}")

            summary['results'].append(entry)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            summary['errors'] += 1
            print(f"  ✗ ERROR: {e}")
            summary['results'].append({
                'id': pid,
                'name': pname,
                'error': str(e)[:500],
                'elapsed_s': round(elapsed, 4),
            })

    total_elapsed = time.perf_counter() - total_t0
    summary['total_elapsed_s'] = round(total_elapsed, 4)

    # Write results
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Total problems: {summary['total_problems']}")
    print(f"  Passed:         {summary['passed']}")
    print(f"  Failed:         {summary['failed']}")
    print(f"  Errors:         {summary['errors']}")
    print(f"  Total time:     {total_elapsed:.2f}s")
    print(f"  Results saved:  {results_file}")

    return summary


if __name__ == '__main__':
    main()
