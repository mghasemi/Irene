#!/usr/bin/env python3
"""P3.6 + P3.7: Phase 3 reduction benchmarks — sparsity, Newton pruning, border basis.

Fixed version: handles NewtonPruner API correctly, fixes border basis Groebner LM.
"""

import json
import math
import os
import sys
import time
from itertools import product

import numpy as np
import yaml
from sympy import symbols, groebner, Poly

# Add Irene to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from Irene.symbolic_engine import engine
from Irene.sparsity import detect_sparsity_from_polys
from Irene.newton_polytope import prune_basis_from_polys
from Irene.border_basis import BorderBasis


# ==========================================================================
# Load gallery
# ==========================================================================

def load_gallery():
    with open('benchmarks/gallery.yaml') as f:
        data = yaml.safe_load(f)
    return data['gallery']


# ==========================================================================
# 1. CORRELATIVE SPARSITY BENCHMARKS
# ==========================================================================

def benchmark_sparsity():
    gallery = load_gallery()
    results = []

    for prob in gallery:
        pid = prob['id']
        name = prob['name']
        n_vars = len(prob['variables'])
        degree = prob.get('degree', 2)

        try:
            vars_sym = symbols(prob['variables'])
            sym_dict = {v: vars_sym[i] for i, v in enumerate(prob['variables'])}
            obj_expr = eval(prob['objective'], {"__builtins__": {}}, sym_dict)
            polys = [obj_expr]
            if 'constraints' in prob:
                for c in prob['constraints']:
                    polys.append(eval(c['expr'], {"__builtins__": {}}, sym_dict))

            sp = detect_sparsity_from_polys(polys, num_vars=n_vars)
            summary = sp.summary()

            entry = {
                'id': pid, 'name': name, 'num_vars': n_vars, 'degree': degree,
                'sparsity': {
                    'is_sparse': sp.is_sparse,
                    'num_components': len(sp.components),
                    'component_sizes': summary.get('component_sizes', []),
                    'component_vars': summary.get('component_vars', []),
                },
            }
            for d in range(1, min(5, degree // 2 + 3) + 1):
                entry[f'reduction_factor_d{d}'] = round(sp.reduction_factor(deg=d), 4)
                partition = sp.moment_matrix_partition(deg=d)
                entry[f'partition_blocks_d{d}'] = len(partition)
            results.append(entry)
        except Exception as e:
            results.append({'id': pid, 'name': name, 'sparsity_error': str(e)[:200]})

    return results


# ==========================================================================
# 2. NEWTON POLYTOPE PRUNING BENCHMARKS
# ==========================================================================

def benchmark_newton_pruning():
    gallery = load_gallery()
    results = []

    for prob in gallery:
        pid = prob['id']
        name = prob['name']
        n_vars = len(prob['variables'])
        degree = prob.get('degree', 2)

        try:
            vars_sym = symbols(prob['variables'])
            sym_dict = {v: vars_sym[i] for i, v in enumerate(prob['variables'])}
            obj_expr = eval(prob['objective'], {"__builtins__": {}}, sym_dict)
            polys = [obj_expr]
            if 'constraints' in prob:
                for c in prob['constraints']:
                    polys.append(eval(c['expr'], {"__builtins__": {}}, sym_dict))

            entry = {'id': pid, 'name': name, 'num_vars': n_vars, 'degree': degree}

            max_r = max(1, degree // 2)
            for r in range(1, max_r + 1):
                max_deg = 2 * r
                try:
                    pruner = prune_basis_from_polys(polys, num_vars=n_vars, max_degree=max_deg)
                    # Use the correct API keys: 'full_basis_size', 'pruned_basis_size', 
                    # 'reduction_ratio', 'matrix_entry_reduction', 'entries_saved'
                    info = pruner.moment_matrix_dimension_reduction()
                    n_full = info['full_basis_size']
                    n_pruned = info['pruned_basis_size']
                    full_mm = n_full * (n_full + 1) // 2   # symmetric moment matrix entries
                    pruned_mm = n_pruned * (n_pruned + 1) // 2

                    entry[f'order_{r}'] = {
                        'full_basis': n_full,
                        'pruned_basis': n_pruned,
                        'full_mm_entries': full_mm,
                        'pruned_mm_entries': pruned_mm,
                        'reduction_ratio': round(info['reduction_ratio'], 4),
                        'entries_saved': info['entries_saved'],
                    }
                except Exception as e:
                    entry[f'order_{r}'] = {'error': str(e)[:200]}
            results.append(entry)
        except Exception as e:
            results.append({'id': pid, 'name': name, 'newton_error': str(e)[:200]})

    return results


# ==========================================================================
# 3. BORDER BASIS CONDITIONING BENCHMARKS
# ==========================================================================

def condition_number(M):
    if M.size == 0:
        return float('inf')
    try:
        s = np.linalg.svd(M, compute_uv=False)
        s = s[s > 1e-15]
        if len(s) == 0:
            return float('inf')
        return float(max(s) / min(s))
    except Exception:
        return float('inf')


def benchmark_border_basis_conditioning():
    x, y = symbols('x y')
    test_ideals = [
        {'id': 'ideal_x2_y2', 'name': '<x^2, y^2>', 'vars': [x, y],
         'gens': [x**2, y**2], 'degrees': [2, 3]},
        {'id': 'ideal_x3_y3', 'name': '<x^3, y^3>', 'vars': [x, y],
         'gens': [x**3, y**3], 'degrees': [2, 3]},
        {'id': 'circle', 'name': '<x^2+y^2-1>', 'vars': [x, y],
         'gens': [x**2 + y**2 - 1], 'degrees': [2, 3]},
        {'id': 'hyperbola', 'name': '<xy-1>', 'vars': [x, y],
         'gens': [x*y - 1], 'degrees': [2, 3]},
        {'id': 'motzkin_grad', 'name': 'Motzkin Grad Ideal',
         'vars': [x, y],
         'gens': [4*x**3*y**2 + 2*x*y**4 - 6*x*y**2,
                  2*x**4*y + 4*x**2*y**3 - 6*x**2*y],
         'degrees': [2]},
    ]

    results = []

    for ideal in test_ideals:
        entry = {'id': ideal['id'], 'name': ideal['name'], 'degrees': {}}

        for d in ideal['degrees']:
            try:
                # --- Border Basis ---
                t0 = time.perf_counter()
                bb = BorderBasis(ideal['vars'], ideal['gens'], degree=d)
                bb_time = (time.perf_counter() - t0) * 1000
                n_bb = len(bb.basis)
                n_border = len(bb.border)
                n_tables = len(bb.mult_tables)

                # Conditioning from multiplication tables
                if bb.mult_tables and n_bb > 0:
                    tbl_rows = list(bb.mult_tables.values())
                    # Filter rows to match number of basis columns
                    valid = [row for row in tbl_rows if len(row) == n_bb]
                    if valid:
                        M_bb = np.array(valid)
                        cond_bb = condition_number(M_bb)
                    else:
                        cond_bb = 1.0
                else:
                    cond_bb = 1.0

                # --- Groebner Basis ---
                t0 = time.perf_counter()
                G = groebner(ideal['gens'], *ideal['vars'], order='grevlex')
                gr_time = (time.perf_counter() - t0) * 1000

                # Count Groebner basis size using proper Poly API
                all_exps = [exp for exp in product(range(d + 1), repeat=len(ideal['vars']))
                           if sum(exp) <= d]
                basis_set = set(all_exps)

                for g in G:
                    try:
                        # Poly gives us the leading monomial's exponent
                        gp = Poly(g, *ideal['vars'])
                        lm = gp.LM()  # Leading monomial
                        lm_exp = tuple(lm.as_powers_dict().get(v, 0) for v in ideal['vars'])
                    except Exception:
                        # Fallback: extract from .as_dict()
                        gd = gp.as_dict()
                        if gd:
                            lm_exp = max(gd.keys(), key=lambda e: (sum(e), e))
                        else:
                            continue
                    # Remove all monomials divisible by the leading monomial
                    to_remove = set()
                    for e in basis_set:
                        if all(e[i] >= lm_exp[i] for i in range(len(ideal['vars']))):
                            to_remove.add(e)
                    basis_set -= to_remove

                n_gr = len(basis_set)
                n_gr_gens = len(G)

                # Build relation matrix for Groebner conditioning
                exp_to_idx = {exp: i for i, exp in enumerate(all_exps)}
                n_all = len(all_exps)
                rows_gr = []
                for g in ideal['gens']:
                    gd = Poly(g, *ideal['vars']).as_dict()
                    g_deg = max((sum(e) for e in gd.keys()), default=0)
                    max_shift = d - g_deg
                    if max_shift < 0:
                        continue
                    shift_exps = [e for e in all_exps if sum(e) <= max_shift]
                    for gamma in shift_exps:
                        shifted = {}
                        for gi, c in gd.items():
                            s_exp = tuple(int(gamma[i]) + int(gi[i]) for i in range(len(ideal['vars'])))
                            shifted[s_exp] = float(c) + shifted.get(s_exp, 0.0)
                        row = np.zeros(n_all)
                        for s_exp, c in shifted.items():
                            if sum(s_exp) <= d and s_exp in exp_to_idx:
                                row[exp_to_idx[s_exp]] += c
                        if np.any(row != 0):
                            rows_gr.append(row)

                if rows_gr:
                    M_gr = np.array(rows_gr)
                    cond_gr = condition_number(M_gr)
                else:
                    cond_gr = 1.0

                cond_ratio = (cond_gr / cond_bb) if (cond_bb > 0 and not math.isinf(cond_bb)) else float('inf')

                entry['degrees'][f'd{d}'] = {
                    'border_basis_size': n_bb,
                    'groebner_basis_size': n_gr,
                    'border_size': n_border,
                    'num_mult_tables': n_tables,
                    'border_basis_condition': round(cond_bb, 2) if not math.isinf(cond_bb) else 'inf',
                    'groebner_condition': round(cond_gr, 2) if not math.isinf(cond_gr) else 'inf',
                    'condition_ratio': round(cond_ratio, 2) if not math.isinf(cond_ratio) else 'inf',
                    'border_time_ms': round(bb_time, 2),
                    'groebner_time_ms': round(gr_time, 2),
                    'groebner_num_gens': n_gr_gens,
                }

            except Exception as e:
                entry['degrees'][f'd{d}'] = {'error': str(e)[:200]}

        results.append(entry)

    return results


# ==========================================================================
# 4. COMBINED SCALING
# ==========================================================================

def benchmark_scaling():
    x_vars = [symbols(f'x{i}') for i in range(6)]
    n = 6

    test_cases = [
        {'id': 'fully_sparse', 'name': 'Fully Sparse (6 indep)',
         'polys': [x_vars[i]**4 for i in range(n)]},
        {'id': 'chain', 'name': 'Chain coupling',
         'polys': [x_vars[i]**2 + x_vars[i+1]**2 + x_vars[i]*x_vars[i+1] for i in range(n-1)]},
        {'id': 'star', 'name': 'Star (x0 hub)',
         'polys': [x_vars[0]**2 + x_vars[i]**2 + x_vars[0]*x_vars[i] for i in range(1, n)]},
        {'id': 'fully_dense', 'name': 'Fully Dense',
         'polys': [(sum(x_vars))**2]},
    ]

    results = []
    for tc in test_cases:
        entry = {'id': tc['id'], 'name': tc['name'], 'num_vars': n}

        # Sparsity
        try:
            sp = detect_sparsity_from_polys(tc['polys'], num_vars=n)
            sp_summ = sp.summary()
            entry['sparsity'] = {
                'is_sparse': sp.is_sparse,
                'num_components': len(sp.components),
                'component_sizes': sp_summ.get('component_sizes', []),
                'reduction_factor_d2': round(sp.reduction_factor(deg=2), 4),
                'reduction_factor_d3': round(sp.reduction_factor(deg=3), 4),
            }
        except Exception as e:
            entry['sparsity'] = {'error': str(e)[:200]}

        # Newton pruning
        try:
            pruner = prune_basis_from_polys(tc['polys'], num_vars=n, max_degree=4)
            info = pruner.moment_matrix_dimension_reduction()
            entry['newton'] = {
                'full_basis': info['full_basis_size'],
                'pruned_basis': info['pruned_basis_size'],
                'reduction_ratio': round(info['reduction_ratio'], 4),
                'entries_saved': info['entries_saved'],
            }
        except Exception as e:
            entry['newton'] = {'error': str(e)[:200]}

        results.append(entry)

    return results


# ==========================================================================
# Main
# ==========================================================================

def main():
    print("=" * 60)
    print("PHASE 3 REDUCTION BENCHMARKS")
    print("=" * 60)

    all_results = {}
    times = {}

    for label, fn in [
        ('sparsity', benchmark_sparsity),
        ('newton_pruning', benchmark_newton_pruning),
        ('border_basis', benchmark_border_basis_conditioning),
        ('scaling', benchmark_scaling),
    ]:
        print(f"\n[{label}] Running...")
        t0 = time.perf_counter()
        all_results[label] = fn()
        elapsed = time.perf_counter() - t0
        times[label] = elapsed
        n = len(all_results[label])
        print(f"  Done in {elapsed:.2f}s ({n} results)")

    # Write
    outdir = 'benchmarks/results'
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'phase3_benchmarks.json')
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {outpath}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Sparsity
    sp = all_results['sparsity']
    sd = [r for r in sp if isinstance(r.get('sparsity'), dict) and r['sparsity'].get('is_sparse')]
    print(f"\nSparsity: {len(sd)}/{len(sp)} problems detect correlative sparsity")
    for r in sd:
        comps = r['sparsity']['num_components']
        sizes = r['sparsity']['component_sizes']
        rf2 = r.get('reduction_factor_d2', 'N/A')
        print(f"  {r['id']}: {comps} components {sizes}, rf_d2={rf2}")

    # Newton
    np_res = all_results['newton_pruning']
    print(f"\nNewton Polytope Pruning:")
    for r in np_res:
        for key in sorted(r.keys()):
            if key.startswith('order_'):
                v = r[key]
                if 'reduction_ratio' in v and v['reduction_ratio'] < 1.0:
                    pct = (1 - v['reduction_ratio']) * 100
                    print(f"  {r['id']} @ {key}: {v['full_basis']} -> {v['pruned_basis']} ({pct:.1f}% reduction)")

    # Border basis
    bb_res = all_results['border_basis']
    print(f"\nBorder Basis Conditioning:")
    for r in bb_res:
        print(f"  {r['name']}:")
        for dk, dv in sorted(r.get('degrees', {}).items()):
            if 'error' not in dv:
                print(f"    {dk}: BB={dv['border_basis_size']} Gr={dv['groebner_basis_size']} "
                      f"cond_BB={dv.get('border_basis_condition','?')} cond_Gr={dv.get('groebner_condition','?')} "
                      f"t_BB={dv.get('border_time_ms','?')}ms t_Gr={dv.get('groebner_time_ms','?')}ms")

    # Scaling
    sc = all_results['scaling']
    print(f"\nCombined Scaling:")
    for r in sc:
        sp_s = r.get('sparsity', {})
        nt = r.get('newton', {})
        rf2 = sp_s.get('reduction_factor_d2', '?') if isinstance(sp_s, dict) else 'error'
        rr = nt.get('reduction_ratio', '?') if isinstance(nt, dict) else 'error'
        print(f"  {r['id']}: sparsity_rf2={rf2}, newton_reduction={rr}")

    return all_results


if __name__ == '__main__':
    main()
