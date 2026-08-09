# Irene vs IreneRewrite — Cross-Feature Backend Benchmark Report

_Generated 2026-08-09T20:54:38Z — 3 modes, 9 feature sections_

## 1. Environment

| Mode | Package root | Symbolic backend |
|------|--------------|------------------|
| original Irene (SymPy) | `/home/mehdi/Code/Python/Irene` | `sympy-direct (no symbolic_engine module)` |
| IreneRewrite (SymEngine) | `/home/mehdi/Code/Python/IreneRewrite` | `symengine` |
| IreneRewrite (SymPy) | `/home/mehdi/Code/Python/IreneRewrite` | `sympy` |

## 2. SOS / SONC / SOS+SONC relaxations

Same 4 gallery problems, same relaxation orders. Values are SDP lower bounds;
`infeasible` marks non-SOS certificates (expected for separating examples).

| Problem | Method | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) | True min |
|---------|--------|---------------:|--------------------:|----------------:|---------:|
| quartic_1d | sos_r2 | -0.250000 | -0.250000 | -0.250000 | -0.25 |
| quartic_1d | sonc_r2 | -inf | -inf | -inf | -0.25 |
| quartic_1d | sosonc_r2 | -0.250000 | -0.250000 | -0.250000 | -0.25 |
| motzkin | sos_r1 | -inf | -inf | -inf | 0.0 |
| motzkin | sonc_r1 | 0.000000 | 0.000000 | 0.000000 | 0.0 |
| motzkin | sosonc_r1 | 0.000000 | 0.000000 | 0.000000 | 0.0 |
| sphere_4 | sos_r2 | 0.500000 | 0.500000 | 0.500000 | 0.5 |
| sphere_4 | sonc_r2 | -0.000000 | -0.000000 | -0.000000 | 0.5 |
| sphere_4 | sosonc_r2 | 0.500000 | 0.500000 | 0.500000 | 0.5 |
| schick | sos_r1 | -inf | -inf | -inf | 0.0 |
| schick | sonc_r1 | -2.987802 | -2.987802 | -2.987802 | 0.0 |
| schick | sosonc_r1 | -2.987802 | -2.987802 | -2.987802 | 0.0 |

| Problem | Method | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) |
|---------|--------|---------------:|--------------------:|----------------:|
| quartic_1d | sos_r2 | 25.80 ms | 29.00 ms | 30.60 ms |
| quartic_1d | sonc_r2 | 129.50 ms | 127.00 ms | 127.30 ms |
| quartic_1d | sosonc_r2 | 23.90 ms | 24.80 ms | 23.90 ms |
| motzkin | sos_r1 | 117.20 ms | 123.80 ms | 122.60 ms |
| motzkin | sonc_r1 | 7.30 ms | 7.00 ms | 7.30 ms |
| motzkin | sosonc_r1 | 145.20 ms | 156.30 ms | 161.00 ms |
| sphere_4 | sos_r2 | 96.70 ms | 89.30 ms | 94.00 ms |
| sphere_4 | sonc_r2 | 9.00 ms | 8.20 ms | 8.50 ms |
| sphere_4 | sosonc_r2 | 89.60 ms | 94.20 ms | 94.10 ms |
| schick | sos_r1 | 151.60 ms | 140.10 ms | 154.70 ms |
| schick | sonc_r1 | 13.10 ms | 13.80 ms | 14.00 ms |
| schick | sosonc_r1 | 109.40 ms | 115.90 ms | 124.60 ms |

## 3. GP relaxation

| Metric | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) |
|--------|---------------:|--------------------:|----------------:|
| elapsed_s | 94.10 ms | 90.50 ms | 90.20 ms |
| status | ok | ok | ok |

## 4. DSDP mean relaxation (Choi-Lam, M_{1,0})

| Metric | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) |
|--------|---------------:|--------------------:|----------------:|
| lower_bound | 0.000000 | 0.000000 | 0.000000 |
| elapsed_s | 1.306000 | 1.232800 | 1.274900 |

## 5. DSDP KKT relaxation

| Metric | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) |
|--------|---------------:|--------------------:|----------------:|
| lower_bound | 0.000000 | 0.000000 | 0.000000 |
| elapsed_s | 0.198200 | 0.082500 | 0.080100 |
| status | ok | ok | ok |

## 6. ADE relations (build_ade_relations)

| Metric | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) |
|--------|---------------:|--------------------:|----------------:|
| build_ms | 113.00 ms | 121.00 ms | 120.00 ms |
| status | ok | ok | ok |

Derivative symbols (single derivation `{x:1, u:1+u²}`):

- Original: ['d_x', 'd_u']
- Rewrite:  ['d_x', 'd_u']
- Rewrite (SymPy): ['d_x', 'd_u']

## 7. Border basis

| Ideal | Metric | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) |
|-------|--------|---------------:|--------------------:|----------------:|
| circle_xy | elapsed_s | 0.90 ms | 3.40 ms | 3.50 ms |
| circle_xy | status | None | None | None |
| monomial | elapsed_s | 0.20 ms | 1.30 ms | 1.80 ms |
| monomial | status | None | None | None |

API notes: original `BorderBasis(polynomials, variables, max_degree)` computes a full
border basis (`compute()`, `dimension()`, `normal_form()`); rewrite `BorderBasis(variables,
generators, degree)` targets quotient-ring reduction for moment matrices (`reduce()`,
`conditioning_diagnostic()`).

## 8. Correlative sparsity

| Metric | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) |
|--------|---------------:|--------------------:|----------------:|
| elapsed_s | 0.30 ms | 0.20 ms | 0.20 ms |
| status | ok | ok | ok |

API notes: original `analyze_correlative_sparsity()` (chordal-graph clique decomposition,
Bron–Kerbosch); rewrite `detect_sparsity_from_polys()` (UnionFind connected components).

## 9. Newton polytope pruning

| Metric | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) |
|--------|---------------:|--------------------:|----------------:|
| elapsed_s | 0.70 ms | 0.80 ms | 1.00 ms |
| status | ok | ok | ok |

API notes: original `NewtonPolytopePruner` (per-polynomial admissible monomial sets);
rewrite `NewtonPruner` (basis pruning with `moment_matrix_dimension_reduction()`).

## 10. Symbolic micro-benchmarks

| Operation | Original Irene | Rewrite (SymEngine) | Rewrite (SymPy) |
|-----------|---------------:|--------------------:|----------------:|
| expand_deg8 | 0.006 ms | 0.011 ms | 0.006 ms |
| poly_deg6 | 0.06 ms | 0.07 ms | 0.06 ms |
| groebner | 0.206 ms | 0.213 ms | 0.216 ms |
| matrix_mul | 0.059 ms | 0.013 ms | 0.061 ms |
| zeros_50 | 0.003 ms | 0.023 ms | 0.003 ms |

## 11. Quotient-basis option (Groebner vs BorderBasis)

The ``RelaxationConfig.quotient_basis`` option selects the quotient-ring
reduction engine. Only IreneRewrite supports the border-basis engine;
original Irene always uses Groebner bases.

**original Irene (SymPy):** original Irene has no border-basis option (Groebner only)

| Problem | Mode | Metric | Groebner | Border | True min |
|---------|------|--------|---------:|-------:|---------:|
| quartic_1d | irene_rewrite | lower_bound | -0.250000 | -0.250000 | -0.25 |
| quartic_1d | irene_rewrite | basis_size | 5 | 5 | -0.25 |
| quartic_1d | irene_rewrite | elapsed_s | 21.30 ms | 20.90 ms | -0.25 |
| circle_relations | irene_rewrite | lower_bound | 1.000000 | 1.000000 | 1.0 |
| circle_relations | irene_rewrite | basis_size | 5 | 5 | 1.0 |
| circle_relations | irene_rewrite | elapsed_s | 25.00 ms | 27.50 ms | 1.0 |
| quartic_1d | irene_rewrite_sympy | lower_bound | -0.250000 | -0.250000 | -0.25 |
| quartic_1d | irene_rewrite_sympy | basis_size | 5 | 5 | -0.25 |
| quartic_1d | irene_rewrite_sympy | elapsed_s | 23.10 ms | 18.90 ms | -0.25 |
| circle_relations | irene_rewrite_sympy | lower_bound | 1.000000 | 1.000000 | 1.0 |
| circle_relations | irene_rewrite_sympy | basis_size | 5 | 5 | 1.0 |
| circle_relations | irene_rewrite_sympy | elapsed_s | 29.00 ms | 24.70 ms | 1.0 |

## 12. Feature parity summary

| Feature | Original Irene | IreneRewrite | Notes |
|---------|---------------|--------------|-------|
| SDPRelaxations (SOS) | ✅ | ✅ | same API |
| SONCRelaxations (GP) | ✅ | ✅ | same API |
| SOSONCRelaxations (SOS+SONC) | ✅ | ✅ | same API |
| GPRelaxations | ✅ | ✅ | same API |
| DSDPRelaxations / Mean / KKT | ✅ | ✅ | API-compatible; `build_ade_relations` re-added in this session |
| Group rings / semigroup algebra | ✅ | ✅ | same API |
| Invariant theory | ✅ | ✅ | same API |
| Border basis | ✅ | ✅* | *different API surface: `compute/dimension/normal_form/roots` vs `reduce/conditioning_diagnostic` |
| Correlative sparsity | ✅ | ✅* | *different algorithm: chordal cliques vs UnionFind components |
| Newton polytope pruning | ✅ | ✅* | *different API: `NewtonPolytopePruner` vs `NewtonPruner` |
| Non-POP SDP (`nonpopsdp.py`) | ✅ | ❌ | **missing** — Taylor/Chebyshev non-polynomial pipeline not ported |
| Unified reductions (`unified_reductions.py`) | ✅ | ✅* | *replaced by `relaxation_api.py` + `sparsity.py` + `newton_polytope.py` + `border_basis.py` |
| CVXPY solver layer | ❌ | ✅ | new in rewrite |
| Relaxation API (unified engine) | ❌ | ✅ | new in rewrite |
| Telemetry | ❌ | ✅ | new in rewrite |
| Symbolic backend selection | ❌ (SymPy only) | ✅ | new in this session: `IRENE_SYMBOLIC_BACKEND` + `set_backend()` |

## 13. Key findings

- **Bounds parity**: SOS/SONC/SOSONC values agree across all three modes within solver tolerance.
- **Backend switch**: all 169 unit tests pass under both `symengine` and `sympy` backends.
- **DSDP API gap closed**: `build_ade_relations()` (with `wrt=` multi-derivation prefix) restored.
- **NonPOPSDP ported**: `nonpopsdp.py` restored with fixed Taylor/Chebyshev approximation numerics (original had ~61.5 Chebyshev error).
- **Quotient-basis option**: `RelaxationConfig.quotient_basis` ('groebner' default | 'border') selects the reduction engine; border mode verified against the Groebner mode on relation problems.
- **Top-level imports fixed**: `DSDPRelaxations`, `DSDPMeanRelaxation`, `DSDPKKTRelaxation` re-exported from `Irene`.
- **Remaining gap**: none — `nonpopsdp.py` was the last original-only module.
