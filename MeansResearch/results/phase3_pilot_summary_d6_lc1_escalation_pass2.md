# d=6, n=5 L-C1 Escalation Pass2 Summary

Input artifacts:
- `MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_uniform.jsonl`
- `MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_boundary.jsonl`
- `MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_mixed.jsonl`

Total records: 18 (nondegenerate templates only; sparse excluded as structurally degenerate)

Solver sequence: `CVXOPT → SDPA → CSDP`  
Per-solver timeout cap: **1200s**

## Final Status Counts

| status | count |
|---|---:|
| timeout | 18 |

## Status by Alpha Template

| alpha_template | timeout |
|---|---:|
| uniform | 6 |
| boundary | 6 |
| mixed | 6 |

## Solver Attempt Outcomes (all 18 records)

| solver | timeout | success | fail |
|---|---:|---:|---:|
| CVXOPT | 18 | 0 | 0 |
| SDPA | 18 | 0 | 0 |
| CSDP | 18 | 0 | 0 |

## Pass1 vs Pass2 Comparison

| pass | cap (s/solver) | timeout | success | fail |
|---|---:|---:|---:|---:|
| pass1 | 600 | 18 | 0 | 0 |
| pass2 | 1200 | 18 | 0 | 0 |

## Interpretation

- Doubling the per-solver timeout from 600s to 1200s produced no change in outcomes.  
- All three solvers (CVXOPT, SDPA, CSDP) timed out uniformly across all template types (uniform, boundary, mixed) and all tested p values (0, 1, 2, 3, 4, 6).
- No solver achieved a successful solve or a hard infeasibility certificate on any nondegenerate case in either pass.

## Verdict

**d=6, n=5 L-C1 is computationally intractable under standard interior-point SDP solvers** (CVXOPT, SDPA, CSDP) at timeout budgets up to 1200s per solver.

This result is consistent with:
- The d=6 n=4 deferment (batches 3–8, all-timeout at up to 1800s).
- The CX-1 tractability ceiling finding (L-C1 n≥5 infeasible under CVXOPT at all tested degrees).
- The d=6 structural SONC zero-recovery finding (L-C2 failed cases robust across all diagnostic configs).

The d=6, n=5 slice is formally deferred. Continuation would require either sparse-structure-exploiting or high-performance SDP solvers (e.g., MOSEK, DSDP, or SDPA-GMP).
