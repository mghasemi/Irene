# d=6, n=5 L-C1 Escalation Pass1 Summary

Input artifact: `MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass1.jsonl`

Total records: 24

## Final Status Counts

| status | count |
|---|---:|
| timeout | 18 |
| inconclusive | 6 |

## Status by Alpha Template

| alpha_template | timeout | inconclusive |
|---|---:|---:|
| uniform | 6 | 0 |
| boundary | 6 | 0 |
| mixed | 6 | 0 |
| sparse | 0 | 6 |

## Solver Attempt Outcomes (all attempted records)

| solver | timeout | success | fail |
|---|---:|---:|---:|
| CVXOPT | 18 | 0 | 0 |
| SDPA | 18 | 0 | 0 |
| CSDP | 18 | 0 | 0 |

## Interpretation

- All nondegenerate template cases (`uniform`, `boundary`, `mixed`) remain unresolved after pass1 and timed out under all three solvers at 600s per solver.
- `sparse` cases are structurally degenerate and remain inconclusive by construction; they are not informative for escalation.
- No solver achieved a successful solve on any nondegenerate case in pass1.

## Pass2 Promotion Guidance

Promote the 18 nondegenerate timeout cases (`uniform`, `boundary`, `mixed`) to pass2 at 1200s per solver.
