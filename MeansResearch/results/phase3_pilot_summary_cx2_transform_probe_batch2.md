# Phase 3 CX-2 Transform Probe Summary

Total rows: 32

| case_id | family_coeff | tolerance | transform | sonc_status | sonc_lb | gp_status | gp_lb |
|---|---:|---:|---|---|---:|---|---:|
| Q-c4 | 4 | 1e-06 | pre | success | -1 | success | -1 |
| Q-c4 | 4 | 1e-08 | pre | success | -1 | success | -1 |
| Q-c3p95 | 3.95 | 1e-06 | pre | success | -1 | success | -1 |
| Q-c3p95 | 3.95 | 1e-08 | pre | success | -1 | success | -1 |
| Q-c3p9 | 3.9 | 1e-06 | pre | success | -1 | success | -1 |
| Q-c3p9 | 3.9 | 1e-08 | pre | success | -1 | success | -1 |
| Q-c3p8 | 3.8 | 1e-06 | pre | success | -1 | success | -1 |
| Q-c3p8 | 3.8 | 1e-08 | pre | success | -1 | success | -1 |
| Rhat-c2 | 2 | 1e-06 | pre | fail |  | fail |  |
| Rhat-c2 | 2 | 1e-08 | pre | fail |  | fail |  |
| Rhat-c2-transform | 2 | 1e-06 | post | success | -1 | fail |  |
| Rhat-c2-transform | 2 | 1e-08 | post | success | -1 | fail |  |
| Rhat-c1p98 | 1.98 | 1e-06 | pre | fail |  | fail |  |
| Rhat-c1p98 | 1.98 | 1e-08 | pre | fail |  | fail |  |
| Rhat-c1p98-transform | 1.98 | 1e-06 | post | fail |  | fail |  |
| Rhat-c1p98-transform | 1.98 | 1e-08 | post | fail |  | fail |  |
| Rhat-c1p95 | 1.95 | 1e-06 | pre | fail |  | fail |  |
| Rhat-c1p95 | 1.95 | 1e-08 | pre | fail |  | fail |  |
| Rhat-c1p95-transform | 1.95 | 1e-06 | post | fail |  | fail |  |
| Rhat-c1p95-transform | 1.95 | 1e-08 | post | fail |  | fail |  |
| Rhat-c1p9 | 1.9 | 1e-06 | pre | fail |  | fail |  |
| Rhat-c1p9 | 1.9 | 1e-08 | pre | fail |  | fail |  |
| Rhat-c1p9-transform | 1.9 | 1e-06 | post | fail |  | fail |  |
| Rhat-c1p9-transform | 1.9 | 1e-08 | post | fail |  | fail |  |
| Rhat-c1p85 | 1.85 | 1e-06 | pre | fail |  | fail |  |
| Rhat-c1p85 | 1.85 | 1e-08 | pre | fail |  | fail |  |
| Rhat-c1p85-transform | 1.85 | 1e-06 | post | fail |  | fail |  |
| Rhat-c1p85-transform | 1.85 | 1e-08 | post | fail |  | fail |  |
| Rhat-c1p8 | 1.8 | 1e-06 | pre | fail |  | fail |  |
| Rhat-c1p8 | 1.8 | 1e-08 | pre | fail |  | fail |  |
| Rhat-c1p8-transform | 1.8 | 1e-06 | post | fail |  | fail |  |
| Rhat-c1p8-transform | 1.8 | 1e-08 | post | fail |  | fail |  |

## Notes

- `family_coeff` is the tail coefficient swept in each family.
- `pre` vs `post` compares solver behavior before/after the linear transform used in the manuscript identity.
- SONC/GP statuses are treated as stress diagnostics; failures are recorded with raw solver messages in JSONL.
