# Phase 3 CX-2 Transform Probe Summary

Total rows: 12

| case_id | tolerance | transform | sonc_status | sonc_lb | gp_status | gp_lb |
|---|---:|---|---|---:|---|---:|
| Q-base | 1e-06 | pre | success | -1 | success | -1 |
| Q-base | 1e-08 | pre | success | -1 | success | -1 |
| Rhat-base | 1e-06 | pre | fail |  | fail |  |
| Rhat-base | 1e-08 | pre | fail |  | fail |  |
| Rhat-transform | 1e-06 | post | success | -1 | fail |  |
| Rhat-transform | 1e-08 | post | success | -1 | fail |  |
| Q-perturbed | 1e-06 | pre | success | -1 | success | -1 |
| Q-perturbed | 1e-08 | pre | success | -1 | success | -1 |
| Rhat-perturbed | 1e-06 | pre | fail |  | fail |  |
| Rhat-perturbed | 1e-08 | pre | fail |  | fail |  |
| Rhat-perturbed-transform | 1e-06 | post | fail |  | fail |  |
| Rhat-perturbed-transform | 1e-08 | post | fail |  | fail |  |

## Notes

- `pre` vs `post` compares solver behavior before/after the linear transform used in the manuscript identity.
- SONC/GP statuses are treated as stress diagnostics; failures are recorded with raw solver messages in JSONL.
