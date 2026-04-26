# SONC Diagnostics Summary

Input: MeansResearch/results/phase3_runs_clean_d6_lc2.jsonl
Output: MeansResearch/results/phase3_sonc_diagnostics_d6_lc2.jsonl
Filters: item=L-C2, d=6, n in [3,4]
Unique failed SONC cases rerun: 48

## Case Count by Template

| alpha_template | cases |
|---|---:|
| boundary | 16 |
| mixed | 16 |
| uniform | 16 |

## Status by Diagnostic Config

| config | success | fail |
|---|---:|---:|
| global_1e-10 | 0 | 48 |
| global_tol | 0 | 48 |
| local_1e-10 | 0 | 48 |
| local_tol | 0 | 48 |

## Status by Template

| alpha_template | success | fail |
|---|---:|---:|
| boundary | 0 | 64 |
| mixed | 0 | 64 |
| uniform | 0 | 64 |

## Template x Config

| alpha_template | config | success | fail |
|---|---|---:|---:|
| boundary | global_1e-10 | 0 | 16 |
| boundary | global_tol | 0 | 16 |
| boundary | local_1e-10 | 0 | 16 |
| boundary | local_tol | 0 | 16 |
| mixed | global_1e-10 | 0 | 16 |
| mixed | global_tol | 0 | 16 |
| mixed | local_1e-10 | 0 | 16 |
| mixed | local_tol | 0 | 16 |
| uniform | global_1e-10 | 0 | 16 |
| uniform | global_tol | 0 | 16 |
| uniform | local_1e-10 | 0 | 16 |
| uniform | local_tol | 0 | 16 |

