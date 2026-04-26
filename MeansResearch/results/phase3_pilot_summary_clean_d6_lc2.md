# Phase 3 Pilot Summary

Filter: d=6, n in [3, 4], tolerances=[1e-06, 1e-08]
Total records: 384

## Method Counts

| Method | Records |
|---|---:|
| GPRelaxations | 192 |
| SONCRelaxations | 192 |

## Status by Method

| Method | success | inconclusive | timeout | fail | infeasible |
|---|---:|---:|---:|---:|---:|
| GPRelaxations | 140 | 48 | 0 | 4 | 0 |
| SONCRelaxations | 48 | 48 | 0 | 96 | 0 |

## SONC vs GP (paired successful cases)

Paired comparisons: 24
Gap range (gp - sonc): min=-4.238e-09, max=2.793e-09

| id | sonc | gp | gp-sonc |
|---|---:|---:|---:|
| L-C2-d6-n3-p0-a4_4_4-tol1e-06 | -1.00000000012 | -1.00000000029 | -1.681e-10 |
| L-C2-d6-n3-p0-a4_4_4-tol1e-08 | -1.00000000127 | -1.00000000029 | 9.843e-10 |
| L-C2-d6-n3-p6-a4_4_4-tol1e-06 | -1.00000000008 | -1.00000000075 | -6.689e-10 |
| L-C2-d6-n3-p6-a4_4_4-tol1e-08 | -1.00000000009 | -1.00000000075 | -6.617e-10 |
| L-C2-d6-n3-p0-a11_1_0-tol1e-06 | -1.0000000015 | -1.00000000448 | -2.977e-09 |
| L-C2-d6-n3-p0-a11_1_0-tol1e-08 | -1.00000000024 | -1.00000000448 | -4.238e-09 |
| L-C2-d6-n3-p6-a11_1_0-tol1e-06 | -1.0000000001 | -1.00000000062 | -5.172e-10 |
| L-C2-d6-n3-p6-a11_1_0-tol1e-08 | -1.00000000226 | -1.00000000062 | 1.643e-09 |
| L-C2-d6-n3-p0-a6_4_2-tol1e-06 | -1.00000000025 | -1.00000000008 | 1.760e-10 |
| L-C2-d6-n3-p0-a6_4_2-tol1e-08 | -1.00000000287 | -1.00000000008 | 2.793e-09 |

## Failure Concentration by Alpha

### GPRelaxations

| alpha | fail_count |
|---|---:|
| (3, 3, 3, 3) | 4 |

### SONCRelaxations

| alpha | fail_count |
|---|---:|
| (4, 4, 4) | 16 |
| (11, 1, 0) | 16 |
| (6, 4, 2) | 16 |
| (3, 3, 3, 3) | 16 |
| (11, 1, 0, 0) | 16 |
| (6, 4, 2, 0) | 16 |

