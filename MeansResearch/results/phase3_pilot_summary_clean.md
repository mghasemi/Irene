# Phase 3 Pilot Summary

Filter: d=4, n in [3, 4], tolerances=[1e-06, 1e-08]
Total records: 192

## Method Counts

| Method | Records |
|---|---:|
| GPRelaxations | 64 |
| SDPRelaxations | 64 |
| SONCRelaxations | 64 |

## Status by Method

| Method | success | inconclusive | fail | infeasible |
|---|---:|---:|---:|---:|
| GPRelaxations | 48 | 16 | 0 | 0 |
| SDPRelaxations | 48 | 16 | 0 | 0 |
| SONCRelaxations | 24 | 16 | 24 | 0 |

## SONC vs GP (paired successful cases)

Paired comparisons: 24
Gap range (gp - sonc): min=-3.451e-09, max=2.101e-09

| id | sonc | gp | gp-sonc |
|---|---:|---:|---:|
| L-C2-d4-n3-p0-a3_3_2-tol1e-06 | -1.00000000015 | -1.00000000096 | -8.099e-10 |
| L-C2-d4-n3-p0-a3_3_2-tol1e-08 | -1.00000000154 | -1.00000000096 | 5.851e-10 |
| L-C2-d4-n3-p4-a3_3_2-tol1e-06 | -1.00000000002 | -1.00000000102 | -1.007e-09 |
| L-C2-d4-n3-p4-a3_3_2-tol1e-08 | -1.00000000002 | -1.00000000102 | -1.005e-09 |
| L-C2-d4-n3-p0-a7_1_0-tol1e-06 | -1.00000000068 | -1.00000000353 | -2.851e-09 |
| L-C2-d4-n3-p0-a7_1_0-tol1e-08 | -1.00000000008 | -1.00000000353 | -3.451e-09 |
| L-C2-d4-n3-p4-a7_1_0-tol1e-06 | -1.00000000009 | -1.00000000032 | -2.283e-10 |
| L-C2-d4-n3-p4-a7_1_0-tol1e-08 | -1.00000000241 | -1.00000000032 | 2.089e-09 |
| L-C2-d4-n3-p0-a4_2_2-tol1e-06 | -1.00000000018 | -1.000000002 | -1.829e-09 |
| L-C2-d4-n3-p0-a4_2_2-tol1e-08 | -1.00000000204 | -1.000000002 | 3.150e-11 |

## Failure Concentration by Alpha

### SONCRelaxations

| alpha | fail_count |
|---|---:|
| (3, 3, 2) | 4 |
| (7, 1, 0) | 4 |
| (4, 2, 2) | 4 |
| (2, 2, 2, 2) | 4 |
| (7, 1, 0, 0) | 4 |
| (4, 2, 2, 0) | 4 |

