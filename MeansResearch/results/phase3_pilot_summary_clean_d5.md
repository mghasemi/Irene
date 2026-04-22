# Phase 3 Pilot Summary

Filter: d=5, n in [3, 4], tolerances=[1e-06, 1e-08]
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
Gap range (gp - sonc): min=-4.756e-09, max=2.281e-09

| id | sonc | gp | gp-sonc |
|---|---:|---:|---:|
| L-C2-d5-n3-p0-a4_3_3-tol1e-06 | -1.00000000013 | -1.00000000161 | -1.481e-09 |
| L-C2-d5-n3-p0-a4_3_3-tol1e-08 | -1.0000000014 | -1.00000000161 | -2.145e-10 |
| L-C2-d5-n3-p5-a4_3_3-tol1e-06 | -1.0000000001 | -1.00000000002 | 7.588e-11 |
| L-C2-d5-n3-p5-a4_3_3-tol1e-08 | -1.00000000012 | -1.00000000002 | 9.644e-11 |
| L-C2-d5-n3-p0-a9_1_0-tol1e-06 | -1.00000000106 | -1.00000000488 | -3.822e-09 |
| L-C2-d5-n3-p0-a9_1_0-tol1e-08 | -1.00000000013 | -1.00000000488 | -4.756e-09 |
| L-C2-d5-n3-p5-a9_1_0-tol1e-06 | -1.0000000001 | -1.00000000046 | -3.676e-10 |
| L-C2-d5-n3-p5-a9_1_0-tol1e-08 | -1.00000000232 | -1.00000000046 | 1.857e-09 |
| L-C2-d5-n3-p0-a5_3_2-tol1e-06 | -1.0000000002 | -1.00000000004 | 1.665e-10 |
| L-C2-d5-n3-p0-a5_3_2-tol1e-08 | -1.00000000232 | -1.00000000004 | 2.281e-09 |

## Failure Concentration by Alpha

### SONCRelaxations

| alpha | fail_count |
|---|---:|
| (4, 3, 3) | 4 |
| (9, 1, 0) | 4 |
| (5, 3, 2) | 4 |
| (3, 3, 2, 2) | 4 |
| (9, 1, 0, 0) | 4 |
| (5, 3, 2, 0) | 4 |

