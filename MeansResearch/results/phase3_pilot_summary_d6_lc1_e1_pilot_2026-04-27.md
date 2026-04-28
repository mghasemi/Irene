# Session 2 E1 Pilot Summary (d=6, n=5, L-C1)

Date: 2026-04-27
Artifact: MeansResearch/results/phase3_runs_clean_d6_lc1_e1_pilot_2026-04-27.jsonl

## Scope

- Goal: Execute Session 2 E1 pilot using stronger-backend-enabled solver sequence.
- Solver sequence: DSDP -> CSDP -> SDPA -> CVXOPT.
- Timeout policy: 600 seconds per solver attempt.
- Intended evaluation set: 6 nondegenerate cases (templates uniform, boundary, mixed at p in {1,4}).

## Data Integrity Note

The first pass used offsets 1,4,7,10,13,16, which unintentionally included two sparse one-hot degenerate rows:
- L-C1-d6-n5-p1-a12_0_0_0_0-tol1e-06
- L-C1-d6-n5-p4-a12_0_0_0_0-tol1e-06

A corrective pass ran offsets 19 and 22 to include the intended mixed nondegenerate cases.

Final file composition:
- 8 total rows in artifact.
- 6 intended nondegenerate rows used for gate decision.
- 2 extra sparse degenerate rows retained as provenance.

## Intended 6-Case Nondegenerate Set

1. L-C1-d6-n5-p1-a3_3_2_2_2-tol1e-06 (uniform)
2. L-C1-d6-n5-p4-a3_3_2_2_2-tol1e-06 (uniform)
3. L-C1-d6-n5-p1-a11_1_0_0_0-tol1e-06 (boundary)
4. L-C1-d6-n5-p4-a11_1_0_0_0-tol1e-06 (boundary)
5. L-C1-d6-n5-p1-a6_4_2_0_0-tol1e-06 (mixed)
6. L-C1-d6-n5-p4-a6_4_2_0_0-tol1e-06 (mixed)

## Results on Intended 6-Case Set

Status counts:
- timeout: 6
- success: 0
- fail: 0
- inconclusive: 0

Solver-attempt outcomes (across 6 cases):
- DSDP timeout: 6
- CSDP timeout: 6
- SDPA timeout: 6
- CVXOPT timeout: 6

Runtime profile:
- Every intended case exhausted full fallback chain at 600s per solver.
- Effective per-case walltime contribution is timeout-dominated across all four solvers.

## Gate Evaluation

Session 2 gate rule:
- If zero non-timeout outcomes, freeze OP1 this cycle.

Observed:
- Non-timeout outcomes on intended nondegenerate set: 0/6.

Decision:
- Gate outcome: FAIL (no non-timeout signal).
- Action: freeze OP1 E1 cycle for this iteration and proceed with OP2/OP3 tracks while retaining this artifact as escalation evidence.

## Reproducibility Context

Environment assumptions used:
- CSDP available at /usr/bin/csdp.
- DSDP available via PATH prefix to /home/mehdi/miniforge3/envs/sage/bin.
- All solver names invoked through Irene-compatible sequence.
