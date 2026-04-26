# OP3 Local-Neighborhood Test Plan (Robinson-hat CX-2)

Date: 2026-04-26  
Scope: Design a local coefficient and transform-neighborhood sweep around the Robinson-hat baseline post-transform SONC-feasible point, with reproducible artifact schema and stop/go gates.

## 1) Objective

Test whether the observed baseline post-transform SONC-feasible point at coefficient c=2.0 is:
- a single-point anomaly,
- part of a small local feasible neighborhood, or
- the boundary of a narrow transform-sensitive transition band.

This plan remains hypothesis-testing only and does not upgrade OP3 to theorem-level claims.

## 2) Baseline Context (Fixed)

Current anchored observations:
- Pre-transform Robinson-hat coefficient sweep c in {2.0, 1.98, 1.95, 1.9, 1.85, 1.8} is SONC/GP-infeasible.
- Post-transform SONC feasibility appears at baseline c=2.0 only in current tested set.
- Nearby tested coefficients remain post-transform SONC-infeasible.

Reference artifacts:
- MeansResearch/results/phase3_runs_clean_cx2_transform_probe.jsonl
- MeansResearch/results/phase3_pilot_summary_cx2_transform_probe.md
- MeansResearch/results/phase3_runs_clean_cx2_transform_probe_batch2.jsonl
- MeansResearch/results/phase3_pilot_summary_cx2_transform_probe_batch2.md

## 3) Neighborhood Design

### 3.1 Coefficient neighborhood around baseline

Primary coefficient grid (local, symmetric around 2.0):
- c in {1.9925, 1.9950, 1.9975, 2.0000, 2.0025, 2.0050, 2.0075}

Fallback refinement grid (if transition detected):
- step 0.001 around first observed status change interval.

### 3.2 Transform-path neighborhood

Let T* denote the current baseline transform used in CX-2 runs.
Define one-parameter path:
- T(t) = (1 - t) I + t T*
- t grid: {0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10}

Optional conditioning guard:
- reject matrices whose condition number exceeds predefined cap (record as skipped-conditioned).

### 3.3 Crossed local sweep

Evaluate crossed pairs (c, t) on the 7x7 grid (49 points), with baseline center (2.0, 1.0) included.

## 4) Execution Protocol

For each (c, t):
1. Build transformed Robinson-hat instance under fixed normalization used in prior CX-2 runs.
2. Run SONC and GP with current standard settings.
3. If SONC=fail on first pass, rerun diagnostics under:
- local_tol
- global_tol
- local_1e-10
- global_1e-10
4. Record final status class:
- robust_success (SONC success in base run),
- robust_fail (fail in all four diagnostics),
- unstable (mixed diagnostic outcomes),
- skipped_conditioned.

## 5) Artifact Schema

Primary row schema for local sweep JSONL:
- experiment_id
- coefficient_c
- transform_t
- transform_condition_number
- sonc_status_base
- gp_status_base
- sonc_status_local_tol
- sonc_status_global_tol
- sonc_status_local_1e-10
- sonc_status_global_1e-10
- final_class
- runtime_seconds

Required output artifacts:
- MeansResearch/results/op3_local_neighborhood_sweep.jsonl
- MeansResearch/results/op3_local_neighborhood_summary.md
- MeansResearch/results/op3_local_neighborhood_table.csv

## 6) Acceptance Gates

Gate H1 (local-neighborhood existence):
- PASS if there is at least one non-center point with robust_success near (2.0, 1.0).

Gate H2 (boundary detectability):
- PASS if both robust_success and robust_fail points appear in the crossed grid with at least one adjacency crossing.

Gate H3 (diagnostic stability):
- PASS if at least 90% of points are classified as robust_success or robust_fail (not unstable).

## 7) Decision Rules

If H1 FAIL:
- Treat baseline as isolated under current resolution; retain anomaly-only wording.

If H1 PASS and H2 PASS:
- Promote from anomaly wording to empirically localized boundary wording in manuscript discussion.

If H3 FAIL:
- Do not strengthen claims; prioritize numerical-conditioning review and rerun policy before interpretation.

## 8) Budget and Run Controls

- Max crossed grid: 49 points in primary pass.
- Diagnostics reruns only for SONC base failures to control budget.
- Per-point timeout inherits current CX-2 standards unless explicitly overridden in run manifest.
- Abort criterion: if first 20 points are all robust_fail with no status variation, pause and review grid placement.

## 9) Immediate Next Action

Implement the 7x7 local sweep generator and runner for Robinson-hat with the above schema, execute primary pass, and publish summary/table artifacts with H1-H3 outcomes.
