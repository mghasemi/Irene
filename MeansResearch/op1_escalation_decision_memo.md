# OP1 Escalation Decision Memo (d=6, n>=4, L-C1)

Date: 2026-04-26  
Scope: Resolve escalation path for deferred d=6, n>=4 SOS/SDP slices without overclaiming theorem-level negatives.

## 1) Decision Summary

We classify the current d=6, n>=4 L-C1 state as a reproducible solver-ceiling deferment under the baseline interior-point stack (CVXOPT -> SDPA -> CSDP), not as mathematical non-membership. The recommended next path is a staged escalation campaign with strict acceptance gates and budget ceilings.

Primary decision:
- Keep manuscript framing as computationally deferred and non-negating.
- Execute one additional solver-generation escalation cycle aimed at producing at least one non-timeout outcome on a nondegenerate case.
- If no gate is met, freeze OP1 as unresolved and proceed with scope-limited publication package (PC1+PC2 core, PC3 exploratory).

## 2) Baseline Evidence Used

Current baseline (already completed):
- d=6, n=4 template probes: timeout-dominated up to 1800s in tested slices.
- d=6, n=5 pass1: 18/18 timeouts at 600s per solver across CVXOPT, SDPA, CSDP.
- d=6, n=5 pass2: 18/18 timeouts at 1200s per solver across CVXOPT, SDPA, CSDP.
- No successful certificates and no hard infeasibility certificates in nondegenerate escalation slices.

Artifact anchors:
- MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass1.jsonl
- MeansResearch/results/phase3_pilot_summary_d6_lc1_escalation_pass1.md
- MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_uniform.jsonl
- MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_boundary.jsonl
- MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_mixed.jsonl
- MeansResearch/results/phase3_pilot_summary_d6_lc1_escalation_pass2.md

## 3) Escalation Compute Path (Decision)

### Stage E1: Stronger SDP backend trial

Goal:
- Replace or augment interior-point baseline with at least one stronger backend (priority order: MOSEK, then SDPA-GMP, then DSDP) while preserving benchmark comparability.

Execution rules:
- Keep same nondegenerate template slices (uniform/boundary/mixed) and same p-grid where possible.
- Preserve JSONL schema parity (include solver attempt traces, timeout flags, runtime fields).
- Start with a pilot subset (6 representative nondegenerate cases) before full sweep.

### Stage E2: Structure-aware reformulation trial

Trigger:
- Run only if E1 fails to produce any non-timeout outcomes.

Candidate changes:
- Symmetry/block reduction or sparsity-exploiting reformulations.
- Scaled/normalized coefficient preprocessing with unchanged mathematical semantics.
- Warm-start or continuation strategy where backend supports it.

Execution rules:
- Record transform/reformulation metadata per case.
- Keep one-to-one mapping back to original instance IDs for comparability.

### Stage E3: Freeze-and-package fallback

Trigger:
- E1 and E2 fail all acceptance gates.

Decision:
- Freeze OP1 as unresolved computational deferment.
- Publish with explicit solver-ceiling protocol and caveat language.
- Defer theorem-level resolution to follow-up compute infrastructure work.

## 4) Budget Assumptions

Assumptions for one escalation cycle:
- Per-solver timeout cap candidate: 1800s to 2400s (upper practical bound for routine runs).
- Pilot size: 6 nondegenerate cases (2 per template family).
- Full-sweep size (if pilot shows signal): 18 nondegenerate cases.
- Maximum solvers in sequence: 3 per cycle to control combinatorial growth.
- Maximum cycle walltime target: 24 to 36 compute-hours equivalent for full sweep.

Operational limits:
- Do not expand p-grid or template families during escalation cycle.
- Do not co-mingle methodological changes and solver changes in the same run unless explicitly tagged.

## 5) Acceptance Criteria (First Non-Timeout Gate)

Minimum gate to keep OP1 active as a live escalation track:
- At least one non-timeout outcome on a nondegenerate d=6, n>=4 case under escalated tooling.

Preferred gate:
- At least one certificate-quality outcome (either feasible decomposition or explicit hard fail) with reproducible rerun.

Escalation success threshold for broader continuation:
- At least 2 independent non-timeout outcomes across different template families or p-values, both reproducible on rerun.

Failure-to-advance rule:
- If pilot subset yields zero non-timeout outcomes, terminate cycle and move to E3 freeze-and-package.

## 6) Reproducibility and Logging Requirements

Mandatory logging fields per attempt:
- instance_id, template_family, p_value, solver_name, timeout_seconds, runtime_seconds, exit_status,
  result_class (success/timeout/hard_fail), and attempt_order.

Required outputs:
- Raw JSONL attempt logs.
- One markdown summary per cycle with counts by solver and template.
- One short decision note stating whether acceptance gates were met.

## 7) Manuscript Policy While OP1 Is Open

- Keep language as "computationally deferred under current solver stack".
- Avoid impossibility phrasing for d=6, n>=4.
- Treat solver-ceiling evidence as method-capability boundary, not cone-membership refutation.

## 8) Immediate Next Action

Run E1 pilot with one stronger backend integrated into solver sequence while preserving current artifact schema. If the pilot has zero non-timeout outcomes, close OP1 cycle with freeze-and-package decision and no further routine reruns.
