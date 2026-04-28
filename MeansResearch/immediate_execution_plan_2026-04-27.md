# Immediate Execution Plan (5 Sessions)

Date: 27 April 2026
Scope: OP1 backend escalation, OP2 support-geometry criterion, OP3 boundary characterization, and integration sync.

## Session 1: OP1 Backend Gating and Readiness

Objective:
- Confirm availability and install path for one stronger SDP backend in priority order:
  1. MOSEK
  2. SDPA-GMP
  3. DSDP

Actions:
1. Lock the E1 pilot manifest to 6 representative nondegenerate cases.
2. Preserve existing artifact schema for strict comparability.
3. Record backend availability outcome with evidence.

Deliverables:
- Backend readiness note.
- Final 6-case pilot manifest.

Gate:
- Proceed only if at least one stronger backend is runnable, or explicitly record unavailability and continue with freeze policy logic.

## Session 2: OP1 E1 Pilot and Go/No-Go

Objective:
- Execute E1 with stronger backend integrated into solver sequence.

Actions:
1. Run pilot on fixed 6-case set.
2. Summarize outcomes by solver, template, and result class.
3. Evaluate first non-timeout gate.

Deliverables:
- Pilot JSONL artifacts.
- Pilot decision summary.

Gate:
- If zero non-timeout outcomes: freeze OP1 for this cycle.
- If at least one non-timeout outcome: schedule full sweep.

## Session 3: OP2 Prototype Instantiation (First Phi Set)

Objective:
- Build first support-geometry descriptor set and test separation quality.

Actions:
1. Instantiate phi(support) on frozen d=4/5 nondegenerate slice.
2. Produce predicted vs observed classification table.
3. Evaluate G1, G2, G3.

Deliverables:
- Classification table artifact.
- Validation summary memo.

Gate:
- If G1 fails, revise descriptor set before strengthening any claim language.

## Session 4: OP3 Mathematical Boundary Characterization

Objective:
- Advance theory-side interpretation of point-level boundary at c = 2.0.

Actions:
1. Use completed sweep evidence as fixed base.
2. Develop Newton-polytope and AM-GM style characterization path.
3. Draft precise, falsifiable criterion candidates.

Deliverables:
- Boundary-analysis note (1-2 pages).
- Candidate lemma or conjecture list with dependencies.

Gate:
- At least one concrete criterion candidate is stated clearly enough to verify or falsify.

## Session 5: Integration and Framing Consistency Pass

Objective:
- Synchronize plan, ledger, and manuscript language.

Actions:
1. Align statuses into three buckets:
  - confirmed
  - deferred
  - conjectural
2. Keep OP1 phrasing solver-ceiling framed if unresolved.
3. Update next-action entries to reflect latest decisions.

Deliverables:
- Consistent cross-document status framing.
- Updated immediate next-action lines.

Gate:
- No contradictions across planning, ledger, and decision memos.

## Execution Rule

1. If Session 2 fails OP1 gate, stop additional OP1 compute spending this week and prioritize OP2 and OP3 theory and packaging.
2. If Session 2 passes OP1 gate, run one additional OP1 compute block while keeping OP2 and OP3 in parallel.
