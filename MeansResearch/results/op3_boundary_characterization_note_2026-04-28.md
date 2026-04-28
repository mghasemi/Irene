# OP3 Boundary Characterization Note (Session 4)

Date: 2026-04-28
Scope: Mathematical characterization candidates for the Robinson-hat point-level SONC boundary at c=2.0.

## Evidence Lock (What Is Treated as Fixed)

Primary evidence artifacts:
- MeansResearch/results/op3_local_neighborhood_table.csv
- MeansResearch/results/op3_local_neighborhood_summary.md
- MeansResearch/results/op3_wide_coeff_scan_table.csv
- MeansResearch/results/op3_wide_coeff_scan_summary.md
- MeansResearch/theorem_ledger.md

Fixed empirical facts used in this note:
1. Local 7x7 sweep around (c,t)=(2.0,1.0): exactly one robust_success point, at center only.
2. All non-center local points are robust_fail; no unstable points.
3. Wide coefficient scan at t=1.0 over c in [1.5,2.5] (discrete tested set): robust_success occurs only at c=2.0.
4. Diagnostic reruns do not flip status classes in tested points (stability gate already passed).

Interpretation boundary condition:
- Use the above only as slice-local structure evidence for OP3; do not claim global transform invariance or global impossibility.

## Working Mathematical Frame

Let f_{c,t} denote the transformed Robinson-hat family element evaluated in the OP3 pipeline.
Let S(c,t) be a SONC-feasibility indicator on the tested formulation.
Empirical support indicates:
- S(c,t)=1 at (2.0,1.0),
- S(c,t)=0 at all other tested local/wide sampled points.

The goal is to propose falsifiable criteria explaining why (2.0,1.0) behaves as a point-level sharp support boundary.

## Candidate Criterion C1 (Balanced Circuit Equality at Center)

Statement (candidate):
- At (c,t)=(2.0,1.0), at least one critical circuit inequality in the SONC decomposition is attained at equality.
- For tested perturbations in c or t, that equality becomes strict in the infeasible direction.

Why this fits evidence:
- Unique robust_success at center and immediate robust_fail under both coefficient and transform perturbation is consistent with equality-to-strict transition.

What to verify next:
1. Extract/construct candidate circuit terms for f_{c,t} near center.
2. Compute symbolic or high-precision numeric circuit-number margins at:
   - center,
   - c=2.0 with t in {0.97,1.03},
   - t=1.0 with c in {1.9975,2.0025}.
3. Check sign pattern: margin(center) approximately 0 and margins(off-center) negative.

Falsifier:
- If any off-center robust_fail point still satisfies equality/positive margin for all critical circuits, C1 is rejected.

## Candidate Criterion C2 (Unique Support Alignment Under T*)

Statement (candidate):
- The transformed support at t=1.0 has a unique affine/barycentric alignment that admits SONC certification only at c=2.0.
- Small t perturbations destroy this alignment even when c remains fixed.

Why this fits evidence:
- At c=2.0, only t=1.0 succeeds while nearby t values fail robustly.

What to verify next:
1. Build support-set geometry snapshots for t in {0.97,1.0,1.03} at c=2.0.
2. Compare barycentric coordinates of candidate tail exponents relative to relevant simplex faces.
3. Check whether center has a uniqueness or exactness property absent in neighbors.

Falsifier:
- If geometric alignment descriptors remain equivalent between center and neighboring t values, C2 is rejected.

## Candidate Criterion C3 (Piecewise Constant Feasibility Map with Isolated Atom)

Statement (candidate):
- On the tested formulation/slice, SONC feasibility behaves as an indicator map with an isolated feasible atom at (2.0,1.0), not a feasible neighborhood.

Why this fits evidence:
- H1 fail with H2/H3 pass is exactly this pattern on sampled grids.

What to verify next:
1. Add one denser micro-grid only for proof support (not exploratory rerun campaign):
   - c in {1.999, 2.000, 2.001},
   - t in {0.995, 1.000, 1.005}.
2. Confirm no additional robust_success appears.

Falsifier:
- Discovery of any non-center robust_success in this micro-grid rejects the isolated-atom claim.

## Dependency Map for Proof-Stage Follow-up

Dependency D1: Circuit extraction and margin computation utility for f_{c,t}.
Dependency D2: Support geometry comparison routine across transform path t.
Dependency D3: High-precision check protocol to separate numeric artifacts from structural margins.

Criterion dependencies:
- C1 depends on D1 and D3.
- C2 depends on D2 and D3.
- C3 depends on D3 and minimal targeted rerun protocol.

## Session 4 Gate Evaluation

Gate condition:
- At least one concrete criterion candidate is precise enough to verify or falsify.

Result:
- PASS.
- C1, C2, and C3 are all explicitly falsifiable with concrete next checks and rejection conditions.

## Recommended Immediate Next Action

Proceed to Session 5 integration pass and reflect OP3 status as:
- empirically point-sharp boundary on tested slice,
- mathematically open with active criterion candidates C1-C3,
- no additional broad perturbation sweeps required at this stage.

## 2026-04-28 C3 Micro-Grid Verification Update

Executed targeted C3 verification micro-grid (proof-support only):
- c in {1.999, 2.000, 2.001}
- t in {0.995, 1.000, 1.005}
- total points: 9

New artifacts:
- MeansResearch/results/op3_c3_microgrid_sweep.jsonl
- MeansResearch/results/op3_c3_microgrid_table.csv
- MeansResearch/results/op3_c3_microgrid_summary.md

Observed outcome:
- robust_success: 1/9 (center only at c=2.000, t=1.000)
- robust_fail: 8/9 (all non-center points)
- unstable/skipped: 0/9

C3 status after micro-grid:
- Not falsified.
- Empirical support strengthened for an isolated feasible atom on the tested neighborhood.

Implication for next proof work:
- De-prioritize additional neighborhood sweeps.
- Prioritize C1/C2 verification via circuit-margin and support-alignment analysis.
