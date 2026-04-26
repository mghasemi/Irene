# OP2 Prototype Plan: Support-Sensitive SONC Criterion

Date: 2026-04-26  
Scope: Prototype a support-sensitive criterion that explains persistent SONC failures on p in {1,2} slices while GP remains feasible on the same tested nondegenerate families.

## 1) Prototype Objective

Construct a criterion template that predicts SONC behavior from support geometry descriptors rather than degree alone, and evaluate whether it separates:
- SONC-feasible slices, and
- robust SONC-failure slices (especially p in {1,2})
within the current tested families.

The prototype is explanatory and computationally falsifiable; it is not yet a theorem-level characterization.

## 2) Baseline Evidence for Prototype Design

Observed pattern to explain:
- On tested nondegenerate d in {4,5}, n in {3,4} grids, GP is broadly feasible while SONC is support-sensitive.
- SONC failures on p in {1,2} are robust under local/global solver modes and tightened tolerances.
- d=6 structural SONC diagnostics also show configuration-robust failure persistence for tracked failed slices.

Primary evidence artifacts:
- MeansResearch/results/phase3_pilot_summary_clean.md
- MeansResearch/results/phase3_pilot_summary_clean_d5.md
- MeansResearch/results/phase3_sonc_diagnostics_summary.md
- MeansResearch/results/phase3_sonc_diagnostics_summary_d5.md
- MeansResearch/results/phase3_sonc_diagnostics_d6_structural.md

## 3) Criterion Prototype Template

Define a score or rule set over support descriptors phi(support) and classify each instance into:
- class F: SONC-feasible expected,
- class R: robust SONC-failure expected,
- class U: undecided/near-boundary.

Candidate descriptor blocks for phi(support):
- simplex-cover compatibility of negative-term supports,
- barycentric deficit of candidate circuit representations,
- overlap and multiplicity structure among tail supports,
- distance-to-circuit envelope under fixed monomial basis,
- optional transform sensitivity tag for known boundary families.

Prototype form options:
- weighted score threshold,
- decision-list rule set,
- two-stage rule: hard exclusion tests then soft score.

## 4) Minimal Data Protocol

Unit of evaluation:
- one benchmark instance with fixed degree, n, p, template family, and support metadata.

Required labels:
- SONC outcome class under diagnostic protocol (success or robust failure),
- GP feasibility flag on same instance,
- robustness flag across local/global and tolerance diagnostics.

Split policy:
- Development slice: d in {4,5}, n in {3,4} nondegenerate cases.
- Stress slice: d=6 structural SONC diagnostics as out-of-slice robustness check.

## 5) Validation Plan

Validation V1 (separation quality on development slice):
- Measure how well prototype separates SONC-feasible vs robust SONC-failure groups.
- Report confusion table and per-template breakdown.

Validation V2 (robustness to diagnostic configuration):
- Confirm predicted robust-failure class aligns with local/global and tightened-tolerance persistence.

Validation V3 (cross-slice sanity check):
- Apply prototype to d=6 structural diagnostic artifacts and report agreement/disagreement patterns.

## 6) Acceptance Gates for OP2 Progression

Gate G1 (minimum viability):
- Prototype must recover a nontrivial separation signal on development slice (better than degree-only baseline).

Gate G2 (robustness consistency):
- Predicted robust-failure class must align with observed diagnostic-persistent failures in the majority of tracked p in {1,2} cases.

Gate G3 (artifact reproducibility):
- Re-running prototype evaluation with same inputs must reproduce identical classification outputs.

Failure rule:
- If G1 fails, keep OP2 open and revise descriptor set before any manuscript-level promotion.

## 7) Deliverables and Artifact-First Outputs

Required deliverables for OP2 prototype cycle:
- One prototype specification note (this file).
- One results table artifact summarizing predicted vs observed classes by template and p.
- One short interpretation memo stating what the prototype explains and where it fails.

Suggested output filenames:
- MeansResearch/results/op2_prototype_classification_table.csv
- MeansResearch/results/op2_prototype_validation_summary.md

## 8) Manuscript and Ledger Policy While OP2 Is Open

- Keep language as support-sensitive conjectural program, not theorem-level criterion.
- Continue separating confirmed empirical behavior from explanatory prototype claims.
- Treat prototype outputs as evidence-weighting tools until formal proof dependencies are established.

## 9) Immediate Next Action

Instantiate a first phi(support) descriptor set on the frozen d in {4,5} nondegenerate slice, produce predicted vs observed class table, and evaluate G1-G3 before any claim-strength upgrade.
