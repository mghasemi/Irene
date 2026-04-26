# Theorem Ledger (Phase 2 Start)

Date initialized: 2026-04-03
Base draft: mean_polynomials_main.tex

Status vocabulary:
- proved
- draft proof
- conjecture
- counterexample candidate
- deferred

## Cone Theory and Core Mean-Polynomial Results

| ID | Statement (short) | Current status | Dependencies | Primary source | Next action |
|---|---|---|---|---|---|
| L-T1 | Power-mean monotonicity for p < q and induced nonnegativity route | proved | Jensen inequality setup | mean_polynomials_main.tex (Prop. prop:Monotonicity) | Normalize notation and tighten proof wording for final version. |
| L-T2 | D-T PSD criterion via AM-GM coefficient inequality | proved | L-T3 (as special case), weighted AM-GM | mean_polynomials_main.tex (Thm. thm:DT_PSD) | Keep as corollary-style presentation under circuit framework. |
| L-T3 | Circuit PSD criterion via convex-combination structure | proved | Weighted AM-GM, convex combination assumptions | mean_polynomials_main.tex (Thm. thm:Circuit_PSD) | Add explicit corollary pointer to L-T2 in final manuscript numbering. |
| L-D1 | SONC cone definition \mathcal{C}_{n,2d} used in the manuscript | proved | L-T3 | mean_polynomials_main.tex (Def. def:SONC) | Keep as canonical definition and align phrasing with SONC literature citations. |
| L-T6 | SONC generation claim via M_{1,0}(Y,\lambda) family | proved | L-T3, L-D1, Remark \ref{rem:SONCinMEANS} in base draft | mean_polynomials_main.tex (Def. def:SONC and Rem. rem:SONCinMEANS) | No further action required; PSD_Mean removed. |

## SOS/SONC/Mean Inclusion-Separation Thread

| ID | Statement (short) | Current status | Dependencies | Primary source | Next action |
|---|---|---|---|---|---|
| L-T4 | M_{2d,0}(X,\alpha) is PSD D-T and SOBS | proved | L-T2, known SOBS result in GM01 | mean_polynomials_main.tex (Prop. prop:M_2d_0_SOBS) | Add compact proof bridge sentence from D-T to SOBS citation context. |
| L-T5 | M_{2d,p} is SOS for d in {1,2,3} and allowed p | proved | L-T4, Hilbert cases, computational confirmations | mean_polynomials_main.tex (Thm. thm:M_2d_p_SOS) | Keep computationally-assisted steps explicitly labeled with Irene citation provenance. |
| L-T7 | Choi-Lam/Robinson transformed forms as separating examples beyond SOSONC | proved | Transformation identity, SONC non-invariance under linear maps, supporting citations | mean_polynomials_main.tex (Prop. prop:ChoiLamRobinson and post-proof paragraph in Section s05) | No further action required; PSD_Mean removed. |

## GMP Linkage and Extended Objectives

| ID | Statement (short) | Current status | Dependencies | Primary source | Next action |
|---|---|---|---|---|---|
| L-G1 | Generalized moment/semigroup bridge integrated into mean-cone framework | deferred | Proposal objective O1, final section architecture | Proposal PDF + source-of-truth O1 | Explicitly deferred to follow-up paper; no content in current manuscript. |
| L-G2 | Differential-algebra/KKT dependency handling integrated with mean framework | deferred | Follow-up paper scope decision | source-of-truth decision 6 | Keep explicitly out of current manuscript; seed follow-up theorem list. |

## Computationally Testable Conjectures

| ID | Statement (short) | Current status | Dependencies | Planned computational hook | Next action |
|---|---|---|---|---|---|
| L-C1 | Extension of L-T5 to higher d beyond 3 | scoped conjecture (confirmed on nondegenerate d in {4,5}, n in {3,4}; confirmed on partial d=6, n=3; deferred/open on d=6, n>=4 under solver ceiling) | L-T5 patterns, sparse structure assumptions | SDPRelaxations.Decompose, SONCRelaxations.solve | Keep manuscript wording aligned with explicit scope boundaries and evidence pointers: confirmed artifacts are `phase3_pilot_summary_clean.md`, `phase3_pilot_summary_clean_d5.md`, `phase3_runs_clean_d6_lc1_batch1.jsonl`, `phase3_runs_clean_d6_lc1_batch2.jsonl`, `phase3_pilot_summary_clean_d6_lc1_batch2.md`; deferred/open artifacts are the d=6 n=4 batch summaries plus `phase3_pilot_summary_d6_lc1_escalation_pass2.md`. |
| L-C2 | Broad SONC membership behavior for M_{2d,p}(X,\alpha) families | conjecture | L-T3, Newton polytope structure | SONCRelaxations.solve, GPRelaxations.solve | Align with manuscript Conjecture~\ref{conj:B2_support_sensitive_sonc}: preserve support-sensitive (not degree-only) wording, keep the d=4/d=5 robust $p\in\{1,2\}$ failure diagnostics as slice-local evidence, and advance to theorem-level by proving a support-geometry criterion (or producing a certified counterexample) that explains SONC-feasible vs robust-failure support classes. |

## 2026-04-20 Computational Update

- L-C1 evidence pointer: `phase3_runs_clean.jsonl`, `phase3_runs_clean_d5.jsonl`, `phase3_pilot_summary_clean.md`, and `phase3_pilot_summary_clean_d5.md` record 48/48 successful SDP SOS decompositions on the nondegenerate $d=4$ grid and 48/48 successful SDP SOS decompositions on the nondegenerate $d=5$ grid (both with $n \in \{3,4\}$); one-hot $\alpha$ cases are tracked as structurally degenerate and therefore inconclusive rather than failed.
- L-C2 evidence pointer: `phase3_pilot_summary_clean.md` and `phase3_pilot_summary_clean_d5.md` show the same pattern at both $d=4$ and $d=5$: GP success on all 48 nondegenerate pilot cases but SONC success on only 24 of them.
- SONC diagnostic pointer: `phase3_sonc_diagnostics.jsonl`, `phase3_sonc_diagnostics_summary.md`, `phase3_sonc_diagnostics_d5.jsonl`, and `phase3_sonc_diagnostics_summary_d5.md` show the same robustness pattern at both $d=4$ and $d=5$: all 24 nondegenerate SONC failures persist under local/global solves and under tighter error bound $10^{-10}$, so they should be treated as structural negatives for the current formulation rather than tuning artifacts.

## 2026-04-26 Computational Update (d=6, n≥4 L-C1 Intractability)

- **d=6, n=3**: All L-C1 uniform-grid cases succeed under CVXOPT (batches 1–2). Conjecture confirmed at this slice.
- **d=6, n=4**: All template types (uniform, boundary, mixed) timeout at budgets up to 1800s under CVXOPT. Deferred (batches 3–8).
- **d=6, n=5 escalation**: Two-pass escalation with solver sequence CVXOPT→SDPA→CSDP produced 18/18 timeout at 600s/solver (pass1) and 18/18 timeout at 1200s/solver (pass2) across all nondegenerate templates (uniform, boundary, mixed) and p-values {0,1,2,3,4,6}. Zero success or hard-fail outcomes across both passes.
- **d=6 structural SONC**: 48 unique L-C2 failed cases across uniform/boundary/mixed templates for n∈{3,4} show zero recovery under all four alternate SONC diagnostic configs. Failures are configuration-robust.
- L-C1 d=6 n=5 evidence artifacts: `phase3_runs_clean_d6_lc1_escalation_pass1.jsonl`, `phase3_pilot_summary_d6_lc1_escalation_pass1.md`, `phase3_runs_clean_d6_lc1_escalation_pass2_uniform.jsonl`, `phase3_runs_clean_d6_lc1_escalation_pass2_boundary.jsonl`, `phase3_runs_clean_d6_lc1_escalation_pass2_mixed.jsonl`, `phase3_pilot_summary_d6_lc1_escalation_pass2.md`.
- **Interpretation**: The d=6 n≥4 L-C1 slice is **computationally intractable** under standard interior-point SDP solvers within practical budgets. This is a solver-ceiling result, not a mathematical negative — the conjecture remains open at d=6 n≥4. Continuation requires MOSEK, DSDP, or SDPA-GMP.

## 2026-04-26 d=6 L-C2 Clean-Pilot Refresh and Cross-Degree Robustness Note

- Refreshed d=6 L-C2 artifacts: `phase3_runs_clean_d6_lc2.jsonl`, `phase3_pilot_summary_clean_d6_lc2.md`, `phase3_sonc_diagnostics_d6_lc2.jsonl`, `phase3_sonc_diagnostics_d6_lc2.md`.
- Clean-pilot totals on the refreshed d=6 slice ($n\in\{3,4\}$, tolerances $10^{-6},10^{-8}$): 384 method records; GP gives 140 successes, 48 inconclusive cases, and 4 fails; SONC gives 48 successes, 48 inconclusive cases, and 96 fails.
- Paired-success signal: 24 SONC/GP paired successes remain, with $\mathrm{gp}-\mathrm{sonc}$ gap range in $[-4.238\times 10^{-9}, 2.793\times 10^{-9}]$.
- Cross-degree SONC robustness pattern: all 24 failed nondegenerate SONC cases at d=4 remain failures under alternate diagnostics, the same 24/24 zero-recovery pattern holds at d=5, and the refreshed d=6 L-C2 slice extends this to 48/48 zero-recovery failures across uniform, boundary, and mixed templates.
- Interpretation: the support-sensitive SONC negatives are now cross-degree robust on the executed clean-pilot ladder and should be framed as formulation-structural evidence on the tested slices rather than as local/global solve or tolerance artifacts.

## Counterexample Tracking

| ID | Candidate | Current status | Why tracked | Next action |
|---|---|---|---|---|
| CX-1 | High-degree M_{2d,p} with sparse nonuniform \alpha potentially outside SOS | counterexample candidate | Could delimit the scope of L-T5 generalization | Generate symbolic families and test with SDP decomposition first. |
| CX-2 | Mean-of-linear-forms examples near SONC boundary under coordinate transforms | counterexample candidate | Could sharpen separation narrative around L-T7 | OP3 sweeps completed (7x7 local grid + wide coefficient scan c=1.5–2.5). Finding: c=2.0 is the unique SONC-success point; H1 FAIL, H2 PASS, H3 PASS. Point-level sharpness confirmed. Next action: mathematical characterisation (Newton-polytope/AM-GM analysis) rather than further solver perturbation. |

## 2026-04-22 CX-2 Update

- Evidence pointer (batch 1): `phase3_runs_clean_cx2_transform_probe.jsonl`, `phase3_pilot_summary_cx2_transform_probe.md`.
- Evidence pointer (batch 2): `phase3_runs_clean_cx2_transform_probe_batch2.jsonl`, `phase3_pilot_summary_cx2_transform_probe_batch2.md`.
- Observed computational pattern under current SONC/GP setup:
	- Choi-Lam tail sweep (`c\in\{4.0,3.95,3.9,3.8\}`) remains SONC+GP feasible.
	- Robinson-hat pre-transform sweep (`c\in\{2.0,1.98,1.95,1.9,1.85,1.8\}`) remains SONC+GP infeasible.
	- Robinson-hat post-transform SONC feasibility is observed only at baseline `c=2.0`; nearby coefficients in the tested sweep remain SONC-infeasible.
- Interpretation update: current evidence supports a **localized transform-sensitive SONC boundary effect** rather than a broad transform-invariance failure across nearby Robinson-hat perturbations.

## 2026-04-26 B3 Formalization Update

- Manuscript integration: B3 is now formalized as a **localized transform-boundary conjecture** (CX-2 linked) in the computational synthesis section.
- Scope guard: wording is explicitly family- and slice-local (Robinson-hat baseline neighborhood) and avoids global transform-invariance claims.
- Required follow-up for promotion: denser coefficient neighborhoods around baseline and transform-path perturbation sweeps with paired SONC/GP reproducibility checks.

## 2026-04-26 Phase 4 Integration Kickoff

### Ranked Publishable Claims (current ordering)

1. **PC1 (strongest):** on the tested nondegenerate d in {4,5}, n in {3,4} slice, SOS/SDP and GP are consistently feasible while SONC remains support-sensitive with robust p in {1,2} failures under diagnostic perturbations.
2. **PC2 (strong):** for L-C1 at d=6, n>=4, pass1/pass2 escalation establishes a reproducible solver-ceiling deferment criterion under CVXOPT->SDPA->CSDP, with explicit non-negation caveat at theorem level.
3. **PC3 (exploratory):** CX-2 indicates a localized transform-sensitive SONC boundary effect in the Robinson-hat track, suitable as a conjectural direction with targeted follow-up sweeps.

### Steering Fallback Direction

- If OP1 remains unresolved under current compute and solver access, prioritize a scope-limited publication package around PC1+PC2 and keep PC3 as a forward-looking conjectural thread.
- Fallback manuscript framing: emphasize robust empirical separation claims and reproducible deferment protocol; avoid overclaiming theorem-level impossibility.

### Next-Quarter Continuation Plan (Task #40)

1. **Primary path (theory-first):** promote B1 and B2 into sharpened theorem/conjecture text with explicit scope qualifiers and proof-dependency notes suitable for manuscript integration.
2. **Computational path (solver escalation):** reattempt deferred d=6, n>=4 L-C1 slices with stronger SDP tooling or structure-aware formulations, preserving JSONL comparability against the current baseline.
3. **Fallback path (scope-limited packaging):** if OP1 remains unresolved, package PC1+PC2 as the core publishable contribution and retain PC3/OP3 as explicit forward-work commitments.

#### Success Criteria for Next Quarter

- At least one of: (a) a theorem-level strengthening for B1/B2, or (b) a first non-timeout/nontrivial outcome on deferred slices under escalated tooling.
- Finalized manuscript framing that separates confirmed statements, deferred statements, and conjectural statements without scope ambiguity.

### Experiment Report and Reproducibility Close-out (Task #38)

#### Close-out Deliverables

1. Final experiment-report synthesis with three buckets: confirmed, weakened/deferred, and anomaly-driven follow-up.
2. Stable artifact pointers for all Phase 3 and escalation evidence used in claims PC1-PC3 and OP1-OP3.
3. Reproducible manuscript build path (`make pdf` in `MeansResearch/`) validated after each Phase 4 synthesis update.

#### Completion Record (2026-04-26)

- Manuscript close-out block added under Phase 4 integration section.
- Ranked publishable claims/open problems and continuation plan integrated in manuscript + ledger.
- Two-pass PDF build path validated through Makefile target.

#### Residual Risk Note

- d=6, n>=4 L-C1 remains a solver-ceiling deferment item; report framing must continue to avoid theorem-level impossibility language until stronger computational evidence is available.

### Ranked Open Problems (procedural priority)

1. **OP1 (highest):** resolve whether d=6, n>=4 L-C1 all-timeout behavior is purely a solver-ceiling artifact or indicates deeper structural complexity in SOS decomposition for the tested families.
2. **OP2 (high):** formulate a support-sensitive SONC membership criterion consistent with persistent p in {1,2} failures under local/global and tightened-tolerance diagnostics.
3. **OP3 (medium):** determine whether the observed CX-2 Robinson-hat baseline post-transform SONC-feasibility point extends to a provable local neighborhood.

## 2026-04-26 OP1 Decision Memo Update

- Decision memo artifact added: `MeansResearch/op1_escalation_decision_memo.md`.
- Memo defines staged escalation path (E1 backend trial, E2 structure-aware reformulation, E3 freeze-and-package fallback).
- Budget assumptions and first non-timeout acceptance gates are now explicit and can be used as stop/go criteria for future OP1 reruns.

## 2026-04-26 OP2 Prototype Plan Update

- Prototype-plan artifact added: `MeansResearch/op2_support_sensitive_sonc_prototype_plan.md`.
- Plan defines support-descriptor based criterion template, development/stress validation slices, and G1-G3 acceptance gates.
- OP2 remains conjectural-program level until prototype separation and reproducibility gates are met.

## 2026-04-26 OP2 Prototype Instantiation (phi/support v1)

- Instantiation artifacts added:
	- `MeansResearch/results/op2_prototype_classification_table.csv`
	- `MeansResearch/results/op2_prototype_validation_summary.md`
- Frozen slice used: d in {4,5}, nondegenerate rows only (48 records).
- Prototype-1 rule outcome: predicted vs observed accuracy = 48/48 (1.000) on the frozen slice; degree-only baseline = 24/48 (0.500).
- Gate status: G1 PASS, G2 PASS, G3 PASS under the prototype-plan criteria.
- Scope caveat retained: this is still an empirical prototype result and not a theorem-level criterion.

## 2026-04-26 OP3 Local-Neighborhood Plan Update

- Local-neighborhood test-plan artifact added: `MeansResearch/op3_local_neighborhood_test_plan.md`.
- Plan specifies crossed coefficient/transform neighborhood sweeps around Robinson-hat baseline, with reproducibility schema and H1-H3 acceptance gates.
- OP3 remains open pending execution artifacts for the local sweep and gate evaluation outcomes.

## 2026-04-26 OP3 Local-Neighborhood Execution Update

- Execution artifacts added:
	- `MeansResearch/results/op3_local_neighborhood_sweep.jsonl`
	- `MeansResearch/results/op3_local_neighborhood_table.csv`
	- `MeansResearch/results/op3_local_neighborhood_summary.md`
- Primary 7x7 sweep outcome at base tolerance 1e-8: 49 total points, with 1 robust_success point and 48 robust_fail points; no unstable or condition-skipped points occurred.
- Center point `(c=2.0, t=1.0)` remained SONC-feasible but GP-infeasible; every non-center grid point was robust_fail across all four SONC diagnostic configurations.
- Gate status: H1 FAIL, H2 PASS, H3 PASS.
- Interpretation update: the first local-neighborhood pass supports a sharply isolated SONC-positive center rather than a visibly extended nearby success region under the tested coefficient and transform perturbations.

## 2026-04-26 OP3 Wide-Coefficient Scan Update

- Wider coefficient scan at fixed t=1.0, c in {1.5, 1.6, 1.7, 1.8, 1.85, 1.9, 1.95, 1.98, 1.99, 1.995, 2.0, 2.005, 2.01, 2.02, 2.05, 2.1, 2.2, 2.5}: 18 points total.
- Execution artifacts:
	- `MeansResearch/results/op3_wide_coeff_scan_sweep.jsonl`
	- `MeansResearch/results/op3_wide_coeff_scan_table.csv`
	- `MeansResearch/results/op3_wide_coeff_scan_summary.md`
- Outcome: c=2.0 is the **unique** SONC-success point in the full tested range (c=1.5 to c=2.5); all 17 other values are robust-fail across all four SONC diagnostic configurations.
- Converging evidence: combining the 7x7 local grid (H1 FAIL) and the wide scan, the Robinson-hat post-transform SONC-feasibility is a **point-level** phenomenon at exactly c=2.0 under the baseline transform t=1.0.
- Manuscript update: B3 paragraph updated to include both OP3 sweep findings and point-level sharpness interpretation.
- Implication for B3 promotion path: further solver-perturbation sweeps are unlikely to be productive; the next required step is a mathematical characterisation of the certificate cone boundary at c=2.0 (Newton-polytope geometry, AM/GM structure, or dual-certificate analysis).

### Conjecture Backlog for Manuscript Integration

- **B1 (L-C1 extension with scoped qualifiers):** extend the L-T5 narrative beyond d<=3 with explicit computational scope tags: confirmed slices (d in {4,5} nondegenerate; d=6 n=3 partial), deferred slices (d=6 n>=4).
- **B2 (SONC scope refinement):** replace broad SONC containment language with a support-sensitive conjecture aligned with d=4/d=5/d=6 diagnostics.
- **B3 (transform-boundary conjecture):** promote CX-2 from anomaly note to a localized transform-boundary conjecture with explicit follow-up tests required for theorem-level promotion.

### Immediate Integration Actions

1. Mirror OP1-OP3 and B1-B3 in the manuscript computational synthesis section.
2. Keep solver-ceiling deferment language explicit for d=6 n>=4 and avoid theorem-level negation claims.
3. Feed B1-B3 into final steering tasks (publishable-claims ranking and next-quarter continuation plan).

## Weekly Update Protocol

At each weekly update, for each ledger item:
1. Confirm status.
2. Record one concrete next action.
3. Attach one evidence pointer (proof fragment, citation, or experiment result).
