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
| L-C1 | Extension of L-T5 to higher d beyond 3 | conjecture (d≤5 confirmed; d=6 n≤3 confirmed; d=6 n≥4 deferred — solver ceiling) | L-T5 patterns, sparse structure assumptions | SDPRelaxations.Decompose, SONCRelaxations.solve | d=4 and d=5 (n∈{3,4}) fully confirmed. d=6 n=3 confirmed. d=6 n≥4 computationally intractable under CVXOPT/SDPA/CSDP at budgets up to 1200s/solver; formally deferred pending high-performance solver access. |
| L-C2 | Broad SONC membership behavior for M_{2d,p}(X,\alpha) families | conjecture | L-T3, Newton polytope structure | SONCRelaxations.solve, GPRelaxations.solve | Replace the broad containment expectation with support-sensitive subcases; the clean $d=4$ pilot shows robust SONC failure on six nondegenerate families for $p \in \{1,2\}$. |

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

## Counterexample Tracking

| ID | Candidate | Current status | Why tracked | Next action |
|---|---|---|---|---|
| CX-1 | High-degree M_{2d,p} with sparse nonuniform \alpha potentially outside SOS | counterexample candidate | Could delimit the scope of L-T5 generalization | Generate symbolic families and test with SDP decomposition first. |
| CX-2 | Mean-of-linear-forms examples near SONC boundary under coordinate transforms | counterexample candidate | Could sharpen separation narrative around L-T7 | Completed focused transform stress tests (batch1+batch2); next action is to formalize a localized boundary conjecture around the Robinson-hat baseline coefficient under this transform family. |

## 2026-04-22 CX-2 Update

- Evidence pointer (batch 1): `phase3_runs_clean_cx2_transform_probe.jsonl`, `phase3_pilot_summary_cx2_transform_probe.md`.
- Evidence pointer (batch 2): `phase3_runs_clean_cx2_transform_probe_batch2.jsonl`, `phase3_pilot_summary_cx2_transform_probe_batch2.md`.
- Observed computational pattern under current SONC/GP setup:
	- Choi-Lam tail sweep (`c\in\{4.0,3.95,3.9,3.8\}`) remains SONC+GP feasible.
	- Robinson-hat pre-transform sweep (`c\in\{2.0,1.98,1.95,1.9,1.85,1.8\}`) remains SONC+GP infeasible.
	- Robinson-hat post-transform SONC feasibility is observed only at baseline `c=2.0`; nearby coefficients in the tested sweep remain SONC-infeasible.
- Interpretation update: current evidence supports a **localized transform-sensitive SONC boundary effect** rather than a broad transform-invariance failure across nearby Robinson-hat perturbations.

## Weekly Update Protocol

At each weekly update, for each ledger item:
1. Confirm status.
2. Record one concrete next action.
3. Attach one evidence pointer (proof fragment, citation, or experiment result).
