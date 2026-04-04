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
| L-T5 | M_{2d,p} is SOS for d in {1,2,3} and allowed p | draft proof | L-T4, Hilbert cases, computational confirmations | mean_polynomials_main.tex (Thm. thm:M_2d_p_SOS) | Split theorem proof into case lemmas and mark computationally-assisted steps explicitly. |
| L-T7 | Choi-Lam/Robinson transformed forms as separating examples beyond SOSONC | proved | Transformation identity, SONC non-invariance under linear maps, supporting citations | mean_polynomials_main.tex (Prop. prop:ChoiLamRobinson and post-proof paragraph in Section s05) | No further action required; PSD_Mean removed. |

## GMP Linkage and Extended Objectives

| ID | Statement (short) | Current status | Dependencies | Primary source | Next action |
|---|---|---|---|---|---|
| L-G1 | Generalized moment/semigroup bridge integrated into mean-cone framework | deferred | Proposal objective O1, final section architecture | Proposal PDF + source-of-truth O1 | Explicitly deferred to follow-up paper; no content in current manuscript. |
| L-G2 | Differential-algebra/KKT dependency handling integrated with mean framework | deferred | Follow-up paper scope decision | source-of-truth decision 6 | Keep explicitly out of current manuscript; seed follow-up theorem list. |

## Computationally Testable Conjectures

| ID | Statement (short) | Current status | Dependencies | Planned computational hook | Next action |
|---|---|---|---|---|---|
| L-C1 | Extension of L-T5 to higher d beyond 3 | conjecture | L-T5 patterns, sparse structure assumptions | SDPRelaxations.Decompose, SONCRelaxations.solve | Design benchmark families by degree and support to test SOS/SONC behavior. |
| L-C2 | Broad SONC membership behavior for M_{2d,p}(X,\alpha) families | conjecture | L-T3, Newton polytope structure | SONCRelaxations.solve, GPRelaxations.solve | Define falsification criteria and classify outcomes by support geometry. |

## Counterexample Tracking

| ID | Candidate | Current status | Why tracked | Next action |
|---|---|---|---|---|
| CX-1 | High-degree M_{2d,p} with sparse nonuniform \alpha potentially outside SOS | counterexample candidate | Could delimit the scope of L-T5 generalization | Generate symbolic families and test with SDP decomposition first. |
| CX-2 | Mean-of-linear-forms examples near SONC boundary under coordinate transforms | counterexample candidate | Could sharpen separation narrative around L-T7 | Build focused transformation stress tests with SONC solver diagnostics. |

## Weekly Update Protocol

At each weekly update, for each ledger item:
1. Confirm status.
2. Record one concrete next action.
3. Attach one evidence pointer (proof fragment, citation, or experiment result).
