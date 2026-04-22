# Phase 3 Experiment Matrix

Date: 2026-04-22
Scope: Step 5 entry artifact for Phase 3 (Weeks 5-8)
Canonical theorem source: mean_polynomials_main.tex

## Execution Snapshot (2026-04-22)

- Completed pilot grids:
  - d=4 clean run completed with summary and SONC diagnostics.
  - d=5 clean run completed with summary and SONC diagnostics.
  - Support-class tagging corrected and consolidated classification artifacts regenerated.
- Current observed pattern:
  - SDP/GP: all nondegenerate pilot families successful in d=4 and d=5 grids.
  - SONC: support-sensitive failures remain robust for the same nondegenerate families at both degrees under local/global solves and tighter error bounds.
- **Critical d=6 L-C1 n=4 Status Update**:
  - Completed cross-template **p=6 cross-validation probe** for n=4 d=6 L-C1: 
    - Uniform (α=(3,3,3,3), p=6): **timeout 1200s**
    - Boundary (α=(11,1,0,0), p=6): **timeout 1200s**
    - Mixed (α=(6,4,2,0), p=6): **timeout 1200s**
  - **Consolidated Evidence Pattern**: n=4 d=6 L-C1 exhibits **timeout-dominated behavior across all template types and p values (p=0–6)** under timeout budgets up to 1800s.
  - **Decision Implication**: Evidence supports **deferment of d=6 L-C1 from manuscript tables** unless specialist solver (e.g., exploited-structure SDP, first-order methods) and expanded timeout budget are adopted.
  - **Recommendation**: Mark d=6 as "computationally deferred—evidence incomplete under standard SDP budget constraints" and prioritize completing d=4/d=5 manuscript integration.
## Goal
Create a falsifiable experiment design for computationally testable ledger items using Irene methods:
- SDPRelaxations.Decompose
- SDPRelaxations.Minimize
- SONCRelaxations.solve
- GPRelaxations.solve
- OptimizationProblem.newton
- OptimizationProblem.delta

## Matrix

| Item ID | Hypothesis / Candidate | Families to Generate | Irene Methods | Primary Metrics | Falsification Criteria | Evidence Output |
|---|---|---|---|---|---|---|
| L-C1 | Extension of SOS result beyond d in {1,2,3} for M_{2d,p} | M_{2d,p}(X,alpha) with d in {4,5} completed and d=6 deferred pending specialist solver for L-C1; n in {3,4,5,6}, alpha patterns: dense uniform, sparse nonuniform, boundary (many zeros) | SDPRelaxations.Decompose, SDPRelaxations.Minimize | SOS feasibility, Gram matrix rank, lower bound consistency, runtime | Any instance with valid PSD certificate but failed SOS decomposition under stable SDP settings | Manuscript table plus decomposition logs for d=4,5 frozen; d=6 L-C1 evidence: batch1/batch2 (uniform n=3, p=0-6) successful; n=4 all-templates all-p timeout pattern (batches 3-8 covering p=0-6, uniform/boundary/mixed). Verdict: d=6 marked deferred. |
| L-C2 | Broad SONC behavior for M_{2d,p}(X,alpha) depends on support geometry | Same as L-C1 plus support classes by Newton polytope type (simplex-like vs non-simplex) | SONCRelaxations.solve, GPRelaxations.solve, OptimizationProblem.newton, OptimizationProblem.delta | SONC feasibility, bound gap to SDP/SOS, solver status, sparsity sensitivity | Repeated SONC failure on a support class where SOS/SDP succeeds with stable numerics | Manuscript classification tables for d=4,5 plus completed d=6 L-C2 artifacts |
| CX-1 | Sparse nonuniform alpha may produce PSD but non-SOS high-degree cases | High-degree sparse alpha families: one dominant exponent, near-degenerate supports | SDPRelaxations.Decompose, SDPRelaxations.Minimize | PSD lower bound, SOS infeasibility flags, decomposition residuals | Robustly non-SOS classification across solver tolerances and seeds | Counterexample candidate dossier with reproducibility parameters |
| CX-1 | Sparse nonuniform alpha may produce PSD but non-SOS high-degree cases | High-degree sparse alpha families: one dominant exponent, near-degenerate supports | SDPRelaxations.Decompose, SDPRelaxations.Minimize | PSD lower bound, SOS infeasibility flags, decomposition residuals | Robustly non-SOS classification across solver tolerances and seeds | **CVXOPT tractability ceiling confirmed at n≤4 across all tested degrees.** Probes at d=3 n=5 (272s/case), d=4 n=5/6 (timeout at 120s), d=5 n=5 (timeout at 900s) all confirm solver infeasibility at n=5+. Single confirmed proof: d=3 n=5 boundary p=0 success at 272s (`phase3_runs_clean_cx1_d3_n5_probe.jsonl`). Full sparse/nonuniform grid at n≥5 deferred pending specialized solver. |
| CX-2 | Mean-of-linear-forms examples near SONC boundary under transforms | Choi-Lam/Robinson-derived transformed families; random perturbations preserving degree/support | SONCRelaxations.solve, GPRelaxations.solve, OptimizationProblem.newton, OptimizationProblem.delta | SONC status pre/post transform, bound degradation, stability | SONC classification change under linear transform while PSD evidence remains | Transformation stress-test report with before/after labels |
## Benchmark Construction Rules

1. Degree ladder: d = 4, 5 completed; d = 6 L-C2 completed with diagnostics; d = 6 L-C1 has batch1 and batch2 (uniform n=3) completed and n=4 marked deferred after timeout-dominated probes.
2. Variable ladder: n = 3 to 6 with fixed alpha templates per level.
3. Alpha templates per (d,n):
- Uniform: as balanced as possible under |alpha| = 2d.
- Sparse: 1-2 large coordinates and many small/zero coordinates.
- Mixed: intermediate spread.
4. Repeat each experiment under at least 2 numeric tolerance settings.

## Logging Schema

Per run record:
- id, timestamp, commit hash
- polynomial family descriptor (d, n, p, alpha, support class)
- method and solver configuration
- status (success/fail/inconclusive)
- objective/bounds
- runtime
- decomposition diagnostics (if applicable)
- notes and anomaly flags

## Execution Order

1. Completed: L-C1 pilot on d=4, n=3/4.
2. Completed: L-C2 pilot on same instances with SONC/GP.
3. Completed: Degree extension repeat on d=5, n=3/4 with matching tolerances and diagnostics.
4. Completed: convert the refreshed d=4/d=5 outputs into manuscript-native tables and local appendix artifacts.
5. Completed: d=6 L-C2 clean run; d=6 L-C1 uniform n=3 (batch1–2) successful; d=6 L-C1 n=4 cross-template p=6 probe **completed**—all timeout. **Decision: d=6 L-C1 marked deferred.**
6. Completed: CX-1 tractability probe — CVXOPT ceiling confirmed at n≤4; n≥5 boundary+mixed families deferred across d=3,4,5. Single d=3 n=5 probe: 272s success (`phase3_runs_clean_cx1_d3_n5_probe.jsonl`). Full CX-1 grid blocked on solver scalability.
7. Next: run CX-2 transform stress set.
8. Next: consolidate confirmed, weakened, and counterexample outcomes.

## Acceptance Criteria for Step 5 Completion

- Every item L-C1, L-C2, CX-1, CX-2 has:
  - at least one concrete family definition,
  - at least one mapped Irene method,
  - explicit falsification criterion,
  - explicit output artifact expectation.
- Matrix is sufficient to begin Step 6 benchmark assembly without additional design decisions.
