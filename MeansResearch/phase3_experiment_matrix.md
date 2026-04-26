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
- **CX-2 Batch 1 Status Update**:
  - Completed initial transform stress micro-batch on Choi-Lam/Robinson-derived families at tolerances $10^{-6},10^{-8}$ (`phase3_runs_clean_cx2_transform_probe.jsonl`, `phase3_pilot_summary_cx2_transform_probe.md`).
  - Observed pattern: SONC and GP both solve Choi-Lam base/perturbed slices; Robinson-hat base fails for both; transformed Robinson-hat base is SONC-feasible but still GP-infeasible.
  - Interpretation: a transform-sensitive SONC classification shift is now observed on the Robinson-hat base path, providing the first concrete CX-2 evidence slice and motivating larger perturbation sweeps.
- **CX-2 Batch 2 Status Update**:
  - Completed coefficient sweep around Choi-Lam/Robinson tails with before/after transform labels (`phase3_runs_clean_cx2_transform_probe_batch2.jsonl`, `phase3_pilot_summary_cx2_transform_probe_batch2.md`).
  - Observed pattern: Choi-Lam family remains SONC+GP feasible for all tested tail coefficients; Robinson-hat family remains pre-transform infeasible for SONC+GP.
  - Boundary behavior: post-transform SONC success is observed only at the baseline Robinson-hat coefficient ($c=2.0$), while nearby coefficients ($c\in\{1.98,1.95,1.9,1.85,1.8\}$) remain SONC-infeasible.
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
| CX-2 | Mean-of-linear-forms examples near SONC boundary under transforms | Choi-Lam/Robinson-derived transformed families; random perturbations preserving degree/support | SONCRelaxations.solve, GPRelaxations.solve, OptimizationProblem.newton, OptimizationProblem.delta | SONC status pre/post transform, bound degradation, stability | SONC classification change under linear transform while PSD evidence remains | Batch 1+2 completed (`phase3_runs_clean_cx2_transform_probe.jsonl`, `phase3_runs_clean_cx2_transform_probe_batch2.jsonl`): Robinson-hat transform flip holds at baseline $c=2.0$, while nearby Robinson-hat coefficients stay SONC-infeasible post-transform; GP remains infeasible on all Robinson-hat sweeps. |
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

### Escalation Protocol for Deferred L-C1 (d=6, n>=5)

For deferred SOS/SDP slices, run solver-sequence escalation with per-solver timeout caps:

1. Fallback chain: `cvxopt -> sdpa -> csdp` (environment-verified).
2. First pass: 600s per solver, sparse/boundary templates first.
3. Second pass (only unresolved): 1200s per solver.
4. Third pass (targeted singletons only): 1800s per solver.
5. Log all attempts via `solver_attempts` in JSONL records using `--sdp-solver-seq`.

Reference command template:

`python scripts/phase3_benchmarks.py --items L-C1 --degrees 6 --variables 5 --tolerances 1e-6 --sdp-solver-seq cvxopt,sdpa,csdp --solve-timeout 600 --output MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass1.jsonl`

## Execution Order

1. Completed: L-C1 pilot on d=4, n=3/4.
2. Completed: L-C2 pilot on same instances with SONC/GP.
3. Completed: Degree extension repeat on d=5, n=3/4 with matching tolerances and diagnostics.
4. Completed: convert the refreshed d=4/d=5 outputs into manuscript-native tables and local appendix artifacts.
5. Completed: d=6 L-C2 clean run; d=6 L-C1 uniform n=3 (batch1–2) successful; d=6 L-C1 n=4 cross-template p=6 probe **completed**—all timeout. **Decision: d=6 L-C1 marked deferred.**
6. Completed: CX-1 tractability probe — CVXOPT ceiling confirmed at n≤4; n≥5 boundary+mixed families deferred across d=3,4,5. Single d=3 n=5 probe: 272s success (`phase3_runs_clean_cx1_d3_n5_probe.jsonl`). Full CX-1 grid blocked on solver scalability.
7. Completed: CX-2 transform stress batch 1 — Choi-Lam/Robinson micro-batch at two tolerances with before/after labels (`phase3_runs_clean_cx2_transform_probe.jsonl`, `phase3_pilot_summary_cx2_transform_probe.md`).
8. Completed: CX-2 transform stress batch 2 — coefficient sweep around Robinson-hat and Choi-Lam tails (`phase3_runs_clean_cx2_transform_probe_batch2.jsonl`, `phase3_pilot_summary_cx2_transform_probe_batch2.md`).
9. Completed: d=6 L-C1 solver-sequence escalation path implemented in benchmark runner (`--sdp-solver-seq`) with smoke artifact (`phase3_runs_clean_d6_lc1_escalation_smoke.jsonl`).
10. Next: execute full d=6/n=5 escalation pass1 (600s per solver) and summarize solver-attempt outcomes.

## Acceptance Criteria for Step 5 Completion

- Every item L-C1, L-C2, CX-1, CX-2 has:
  - at least one concrete family definition,
  - at least one mapped Irene method,
  - explicit falsification criterion,
  - explicit output artifact expectation.
- Matrix is sufficient to begin Step 6 benchmark assembly without additional design decisions.
