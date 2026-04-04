# Phase 3 Experiment Matrix

Date: 2026-04-03
Scope: Step 5 entry artifact for Phase 3 (Weeks 5-8)
Canonical theorem source: mean_polynomials_main.tex

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
| L-C1 | Extension of SOS result beyond d in {1,2,3} for M_{2d,p} | M_{2d,p}(X,alpha) with d in {4,5,6}, n in {3,4,5,6}, alpha patterns: dense uniform, sparse nonuniform, boundary (many zeros) | SDPRelaxations.Decompose, SDPRelaxations.Minimize | SOS feasibility, Gram matrix rank, lower bound consistency, runtime | Any instance with valid PSD certificate but failed SOS decomposition under stable SDP settings | Table of feasible/infeasible by (d,n,p,alpha); decomposition logs |
| L-C2 | Broad SONC behavior for M_{2d,p}(X,alpha) depends on support geometry | Same as L-C1 plus support classes by Newton polytope type (simplex-like vs non-simplex) | SONCRelaxations.solve, GPRelaxations.solve, OptimizationProblem.newton, OptimizationProblem.delta | SONC feasibility, bound gap to SDP/SOS, solver status, sparsity sensitivity | Repeated SONC failure on a support class where SOS/SDP succeeds with stable numerics | Support-class heatmap and comparative bound table |
| CX-1 | Sparse nonuniform alpha may produce PSD but non-SOS high-degree cases | High-degree sparse alpha families: one dominant exponent, near-degenerate supports | SDPRelaxations.Decompose, SDPRelaxations.Minimize | PSD lower bound, SOS infeasibility flags, decomposition residuals | Robustly non-SOS classification across solver tolerances and seeds | Counterexample candidate dossier with reproducibility parameters |
| CX-2 | Mean-of-linear-forms examples near SONC boundary under transforms | Choi-Lam/Robinson-derived transformed families; random perturbations preserving degree/support | SONCRelaxations.solve, GPRelaxations.solve, OptimizationProblem.newton, OptimizationProblem.delta | SONC status pre/post transform, bound degradation, stability | SONC classification change under linear transform while PSD evidence remains | Transformation stress-test report with before/after labels |

## Benchmark Construction Rules

1. Degree ladder: d = 4, 5, 6 first; extend only if signals are stable.
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

1. L-C1 pilot on d=4, n=3/4.
2. L-C2 pilot on same instances with SONC/GP.
3. Expand to CX-1 sparse families.
4. Run CX-2 transform stress set.
5. Consolidate confirmed, weakened, and counterexample outcomes.

## Acceptance Criteria for Step 5 Completion

- Every item L-C1, L-C2, CX-1, CX-2 has:
  - at least one concrete family definition,
  - at least one mapped Irene method,
  - explicit falsification criterion,
  - explicit output artifact expectation.
- Matrix is sufficient to begin Step 6 benchmark assembly without additional design decisions.
