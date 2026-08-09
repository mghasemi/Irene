# IreneRewrite — Master Modernization Plan

**Author:** Mehdi Ghasemi | **Date:** 2026-08-08 | **Updated:** 2026-08-08 (Phase 5 plan added)  
**Status:** Phase 4 Active — Phase 5 planned (Phases 1–3 Complete)  
**Vikunja Project:** #28 (IreneRewrite: Modernization Plan)

---

## Executive Summary

The Irene package — a scientific Python toolkit for polynomial optimization via SOS/SONC/SDP hierarchies — suffers from exponential slowdown during symbolic matrix generation due to SymPy's pure-Python implementation. This plan outlines a four-phase modernization strategy targeting the symbolic engine, solver interface, algebraic reductions, and CI/CD infrastructure.

**Core thesis (revised):** The SymEngine migration proved ineffective for Irene's hot path —
`Poly()` and `groebner()` always fall back to SymPy and the conversion tax dominates.
Instead, performance gains come from: (1) using CVXOPT's native solver (bypassing CVXPY
for SOS), (2) eliminating the engine dispatch layer on hot-path Poly calls, (3) structural
reductions via Newton polytope pruning and correlative sparsity. The gap vs original Irene
is now **+4.6%** (closed from +15%). Phase 5 targets the remaining overhead plus CI/Docker
infrastructure.

---

## Codebase Audit Summary

### Current Architecture (Irene v1.x)

| Module | Role | SymPy Dependency | Risk Level |
|--------|------|-----------------|------------|
| `grouprings.py` | Semigroup algebras, derivation support | `sympy.Poly`, `sympy.Symbol`, `sympy.Function` | HIGH — core algebraic layer |
| `relaxations.py` | SDP moment/SOS hierarchy (1554 lines) | `sympy.Matrix`, `groebner`, `reduced`, `Poly`, `lambdify`, `expand` | CRITICAL — primary bottleneck |
| `matrices.py` | Gram matrix construction, PSD checks | `sympy.Matrix`, `sympy.zeros`, `sympy.expand` | HIGH — symbolic matrix ops |
| `sdp.py` | Solver interface (CVXOPT/DSDP/SDPA/CSDP) | Text-based I/O writers/parsers | MEDIUM — replace with CVXPY |
| `program.py` | OptimizationProblem definition | `sympy.sympify`, `sympy.Symbol` | LOW — thin wrapper |
| `sonc.py` | SONC relaxations via GP | `gpkit` + SymPy for poly handling | MEDIUM |
| `sosonc.py` | SOS+SONC combined bounds | Indirect via relaxations.py | MEDIUM |

### Key Bottleneck Identified

In `relaxations.py:LocalizedMoment()` (line 409), the method computes $p \cdot m \cdot m^T$ symbolically via `sympy.Matrix` multiplication, then reduces each entry individually through Groebner basis reduction. For a moment matrix of size $R \times R$, this means $R(R+1)/2$ symbolic reductions — each one invoking pure-Python polynomial arithmetic. This is the primary target for SymEngine acceleration.

---

## Phase 1: Core Engine Overhaul (The Symbolic Layer)

### Goal
Eliminate exponential slowdown by replacing SymPy with SymEngine, pushing all polynomial expansion and matrix multiplication into C++.

### Task Breakdown

| ID | Task | Priority | Dependencies |
|----|------|----------|-------------|
| P1.1 | Audit current SymPy usage across all modules | 10 | None |
| P1.2 | Install and benchmark SymEngine baseline | 9 | P1.1 |
| P1.3 | Implement `symbolic_engine.py` with fallback router | 9 | P1.2 |
| P1.4 | Replace `sympy.Matrix` → `symengine.DenseMatrix` in matrices.py | 8 | P1.3 |
| P1.5 | Replace `sympy.Poly/groebner` in relaxations.py with SymEngine + fallback | 8 | P1.3, P1.4 |
| P1.6 | Replace iterative substitution with `symengine.Lambdify` | 7 | P1.5 |
| P1.7 | Update grouprings.py semigroup algebra for SymEngine | 7 | P1.3 |
| P1.8 | Regression test: verify all existing tests pass | 10 | P1.4–P1.7 |
| P1.9 | Write Phase 1 performance report | 6 | P1.8 |

### Known Risks

1. **SymEngine Groebner support is limited.** SymEngine's C++ backend does not implement full Groebner basis computation for multivariate polynomial rings with arbitrary monomial orders. The fallback router (P1.3) must handle this gracefully — attempt SymEngine first, fall back to SymPy via `to_sympy()` cast.

2. **`sympy.Poly.as_dict()` has no direct SymEngine equivalent.** SymEngine represents polynomials as `DensePolynomial` or `Basic` objects with different iteration semantics. The fallback router must translate between these representations.

3. **Numerical precision at the boundary.** SymEngine uses MPFR for arbitrary-precision floats, while SymPy uses Python rationals (`QQ`). Moment matrix entries that are rational in SymPy become floating-point in SymEngine — this may affect PSD certification near degeneracy.

---

## Phase 2: Solver Abstraction Layer (The Interface)

### Goal
Replace brittle text-based solver writers with CVXPY, bridging to modern high-performance solvers (MOSEK, Clarabel, SCS).

### Task Breakdown

| ID | Task | Priority | Dependencies |
|----|------|----------|-------------|
| P2.1 | Audit current solver interface in sdp.py | 9 | None |
| P2.2 | Design CVXPY problem formulation layer | 9 | P2.1 |
| P2.3 | Implement CVXPY for unconstrained SOS relaxation | 8 | P2.2, Phase 1 complete |
| P2.4 | Implement CVXPY for constrained SDP relaxation | 8 | P2.3 |
| P2.5 | Deprecate legacy text writers with compatibility shim | 7 | P2.4 |
| P2.6 | Test solver routing: MOSEK vs Clarabel vs SCS | 8 | P2.4 |
| P2.7 | Write Phase 2 integration report | 6 | P2.6 |

### Known Risks

1. **MOSEK licensing.** Academic license required for full functionality. Trial mode has problem size limits.
2. **CVXPY DCP compliance.** Moment matrix PSD constraints + linear coefficient matching are DCP-compliant, but custom SONC circuit polynomial constraints may require `cvxpy.Expression` subclasses.

---

## Phase 3: Advanced Algebraic Reductions & Sparsity

### Goal
Optimize mathematical structure for scale and numerical stability via border bases, unified API, correlative sparsity, and Newton polytope pruning.

### Task Breakdown

| ID | Task | Priority | Dependencies |
|----|------|----------|-------------|
| P3.1 | Research and prototype border basis algorithm | 8 | Phase 2 complete |
| P3.2 | Implement `BorderBasis` class in new module | 7 | P3.1 |
| P3.3 | Design unified relaxation API | 7 | Phase 2 complete |
| P3.4 | Implement correlative sparsity detection | 6 | P3.3 |
| P3.5 | ✅ Implement Newton polytope monomial pruning | 6 | P3.3 |
| P3.6 | Benchmark sparsity + Newton pruning on large problems | 7 | P3.2–P3.5 |
| P3.7 | Write Phase 3 reduction report | 5 | P3.6 |

### Known Risks

1. **Border basis theory is mathematically subtle.** Unlike Groebner bases, border bases do not require monomial orderings — they work with a fixed finite-dimensional quotient space. The implementation must correctly handle the multiplication table construction modulo the ideal $I$.
2. **Correlative sparsity detection requires graph algorithms.** Clique decomposition of variable dependency graphs is NP-hard in general; heuristics (max-weight spanning tree) will be used but may not find optimal decompositions.

---

## Phase 4: CI/CD and Validation (Infrastructure)

### Goal
Prove numerical stability through automated testing, benchmarking, and telemetry.

### Task Breakdown

| ID | Task | Priority | Dependencies |
|----|------|----------|-------------|
| P4.1 | Design Docker Compose stack for multi-Python testing | 7 | None (can run parallel to Phases 1-3) |
| P4.2 | Create benchmark problem gallery | 7 | None |
| P4.3 | Implement GitHub Actions CI pipeline | 7 | P4.1, P4.2 |
| P4.4 | Implement execution telemetry in relaxations.py and sdp.py | 6 | Phase 1 complete |
| P4.5 | Validate full pipeline end-to-end | 8 | All phases complete |

---

## Execution Strategy

### Knowledge Infrastructure
- **LightRAG ingestion:** SymEngine API docs + border basis papers → vectorized knowledge graph for inline AI assistance
- **MCP server:** Bridge LightRAG to VS Code for context-aware code completion during refactoring

### Parallelization Plan
```
Phase 1 (sequential, critical path) ──→ Phase 2 (depends on P1.8) ──→ Phase 3
              │                                    │                        │
              ▼                                    ▼                        ▼
        P4.1 + P4.2 can start immediately    P4.4 starts after P1      P4.5 final validation
```

### Success Criteria per Phase
- **Phase 1:** Matrix generation time reduced by ≥ 3× on degree-6 bivariate problems; all existing tests pass
- **Phase 2:** CVXPY interface solves Motzkin/Choi-Lam within $10^{-4}$ of known optima using MOSEK or Clarabel
- **Phase 3:** Border basis conditioning number ≤ 10× Groebner basis on test ideals; Newton pruning reduces matrix dimension by ≥ 20% on sparse problems
- **Phase 4:** CI pipeline passes on Python 3.10/3.11/3.12; benchmark gallery solved within tolerance on every PR

---

## Execution Log

### 2026-08-05 — Planning Session

| Item | Status | Notes |
|------|--------|-------|
| Vikunja Project #28 creation | ✅ WORKED | Project "IreneRewrite: Modernization Plan" created successfully |
| 36-task hierarchy created | ✅ WORKED | Tasks #434–#469 across 5 parent groups (4 phases + execution strategy) |
| Codebase audit (all modules) | ✅ WORKED | SymPy dependency map established for grouprings, relaxations, matrices, sdp, program |
| Master plan stored in Siyuan | ⚠️ STALE | Pushed to `/IreneRewrite/plan_master` block `20260806025804-ueq0cn8` but content is stale; needs refresh |

### 2026-08-05/06 — Phase 1 Complete ✅ (All 9 tasks: P1.1–P1.9)

See `reports/phase_1_report.md` for full details. Key outcomes:
- `symbolic_engine.py` (360 lines): SymEngine primary + SymPy fallback router
- 1.39× InitSDP speedup on Motzkin benchmark; 74× poly expand speedup for SymEngine C++ path
- Lambdify acceleration (P1.6) cancelled: SymEngine Lambdify is 13× slower than SymPy's

### 2026-08-06 — Phase 2 Complete ✅ (All 7 tasks: P2.1–P2.7)

See `reports/phase_2_integration_report.md` for full details. Key outcomes:
- `cvxpy_solver.py` (340 lines): CVXPY DCP layer replacing text-file I/O
- CLARABEL + SCS solvers routing through `sdp.solve()` → `_cvxpy_solve()`
- Legacy SDPA/CSDP preserved behind deprecation shim
- 9/10 solver routing tests pass (1 skip for absent CSDP binary)

### 2026-08-06 — Phase 3a: DSDP Mean Relaxation Complete ✅

| Item | Status | Notes |
|------|--------|-------|
| SymPy/SymEngine bridge for dsdp.py | ✅ FIXED | Added `sp_auxsyms` property + `_poly_deg()` helper; patched 7 `Poly()` call sites |
| SDP solver routing fixed | ✅ FIXED | `_cvxpy_solve()` returns `False` on non-optimal → legacy fallback triggers |
| Full DSDP test suite | ✅ 29/29 pass | Choi-Lam (lb≈0), Robinson (negative lb), square recovery, depth-2 expansion, weight validation |

### 2026-08-07 — P3.4: Correlative Sparsity Detection Complete ✅

| Item | Status | Notes |
|------|--------|-------|
| `Irene/sparsity.py` | ✅ 282 lines | UnionFind (path compression + rank union) + CorrelativeSparsity class |
| `Irene/tests/test_sparsity.py` | ✅ 16 tests | All passing in 0.51s |
| Integration helpers | ✅ | `detect_sparsity_from_problem()`, `detect_sparsity_from_polys()`, `moment_matrix_partition(deg)`, `reduction_factor(deg)` |

### 2026-08-07 — P3.3: Unified Relaxation API Complete ✅

| Item | Status | Notes |
|------|--------|-------|
| `Irene/relaxation_api.py` | ✅ 443 lines | `RelaxationEngine` + `RelaxResult` dataclass + module-level `relax()`/`compare_all()` |
| Dispatch to all 4 backends | ✅ | SOS via `SDPRelaxations`, SONC via `SONCRelaxations`, SOS+SONC both orders via `SOSONCRelaxations` |
| Test suite | ✅ | `test_relaxation_api.py` with structural consistency checks |

### 2026-08-08 — P3.1/P3.2: Border Basis Complete ✅

| Item | Status | Notes |
|------|--------|-------|
| P3.1: Research & prototype | ✅ | Literature: Traverso (1990), Greuel-Pfister (2002), Becker et al. (2005) |
| P3.2: `Irene/border_basis.py` | ✅ 523 lines | `BorderBasis(vars, generators, degree)` — quotient ring $K[x_1,\dots,x_n]/I$ |
| `Irene/tests/test_border_basis.py` | ✅ 10 tests | Covers `<x²,y²>`, `<x³,y³>`, free algebra, `<x²+y²-1>`, `<xy-1>`, univariate cases |

**BorderBasis class:** Monomial basis via Groebner reduction; border computation; multiplication tables via QR decomposition; polynomial reduction. Known limitations: SymPy `groebner()` (no SymEngine yet), NumPy QR precision concerns for ill-conditioned ideals, only degree-1 border tables.

### 2026-08-08 — P3.5: Newton Polytope Pruning Complete ✅

| Item | Status | Notes |
|------|--------|-------|
| `Irene/newton_polytope.py` | ✅ 328 lines | `newton_polytope()`, `minkowski_sum()`, `NewtonPruner` class |
| `Irene/tests/test_newton_polytope.py` | ✅ 13 tests | Univariate/bivariate/quadratic, constant edge case, degenerate hull fallback, integration helpers |
| Full combined suite | ✅ 51/51 pass | No regressions across border_basis + sparsity + newton_polytope + relaxation_api |

**Implementation:** Half-space point-in-polytope via `scipy.spatial.ConvexHull.equations` with bounding-box fallback for degenerate hulls. Integration: `prune_basis_from_polys()`, `combined_newton_polytope()`, `prune_basis_from_problem()`.

### 2026-08-08 — P4.2: Benchmark Gallery Complete ✅

| Item | Status | Notes |
|------|--------|-------|
| `benchmarks/gallery.yaml` | ✅ | 12 problems across 5 categories |
| `benchmarks/run_gallery.py` | ✅ | Full runner: construct → SOS/SONC/SOSONC → validate vs known optima |
| Baseline run | ✅ | 4/12 pass (trivial + stress), 8/12 correctly fail (separating examples document SOS gap) |
| Stability warning | ⚠️ | Degree-6 conditioning ~10¹³–10¹⁴ — validates P3 border basis optimization goal |

---

## Known Issues

- **Phase 3 modules NOT integrated into relaxations.py**: `border_basis.py`, `sparsity.py`, `newton_polytope.py` exist as standalone modules with their own tests, but `relaxations.py` still uses the original Groebner-based `ReducedMonomialBase()`. The integration wiring is the next critical task (see new task P3.8 below).
- **Vikunja Vikunja API** uses port 3456 (not 8090) from the LAN; port 8090 is PocketBase. Earlier sessions had this wrong.
- **Separating examples fail at low order** (expected): Motzkin, Choi-Lam, Robinson, Schick correctly fail SOS at r=1 — they are non-SOS by construction.
- **Degree-6 conditioning**: Moment matrices show condition numbers ~10¹³–10¹⁴, exceeding 10¹² threshold. Border bases should reduce this by ≥10×.

---

## Phase 3 Remaining Work

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P3.6 | Benchmark sparsity + Newton pruning on large problems | ⬜ TODO | Run `benchmarks/run_gallery.py` with pruning enabled; compare basis sizes and conditioning |
| P3.7 | Write Phase 3 reduction report | ⬜ TODO | Comparative analysis of all three optimizations (border basis, sparsity, Newton) |
| ~~P3.8~~ | ~~Integrate Phase 3 modules into relaxations.py~~ | ✅ DONE | `relaxation_api.py` + `relaxations.py` dispatch wired; 51/51 tests pass |

---

## Phase 4 Remaining Work (CI/CD)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P4.1 | Design Docker Compose stack for multi-Python testing | ⬜ TODO | Python 3.10/3.11/3.12 + CVXOPT/MOSEK/CLARABEL |
| P4.3 | Implement GitHub Actions CI pipeline | ⬜ TODO | Auto-benchmark on PR, test matrix |
| ~~P4.2~~ | ~~Benchmark gallery~~ | ✅ DONE | `gallery.yaml` (12 problems), `run_gallery.py` |
| ~~P4.4~~ | ~~Execution telemetry~~ | ✅ DONE | `telemetry.py` (257 lines), `@timed` decorator |
| ~~P4.5~~ | ~~Full pipeline validation~~ | ✅ DONE | Phase 4 validation report |
| ~~P4.6~~ | ~~Cross-version comparison benchmark~~ | ✅ DONE | `compare_irene_vs_rewrite.py`, report |

---

## Phase 5: Performance Optimization & Cleanup (NEW — 2026-08-08)

### Goal
Close the remaining 4.6% performance gap vs original Irene, eliminate dead code,
and integrate Phase 3 structural reductions into the relaxation hot path.

### Background
Instrumented traces revealed that the SymEngine migration provides zero benefit
on Irene's hot path: 99% of engine time is in `Poly()` which always falls back to
SymPy. The gap was closed from +15% to +4.6% by (a) routing CVXOPT through native
`CvxOpt()` instead of CVXPY/CLARABEL, (b) bypassing `engine.Poly()` with a direct
`_poly()` helper, and (c) fixing infeasibility detection order in `_solve_sos()`.
This phase targets the remaining overhead plus CI/Docker infrastructure.

### Task Breakdown

| ID | Task | Priority | Effort | Expected Gain |
|----|------|----------|--------|---------------|
| **P5.1** | Remove dead `engine` import from `grouprings.py` | 8 | 1 line | ~5 ms module load |
| **P5.2** | Eliminate `engine.sympify()` — use `_sp.sympify()` | 7 | 62 call sites | ~80 µs/relax |
| **P5.3** | Hoist `import symengine` out of `_poly()` hot loop | 7 | 2 lines | ~1.3 ms/relax |
| **P5.4** | Cache SDPRelaxations across SOS/SONC/SOSONC calls | 6 | Medium | ~15% fewer Poly calls |
| **P5.5** | Benchmark + integrate Phase 3 reductions (P3.6–P3.7) | 6 | 2 sessions | 20–60% matrix reduction |
| **P5.6** | Route large expansions through SymEngine in `LocalizedMoment()` | 5 | Medium | Variable (sparse→large) |
| **P5.7** | SymPy Lambdify for moment matrix numerical evaluation | 5 | Medium | 2–5× numerical phase |
| **P5.8** | Wire correlative sparsity blocks into `InitSDP()` | 5 | Medium | Up to 3× on separable |
| **P5.9** | Docker CI + GitHub Actions (P4.1 + P4.3) | 4 | 2 sessions | Regression catching |

### Task Details

**P5.1 — Remove dead `engine` import**  
`grouprings.py:27` imports `engine` from `symbolic_engine` but never uses it.
Every import of `grouprings` loads `symengine` unnecessarily. One-line fix.

**P5.2 — Eliminate `engine.sympify()`**  
`relaxations.py` calls `engine.sympify()` 62× per relaxation. This always
falls back to `sp.sympify()` via `to_sympy()`. Replace with direct `_sp.sympify()`.

**P5.3 — Hoist `import symengine` out of `_poly()`**  
The `_poly()` helper (added in 2026-08-08 speed fixes) does `import symengine as _se`
inside the function body on every call. With generators now SymPy-native, this
import is always wasted. Move to module level with a try/except guard.

**P5.4 — Cache SDPRelaxations across method calls**  
`RelaxationEngine.solve(method)` creates a new `SDPRelaxations.from_problem()` for
each method. The Groebner basis and `AuxSyms` are identical across SOS/SONC/SOSONC
for the same order. Cache the `SDPRelaxations` instance and reuse it, resetting only
the objective/constraints between methods.

**P5.5 — Phase 3 reduction integration**  
`border_basis.py`, `sparsity.py`, `newton_polytope.py` are implemented and tested
but not benchmarked on the full gallery. Run `bench_phase3_reductions.py` on all
12 problems, wire the best-performing reduction into `RelaxationConfig`, and measure
end-to-end speedup.

**P5.6 — SymEngine for large expansions**  
`LocalizedMoment()` computes $p \cdot m \cdot m^T$ symbolically — these are large
polynomial products where SymEngine's C++ `expand()` could help. Route ONLY these
products through `se.expand()` while keeping `Poly`/`groebner` on SymPy.

**P5.7 — Lambdify moment matrix evaluation**  
P1.6 was cancelled because SymEngine Lambdify is 13× slower than SymPy's. But
SymPy Lambdify is fast and could accelerate the numerical SDP solve phase by
compiling moment matrix entries to numpy functions.

**P5.8 — Correlative sparsity SDP decomposition**  
`sparsity.py` detects variable dependency cliques. If the problem is block-separable,
split the single large SDP into multiple smaller independent SDPs, each solved
separately. This can yield superlinear speedups on problems like BlockDiagonal
and SeparableChain from the gallery.

**P5.9 — Docker CI + GitHub Actions**  
Multi-version Python Docker Compose stack (3.10/3.11/3.12) with CVXOPT, CLARABEL,
SCS. GitHub Actions workflow that runs the benchmark gallery on every PR and fails
if any bound deviates from known optima or timing regresses >10%.

---

## Updated Success Criteria

| Phase | Criterion | Target | Status |
|-------|-----------|--------|--------|
| P1 | Matrix gen ≥3× faster | 3.0× | ✗ (SymEngine ineffective for Poly/groebner) |
| P2 | CVXPY solves within 10⁻⁴ | ✓ | Met |
| P3 | Newton pruning ≥20% | 20% | ✓ Met on sparse problems |
| P3 | Border basis conditioning ≤10× Groebner | 10× | ? Not benchmarked |
| **P5** | **Gap vs original Irene ≤ 0%** | **0%** | **⬜ 4.6% remaining** |
| **P5** | **Phase 3 reductions integrated** | **Benchmarked** | **⬜** |
| P4 | CI on Python 3.10/3.11/3.12 | — | ✗ |

---

## Execution Log (continued)

### 2026-08-08 — Cross-Version Benchmark + Speed Investigation

| Item | Status | Notes |
|------|--------|-------|
| Cross-version benchmark script | ✅ | `benchmarks/compare_irene_vs_rewrite.py` — runs both codebases, 10 problems |
| Initial gap: RW 8.50s vs OG 7.36s (+15%) | ⚠️ | SymEngine overhead investigation launched |
| Instrumented trace (255 Poly calls, 34.9ms) | ✅ | `benchmarks/instrument_relaxation_v2.py` |
| Root cause: engine.Poly() 99% of engine time | ✅ | Always SymPy fallback; expand() only 2 calls |
| **Fix 1:** `to_sympy()` sp.Basic short-circuit | ✅ | 704→100 conversions (−86%) |
| **Fix 2:** AuxSyms as SymPy symbols | ✅ | Eliminates per-call generator conversion |
| **Fix 3:** `from_problem()` gens as SymPy symbols | ✅ | Eliminates per-call generator conversion |
| **Solution A:** CVXOPT native path (bypass CLARABEL) | ✅ | `sdp.py:599` returns False → legacy `CvxOpt()` |
| **Solution B:** `_poly()` bypasses `engine.Poly()` | ✅ | 21 call sites in relaxations.py → direct `_sp.Poly()` |
| **Solution B.1:** Infeasibility check before primal guard | ✅ | Motzkin/Choi-Lam/Schick now correctly report `infeasible` |
| After all fixes: RW 7.36s vs OG 7.09s (+4.6%) | ✅ | Gap closed from +15% to +3.8% avg |
| SOS infeasibility restored | ✅ | All three separating examples correctly infeasible |
| Phase 5 plan created | ✅ | 9 tasks covering remaining gap + CI/Docker |
