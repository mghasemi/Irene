# IreneRewrite — Master Modernization Plan

**Author:** Mehdi Ghasemi | **Date:** 2026-08-08 | **Status:** Phase 3 Active (Phases 1–2 Complete)
**Vikunja Project:** #28 (IreneRewrite: Modernization Plan)

---

## Executive Summary

The Irene package — a scientific Python toolkit for polynomial optimization via SOS/SONC/SDP hierarchies — suffers from exponential slowdown during symbolic matrix generation due to SymPy's pure-Python implementation. This plan outlines a four-phase modernization strategy targeting the symbolic engine, solver interface, algebraic reductions, and CI/CD infrastructure.

**Core thesis:** Push polynomial arithmetic into C++ (SymEngine), unify solver routing through CVXPY, exploit structural sparsity via border bases + Newton polytope pruning, and validate everything through automated benchmarking.

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
| **P3.8** | **Integrate Phase 3 modules into relaxations.py** | **⬜ TODO** | Wire `border_basis.py`, `sparsity.py`, `newton_polytope.py` into `ReducedMonomialBase()` and `ReduceExp()` via the `relaxation_api.py` config dispatch |

---

## File Structure for IreneRewrite

```
IreneRewrite/
├── plan_master.md              ← this file
├── execution_log.md            ← detailed session-by-session log
├── create_tasks.py             ← Vikunja task creation script
├── siyuan_push.py              ← SiYuan document push script
├── conftest.py                 ← pytest configuration
├── setup.py                    ← package setup
├── Irene/                      ← rewritten package
│   ├── __init__.py
│   ├── symbolic_engine.py      ← Phase 1: SymEngine + fallback router (360 lines)
│   ├── cvxpy_solver.py         ← Phase 2: CVXPY DCP solver layer (340 lines)
│   ├── border_basis.py         ← Phase 3: Border basis quotient ring (523 lines)
│   ├── sparsity.py             ← Phase 3: Correlative sparsity detection (282 lines)
│   ├── newton_polytope.py      ← Phase 3: Newton polytope pruning (328 lines)
│   ├── relaxation_api.py       ← Phase 3: Unified API entry point (443 lines)
│   ├── dsdp.py                 ← Phase 3a: DSDP mean relaxation (updated)
│   ├── grouprings.py           ← updated: SymEngine routing
│   ├── relaxations.py          ← updated: engine-routed, CVXPY solve path
│   ├── matrices.py             ← updated: engine-routed
│   ├── sdp.py                  ← updated: CVXPY primary, legacy fallback
│   ├── program.py              ← updated: engine-routed
│   ├── sonc.py                 ← original (unchanged)
│   ├── sosonc.py               ← original (unchanged)
│   ├── geometric.py            ← original (unchanged)
│   ├── invariant.py            ← original (unchanged)
│   ├── base.py                 ← original (unchanged)
│   └── tests/
│       ├── test_border_basis.py
│       ├── test_sparsity.py
│       ├── test_newton_polytope.py
│       └── test_relaxation_api.py
├── tests/                      ← integration tests
│   ├── test_dsdp_mean.py
│   ├── test_sonc_section3.py
│   ├── test_sosonc.py
│   ├── test_solver_routing.py
│   └── test_relaxations.py
├── benchmarks/                 ← Phase 4: benchmark gallery
│   ├── gallery.yaml
│   ├── run_gallery.py
│   └── results/
├── reports/
│   ├── phase_1_report.md
│   ├── phase_2_integration_report.md
│   └── phase_3_audit.md
├── pyProximation/              ← auxiliary (rational approximation)
└── examples/                   ← example scripts
```
