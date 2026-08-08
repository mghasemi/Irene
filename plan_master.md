# IreneRewrite — Master Modernization Plan

**Author:** Mehdi Ghasemi | **Date:** 2026-08-05 | **Status:** Planning Phase
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
| P3.5 | Implement Newton polytope monomial pruning | 6 | P3.3 |
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
| Codebase audit (grouprings.py, relaxations.py, matrices.py, sdp.py, program.py) | ✅ WORKED | All modules read; SymPy dependency map established |
| Task hierarchy creation script | ⏳ PENDING | Script written to `create_tasks.py`; awaiting execution |
| Master plan document | ✅ WORKED | This file — will be stored in Siyuan |

### Known Issues (Planning Phase)
- None yet. All tool calls returned successfully during initial audit.

---

## File Structure for IreneRewrite

```
IreneRewrite/
├── plan_master.md              ← this file
├── create_tasks.py             ← Vikunja task creation script
├── Irene/                      ← rewritten package (mirrors original structure)
│   ├── __init__.py
│   ├── symbolic_engine.py      ← NEW: SymEngine + fallback router
│   ├── border_basis.py         ← NEW: Border basis quotient ring projection
│   ├── cvxpy_interface.py      ← NEW: CVXPY problem formulation layer
│   ├── relaxation_api.py       ← NEW: Unified API entry point
│   ├── grouprings.py           ← updated from original
│   ├── relaxations.py          ← updated from original
│   ├── matrices.py             ← updated from original
│   ├── sdp.py                  ← updated from original (deprecated writers)
│   ├── program.py              ← updated from original
│   └── tests/
├── benchmarks/                 ← benchmark problem gallery (JSON/YAML)
├── docker-compose.yml          ← multi-Python test matrix
├── .github/workflows/ci.yml    ← CI pipeline
└── Reports/
    ├── phase1_performance.md
    ├── phase2_integration.md
    └── phase3_reduction.md
```
