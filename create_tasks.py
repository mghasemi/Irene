#!/usr/bin/env python3
"""Create full IreneRewrite task hierarchy in Vikunja project #28 using Python API directly."""

import sys
sys.path.insert(0, "/home/mehdi/.hermes/profiles/math/skills/productivity/vikunja")
from vikunja_tool import create_task, VikunjaError

PROJECT_ID = 28
results = []

def mk(title, desc="", priority=5, labels=None, parent_id=None):
    try:
        r = create_task(PROJECT_ID, title, description=desc, priority=priority,
                        labels=labels, parent_task_id=parent_id)
        tid = r.get("id") if isinstance(r, dict) else None
        results.append(f"  ✅ {title} → ID {tid}")
        return tid
    except VikunjaError as e:
        results.append(f"  ❌ {title}: {e}")
        return None

print("=" * 70)
print("IreneRewrite — Vikunja Task Creation (Project #28)")
print("=" * 70)

# ============================================================
# PHASE 1: Core Engine Overhaul (The Symbolic Layer)
# ============================================================
print("\n[Phase 1] Core Engine Overhaul")
p1 = mk("Phase 1: Core Engine Overhaul — Symbolic Layer",
    "Replace SymPy with SymEngine for C++-backed polynomial arithmetic. Add fallback router and Lambdify acceleration.",
    priority=10, labels=["phase-1", "symbolic"])

mk("P1.1 Audit current SymPy usage across all modules",
    "Scan grouprings.py, relaxations.py, matrices.py, sdp.py, sonc.py for every import sympy / from sympy. Map which APIs are used (symbols, Matrix, Poly, groebner, lambdify, expand, reduced). Produce a dependency matrix.",
    priority=10, labels=["phase-1", "audit"], parent_id=p1)

mk("P1.2 Install and benchmark SymEngine baseline",
    "Install symengine in Irene/.venv/. Run micro-benchmarks: (a) polynomial expansion of degree-8 bivariate, (b) 50x50 symbolic matrix multiply, (c) groebner basis on 3-constraint system. Compare wall-clock vs current SymPy.",
    priority=9, labels=["phase-1", "benchmark"], parent_id=p1)

mk("P1.3 Implement symbolic_engine.py with to_sympy() fallback router",
    "Create Irene/symbolic_engine.py: (a) primary path uses symengine.symbols/DenseMatrix/Basic, (b) to_sympy()/from_sympy() cast utilities, (c) try/except fallback that catches AttributeError and reroutes through SymPy. Unit-test each cast round-trip.",
    priority=9, labels=["phase-1", "implementation"], parent_id=p1)

mk("P1.4 Replace sympy.Matrix with symengine.DenseMatrix in matrices.py",
    "Update get_gram_matrix(), is_psd_symbolic() to use DenseMatrix. Verify PSD check still works via numpy interface.",
    priority=8, labels=["phase-1", "implementation"], parent_id=p1)

mk("P1.5 Replace sympy.Poly/groebner in relaxations.py with SymEngine equivalents",
    "SymEngine has limited groebner support — implement fallback router: attempt symengine.groebner first, fall back to sympy.groebner via to_sympy() cast. Update ReducedMonomialBase(), ReduceExp(), LocalizedMoment().",
    priority=8, labels=["phase-1", "implementation"], parent_id=p1)

mk("P1.6 Replace iterative substitution with symengine.Lambdify in relaxations.py",
    "Where moment matrices are evaluated numerically (e.g., MomentMat()), compile symbolic expressions to C/LLVM functions via lambdify. Benchmark speedup on Motzkin and Choi-Lam test cases.",
    priority=7, labels=["phase-1", "implementation"], parent_id=p1)

mk("P1.7 Update grouprings.py semigroup algebra to use SymEngine for polynomial ops",
    "The SemigroupAlgebraElement class uses sympy.Poly internally for degree computation and reduction. Replace with symengine equivalents where possible; keep SymPy fallback for advanced Poly features.",
    priority=7, labels=["phase-1", "implementation"], parent_id=p1)

mk("P1.8 Regression test: verify all existing tests pass with SymEngine backend",
    "Run pytest -v in Irene/.venv/ against test_mean.py, test_sonc_section3.py, test_sosonc.py, test_dsdp_mean.py. Document any failures and their root cause (SymEngine API gap vs actual regression).",
    priority=10, labels=["phase-1", "testing"], parent_id=p1)

mk("P1.9 Write Phase 1 performance report",
    "Compare matrix generation time before/after for each benchmark problem. Store results in Siyuan and IreneRewrite/Reports/. Flag any operation where SymEngine is slower than SymPy.",
    priority=6, labels=["phase-1", "reporting"], parent_id=p1)

# ============================================================
# PHASE 2: Solver Abstraction Layer (The Interface)
# ============================================================
print("\n[Phase 2] Solver Abstraction Layer")
p2 = mk("Phase 2: Solver Abstraction Layer — CVXPY Integration",
    "Replace legacy text-based solver writers with CVXPY. Map moment matrices to cvxpy.Variable/Problem objects.",
    priority=9, labels=["phase-2", "solver"])

mk("P2.1 Audit current solver interface in sdp.py",
    "Map write_sdpa_dat(), sdpa(), csdp(), dsdp() text writers and parse_solution_matrix()/read_*_out() parsers. Identify which features each solver provides that CVXPY must replicate.",
    priority=9, labels=["phase-2", "audit"], parent_id=p2)

mk("P2.2 Design CVXPY problem formulation layer",
    "Draft Irene/cvxpy_interface.py: (a) MomentMatrixVariable class wrapping cvxpy.Variable(PSD=True), (b) constraint builder that maps coefficient matching to cvxpy equality constraints, (c) solver router supporting MOSEK/Clarabel/SCS/CVXOPT.",
    priority=9, labels=["phase-2", "design"], parent_id=p2)

mk("P2.3 Implement CVXPY interface for unconstrained SOS relaxation",
    "Start with simplest case: global SOS minimum of univariate polynomial. Build cvxpy.Problem from moment matrix variables + coefficient constraints. Verify against known analytical solution.",
    priority=8, labels=["phase-2", "implementation"], parent_id=p2)

mk("P2.4 Implement CVXPY interface for constrained SDP relaxation",
    "Extend to localized moment matrices with inequality constraints. Map LocalizedMoment() output to PSD block constraints in cvxpy.",
    priority=8, labels=["phase-2", "implementation"], parent_id=p2)

mk("P2.5 Deprecate legacy text writers with compatibility shim",
    "Keep write_sdpa_dat() as deprecated method that raises warning but still works for backward compatibility. New code path routes through CVXPY.",
    priority=7, labels=["phase-2", "migration"], parent_id=p2)

mk("P2.6 Test solver routing: MOSEK vs Clarabel vs SCS on benchmark suite",
    "Run Motzkin, Choi-Lam, Robinson through each available solver backend. Compare solution quality, solve time, and numerical stability.",
    priority=8, labels=["phase-2", "testing"], parent_id=p2)

mk("P2.7 Write Phase 2 integration report",
    "Document which solvers are available, performance comparison table, and any features lost during migration from legacy writers.",
    priority=6, labels=["phase-2", "reporting"], parent_id=p2)

# ============================================================
# PHASE 3: Advanced Algebraic Reductions & Sparsity
# ============================================================
print("\n[Phase 3] Advanced Algebraic Reductions")
p3 = mk("Phase 3: Advanced Algebraic Reductions — Border Bases, Sparsity, Newton Polytope",
    "Integrate border bases for better conditioning, unify relaxation API, implement correlative sparsity and Newton polytope pruning.",
    priority=7, labels=["phase-3", "reduction"])

mk("P3.1 Research and prototype border basis algorithm",
    "Study Trager-Zachos border basis theory. Prototype quotient ring projection R[x]/I using border basis instead of Groebner basis. Compare conditioning of transformation matrices on test ideals.",
    priority=8, labels=["phase-3", "research"], parent_id=p3)

mk("P3.2 Implement BorderBasis class in Irene/border_basis.py",
    "New module: (a) compute border from monomial basis, (b) multiplication tables mod I, (c) moment matrix representation in border basis coordinates. Include numerical conditioning diagnostics.",
    priority=7, labels=["phase-3", "implementation"], parent_id=p3)

mk("P3.3 Design unified relaxation API",
    "Draft Irene/relaxation_api.py: single entry point solve(problem, relaxation='sos'|'sonc'|'mean', solver='auto'). Internally dispatches to correct relaxation class.",
    priority=7, labels=["phase-3", "design"], parent_id=p3)

mk("P3.4 Implement correlative sparsity detection",
    "Build dependency graph from variable cliques in objective + constraints. Use clique decomposition (e.g., max-weight spanning tree heuristic) to split large PSD blocks into smaller block-diagonal constraints.",
    priority=6, labels=["phase-3", "implementation"], parent_id=p3)

mk("P3.5 Implement Newton polytope monomial pruning",
    "Extend existing newton() method in program.py: after computing polytope, filter moment basis to only include exponents inside or on boundary of polytope. Measure dimension reduction.",
    priority=6, labels=["phase-3", "implementation"], parent_id=p3)

mk("P3.6 Benchmark sparsity + Newton pruning on large problems",
    "Test on 4+ variable problems with degree >= 6. Compare matrix dimensions and solve times with vs without reductions.",
    priority=7, labels=["phase-3", "testing"], parent_id=p3)

mk("P3.7 Write Phase 3 reduction report",
    "Document conditioning improvements from border bases, dimension reductions from sparsity/Newton pruning, and unified API usage examples.",
    priority=5, labels=["phase-3", "reporting"], parent_id=p3)

# ============================================================
# PHASE 4: CI/CD and Validation (Infrastructure)
# ============================================================
print("\n[Phase 4] CI/CD and Validation")
p4 = mk("Phase 4: CI/CD and Validation — Docker, Benchmarks, Telemetry",
    "Dockerized test matrix, GitHub Actions benchmarking pipeline, execution telemetry tracking.",
    priority=6, labels=["phase-4", "infra"])

mk("P4.1 Design Docker Compose stack for multi-Python testing",
    "docker-compose.yml with services: (a) Python 3.10, 3.11, 3.12 test runners, (b) solver containers (MOSEK trial, Clarabel, SCS), (c) shared volume for benchmark data.",
    priority=7, labels=["phase-4", "docker"], parent_id=p4)

mk("P4.2 Create benchmark problem gallery",
    "Define JSON/YAML files for: Motzkin, Choi-Lam, Robinson, Hopf, and 3 custom high-degree problems. Each file specifies polynomial coefficients, expected optimum, and tolerance.",
    priority=7, labels=["phase-4", "benchmark"], parent_id=p4)

mk("P4.3 Implement GitHub Actions CI pipeline",
    ".github/workflows/ci.yml: on push/PR → (a) install deps in venv, (b) run pytest suite, (c) solve benchmark gallery, (d) assert optima within tolerance, (e) upload timing results as artifact.",
    priority=7, labels=["phase-4", "cicd"], parent_id=p4)

mk("P4.4 Implement execution telemetry in relaxations.py and sdp.py",
    "Add context managers that time: (a) symbolic matrix generation, (b) numerical evaluation, (c) solver execution. Log to JSONL file for trend analysis.",
    priority=6, labels=["phase-4", "telemetry"], parent_id=p4)

mk("P4.5 Validate full pipeline end-to-end",
    "Run CI locally via Docker Compose. Verify all 4 phases produce correct results on benchmark gallery. Document any environment-specific failures.",
    priority=8, labels=["phase-4", "validation"], parent_id=p4)

# ============================================================
# EXECUTION STRATEGY TASKS
# ============================================================
print("\n[Execution Strategy]")
es = mk("Execution Strategy — Knowledge Infrastructure and AI Tooling",
    "LightRAG ingestion of SymEngine docs + border basis papers. MCP server for VS Code integration.",
    priority=5, labels=["execution-strategy"])

mk("ES.1 Ingest SymEngine API documentation into LightRAG",
    "Download symengine.py wrapper source + official docs. Chunk and ingest via lightrag-ingest skill. Verify retrievability with test queries.",
    priority=6, labels=["execution-strategy", "knowledge"], parent_id=es)

mk("ES.2 Ingest border basis algorithmic papers into LightRAG",
    "Find and ingest: Trager-Zachos (1991), Galligo (1988), and modern survey on border bases for polynomial optimization.",
    priority=5, labels=["execution-strategy", "knowledge"], parent_id=es)

mk("ES.3 Configure MCP server for VS Code inline assistance",
    "Set up local MCP server pointing to LightRAG knowledge graph. Test that Copilot/inline assistant can retrieve SymEngine API info during coding sessions.",
    priority=5, labels=["execution-strategy", "tooling"], parent_id=es)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
for r in results:
    print(r)
success = sum(1 for r in results if "✅" in r)
failed = sum(1 for r in results if "❌" in r)
print(f"\nTotal: {len(results)} tasks | ✅ {success} created | ❌ {failed} failed")
