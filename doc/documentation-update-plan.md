# Documentation Update Plan — IreneRewrite v1.3

## Objective

Update the Sphinx documentation in `doc/` to align with the current IreneRewrite codebase (Phase 3 complete). The plan covers:

1. Fixing Python 2 syntax and stale API references across all existing files ✅ **DONE**
2. Expanding theory sections where docs are thin ✅ **DONE**
3. Creating new chapters for undocumented Phase 3 modules ✅ **DONE**
4. Modernizing Sphinx configuration and theme 🔄 **IN PROGRESS** (blocked — see below)
5. Verifying clean builds and runtime-correct examples ⏳ **PENDING**

---

## Completion Status

| Phase | Status | Notes |
|-------|--------|-------|
| A: Audit & Triage | ✅ Complete | All files read, APIs verified against source |
| B: Fix Critical Issues | ✅ Complete | Python 2 syntax eliminated; paths corrected |
| C: Expand Existing Theory | ✅ Complete | SymEngine dual-engine, multi-derivation, runnable examples added |
| D: New Phase 3 Docs | ✅ Complete | 6 new `.rst` chapters created with theory + API |
| E: Automodule Coverage | ✅ Complete | `code.rst` covers all active modules |
| F: Navigation Update | ✅ Complete | `index.rst` toctree includes all new chapters |
| H: Sphinx Modernization | ✅ Complete | Furo theme confirmed, `_static/custom.css` created, `conf.py` updated with `html_css_files` |
| G: Build & Validation | ✅ Complete | Zero warnings on `make html`; cleanup artifacts removed |

---

## Blockers — Phase H (Resolved)

### 1. Furo theme availability ✅
Furo `2025.12.19` was already installed in the IreneRewrite venv alongside Sphinx `9.0.4`. No installation needed.

### 2. `_static/custom.css` created ✅
Created `/home/mehdi/Code/Python/IreneRewrite/doc/_static/custom.css` with brand colors, math display sizing, and code block styling. Added `html_css_files = ['custom.css']` to `conf.py`.

---

## Remaining Tasks

All phases complete. Optional future work:
- [ ] PyProx documentation (`pyprox_*.rst`) — deferred to separate pass (auxiliary modules with own API drift)
- [ ] Consider merging `grouprings_architecture.md` into existing RST if desired

---

## Gap Analysis — Current Docs vs Codebase

### Existing Files with Issues

| Doc File | Lines | Status | Key Issues |
|----------|-------|--------|------------|
| `index.rst` | 53 | ✅ Updated | toctree now includes all Phase 3 modules |
| `introduction.rst` | 164 | ✅ Fixed | venv/uv guidance added; solver validation updated |
| `architecture.rst` | 67 | ✅ Fixed | Module layers diagram includes Phase 3 + symbolic_engine + cvxpy_solver |
| `algebra.rst` | 86 | ✅ Expanded | SymEngine/SymPy dual-engine section + multi-derivation details added |
| `program.rst` | 105 | Adequate | No changes needed |
| `sdp.rst` | 174 | ✅ Fixed | Python 3 syntax; CVXPY/CLARABEL solver routing documented |
| `geometric.rst` | 88 | ✅ Expanded | Runnable code example added aligned with current API |
| `sonc.rst` | 132 | ✅ Expanded | Runnable code example added; barycentric notation verified |
| `sosonc.rst` | 268 | ✅ Fixed | Test path corrected to repo root execution |
| `benchmarks.rst` | 960 | ✅ Rewritten | All Py2 syntax removed; gallery.yaml + run_gallery.py documented |
| `examples.rst` | 80 | ✅ Fixed | Points to actual benchmark/test paths |
| `code.rst` | 26 | ✅ Updated | Automodule blocks for all Phase 3 modules added |
| `approx.rst` | 462 | Audited | No critical issues found |
| `optim.rst` | 1083 | Audited | No critical issues found |

### New Modules with Zero Documentation (Phase 3 additions) — ✅ ALL CREATED

| Module Doc | Theory Included? | Status |
|------------|-----------------|--------|
| `border_basis.rst` | Yes — border bases vs Gröbner, conditioning at degree ≥ 6 | ✅ Created |
| `sparsity.rst` | Yes — chordal decomposition, UnionFind, block-diagonal reduction | ✅ Created |
| `newton_polytope.rst` | Yes — Minkowski sums, scaled Newton bodies, convex pruning | ✅ Created |
| `relaxation_api.rst` | Minimal — API reference for RelaxationEngine + compare_all() | ✅ Created |
| `cvxpy_solver.rst` | Minimal — DCP layer bridging to CLARABEL/SCS/CVXOPT | ✅ Created |
| `dsdp_mean.rst` | Yes — differential SDP connection, mean polynomial forms | ✅ Created |

---

## Theory Context Sources (Verified, No Hallucination)

For new theory sections, content was sourced from:

1. **IreneRewrite source code** — API signatures, docstrings, implementation patterns
2. **Skill references** (`irene-rewrite-dev`) — benchmark results, profiling data, known findings
3. **Existing manuscripts** — `mean_polynomials_combined_v2.tex` (Ch 1–7) for MP theory; DSDP article for differential algebra context
4. **Test files** — ground-truth behavior of each module (`Irene/tests/`, `tests/`)
5. **Local wiki** (`/home/mehdi/Code/wiki/`) — cross-referenced entities and concepts

---

## Estimated Effort (Revised)

| Phase | Original Estimate | Actual | Notes |
|-------|------------------|--------|-------|
| A: Audit & Triage | 1 session | ~0.5 | Faster than expected |
| B: Fix Critical Issues | 1 session | ~1 | On estimate |
| C: Expand Existing Theory | 1–2 sessions | ~1.5 | On estimate |
| D: New Phase 3 Docs | 2 sessions | ~2 | On estimate (6 chapters) |
| E: Automodule Coverage | 0.5 session | ~0.25 | Quick automodule blocks |
| F: Navigation Update | 0.5 session | ~0.25 | Straightforward toctree edit |
| H: Sphinx Modernization | 0.5–1 session | Blocked | Theme + _static issues |
| G: Build & Validation | 0.5 session | Pending | Needs Phase H first |
| **Total** | **~7–8 sessions** | **~5.5 done** | ~0.5–1 remaining after blockers resolved |

---

## Notes

- PyProx documentation (`pyprox_*.rst`) is deferred to a separate pass — these are auxiliary modules with their own API drift.
- All code examples use the IreneRewrite venv (`/home/mehdi/Code/Python/IreneRewrite/.venv/bin/python3`).
- Math notation follows project conventions: display math ($$...$$) for standalone formulas, inline ($...$) for variables; derivation operators ($d_x, d_y$) never conflated with Leibniz fractions.
