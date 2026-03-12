## Reviewer Plan Tracker: Core Solver and Relaxation Review

Purpose: Track reviewer-approved review work for possible uncaught errors, optimization opportunities, and readability improvements in core Irene modules.

### Scope

- In scope:
  - Irene/base.py
  - Irene/relaxations.py
  - Irene/sdp.py
  - Irene/program.py
- Out of scope for this pass:
  - External solver integration validation requiring SDPA/CSDP/CVXOPT availability checks beyond lightweight local checks
  - Algorithm redesign
  - Repository-wide style migration

### Review Mode

- Static deep review: Enabled
- Lightweight runtime checks: Enabled
- Full solver integration execution: Disabled in this pass

### Review Status Dashboard

| Phase | Description                                     | Depends On | Owner   | Reviewer | Status    | Target Date | Evidence                           |
| ----- | ----------------------------------------------- | ---------- | ------- | -------- | --------- | ----------- | ---------------------------------- |
| 1     | Review setup and risk mapping                   | None       | Copilot | Done     | Done      | 2026-03-12  | Initial findings section below     |
| 2     | Correctness and error-handling audit            | 1          | Copilot | Done     | Done      | 2026-03-12  | Prioritized findings section below |
| 3     | Optimization and readability audit              | 2          | Copilot | Done     | Done      | 2026-03-12  | Prioritized findings section below |
| 4     | Lightweight runtime validation                  | 2          | Copilot | Done     | Done      | 2026-03-12  | Verification log entries below     |
| 5     | Prioritized findings list and reviewer hand-off | 2, 3, 4    | Copilot | TBD      | In Review | 2026-03-12  | Prioritized findings section below |

Allowed status values: Not Started, In Progress, Blocked, In Review, Approved, Done.

### Acceptance Criteria

1. Phase 1

- File-level API and call-path map is captured for all in-scope modules.
- Severity rubric and evidence format are agreed.

2. Phase 2

- High-risk correctness issues are identified with line references and failure modes.
- Error-handling and edge-case coverage gaps are explicitly listed.

3. Phase 3

- Performance hotspots are identified with expected benefit and effort.
- Readability pain points are identified with actionable refactor directions.

4. Phase 4

- Compile/import checks pass for all four modules.
- Baseline tests run and results are logged.
- Runtime probes remain solver-independent for this pass.

5. Phase 5

- Findings are prioritized by severity first, then fix cost.
- Each Critical/High finding includes fix direction and test recommendation.

### Reviewer Checklist

- [ ] Scope and exclusions approved.
- [ ] Severity rubric approved.
- [ ] Review mode approved (static + lightweight runtime only).
- [ ] Evidence requirements approved.
- [ ] Final prioritized findings accepted.

### Verification Log

| Date       | Check                                  | Result | Notes                                                                                                                                                                                                                                                                              | Reviewer |
| ---------- | -------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 2026-03-12 | Tracker initialized                    | Pass   | Plan file created in doc/ and ready for reviewer approval                                                                                                                                                                                                                          | Copilot  |
| 2026-03-12 | Module compile checks                  | Pass   | /home/mehdi/Code/Irene/.venv/bin/python -m py_compile Irene/base.py Irene/relaxations.py Irene/sdp.py Irene/program.py                                                                                                                                                             | Copilot  |
| 2026-03-12 | Baseline unit tests                    | Pass   | /home/mehdi/Code/Irene/.venv/bin/python -m unittest tests.test_quality_plan tests.test_sonc_section3 (Ran 17 tests, OK)                                                                                                                                                            | Copilot  |
| 2026-03-12 | Editor problem scan                    | Pass   | No reported problems in Irene/base.py, Irene/relaxations.py, Irene/sdp.py, Irene/program.py                                                                                                                                                                                        | Copilot  |
| 2026-03-12 | Prioritized findings draft published   | Pass   | Phase 5 reviewer package added with severity ordering, test gaps, and remediation sequence                                                                                                                                                                                         | Copilot  |
| 2026-03-12 | Finding 6 remediation check            | Pass   | Updated relaxation state I/O to context-managed binary pickle in Irene/relaxations.py and reran compile + baseline tests (17 tests, OK)                                                                                                                                            | Copilot  |
| 2026-03-12 | Finding 1 remediation check            | Pass   | Updated Irene/sdp.py constructor default to solver_path=None with defensive dict copy; added regression test tests/test_quality_plan.py::TestSdpFixes::test_solver_path_is_copied_on_init; suite now passes (18 tests, OK)                                                         | Copilot  |
| 2026-03-12 | Finding 2 remediation check            | Pass   | Replaced assert-based solver validation in Irene/sdp.py with explicit ValueError gate; added tests/test_quality_plan.py::TestSdpFixes::test_invalid_solver_raises_value_error; suite now passes (19 tests, OK)                                                                     | Copilot  |
| 2026-03-12 | Finding 3 remediation check            | Pass   | Updated sparse SDPA writer in Irene/sdp.py to suppress near-zero entries via tolerance threshold; added tests/test_quality_plan.py::TestSdpFixes::test_sparse_writer_ignores_near_zero_entries; suite now passes (20 tests, OK)                                                    | Copilot  |
| 2026-03-12 | Finding 4 remediation check            | Pass   | Updated Irene/sdp.py CSDP execution path to raise RuntimeError on subprocess failure before parsing output; added tests/test_quality_plan.py::TestSdpFixes::test_csdp_failure_raises_runtime_error_before_parsing; suite now passes (21 tests, OK)                                 | Copilot  |
| 2026-03-12 | Finding 5 remediation check            | Pass   | Extracted parallel Calpha worker management in Irene/relaxations.py into a helper that joins started workers and terminates them on failure; added tests/test_quality_plan.py::TestRelaxationsFixes worker-lifecycle coverage; suite now passes (23 tests, OK)                     | Copilot  |
| 2026-03-12 | SDPA return-code remediation check     | Pass   | Updated Irene/sdp.py SDPA execution path to raise RuntimeError on non-zero subprocess exit before parsing output; added tests/test_quality_plan.py::TestSdpFixes::test_sdpa_failure_raises_runtime_error_before_parsing; suite now passes (24 tests, OK)                           | Copilot  |
| 2026-03-12 | Parser hardening check                 | Pass   | Hardened Irene/sdp.py::parse_solution_matrix against incomplete and inconsistent matrix blocks; added tests/test_quality_plan.py::TestSdpFixes::test_parse_solution_matrix_rejects_incomplete_matrix; suite now passes (25 tests, OK)                                              | Copilot  |
| 2026-03-12 | CSDP parser hardening check            | Pass   | Hardened Irene/sdp.py::read_csdp_out to use whitespace-robust tokenization and explicit row validation; added tests/test_quality_plan.py::TestSdpFixes::test_read_csdp_out_accepts_irregular_whitespace; suite now passes (26 tests, OK)                                           | Copilot  |
| 2026-03-12 | Symbolic objective serialization check | Pass   | Coerced symbolic objective coefficients to floats in Irene/sdp.py writers/CVXOPT path; added tests/test_quality_plan.py::TestSdpFixes::test_sparse_writer_coerces_symbolic_objective_coefficients; DropWave example now solves with CSDP and targeted suite passes (27 tests, OK)  | Copilot  |
| 2026-03-12 | Constraint-type check remediation      | Pass   | Replaced identity-based relation branching in Irene/relaxations.py::AddConstraint with isinstance checks and added tests/test_quality_plan.py::TestRelaxationsFixes::test_add_constraint_accepts_equality_subclass; suite now passes (28 tests, OK)                                | Copilot  |
| 2026-03-12 | Localized-moment symbolic guard check  | Pass   | Replaced broad exception fallback in localized moment degree handling with explicit polynomial validation in Irene/relaxations.py; added tests/test_quality_plan.py::TestRelaxationsFixes::test_localized_moment_rejects_non_polynomial_localizer; suite now passes (29 tests, OK) | Copilot  |
| 2026-03-12 | linear_combination guard check         | Pass   | Hardened Irene/program.py::linear_combination with explicit vertex/dimension/singularity validation; added tests/test_quality_plan.py::TestProgramFixes::test_linear_combination_rejects_missing_vertices and ::test_linear_combination_rejects_singular_vertex_matrix; suite now passes (33 tests, OK) | Copilot  |
| 2026-03-12 | LaTeX coupling remediation check       | Pass   | Decoupled Irene/base.py::LaTeX from runtime Irene imports via duck-typed __latex__ and SymPy Basic fallback; added tests/test_quality_plan.py::TestBaseFixes LaTeX coverage; suite now passes (33 tests, OK) | Copilot  |
| 2026-03-12 | Solver-discovery refactor check        | Pass   | Refactored Irene/base.py::AvailableSDPSolvers into table-driven platform-aware helper _solver_is_available; added tests/test_quality_plan.py::TestBaseFixes solver path-discovery coverage for non-Windows and Windows paths; suite now passes (35 tests, OK) | Copilot  |
| 2026-03-12 | pInitSDP stage-refactor check          | Pass   | Extracted duplicated commit/interrupt stage logic in Irene/relaxations.py into _commit_stage_state and replaced repeated blocks in pInitSDP stages; added tests/test_quality_plan.py::TestRelaxationsFixes commit-stage helper coverage; suite now passes (37 tests, OK) | Copilot  |

### Initial Findings Snapshot (Draft for Reviewer Approval: Approved)

1. Critical: Mutable default argument in SDP constructor can leak state across instances.

- File: Irene/sdp.py:30
- Evidence: def __init__(self, solver='cvxopt', solver_path=None)
- Risk: Shared dictionary default can cause cross-instance path pollution.
- Suggested fix: Use solver_path=None and initialize a new dict inside __init__.
- Status: Implemented on 2026-03-12 in Irene/sdp.py at lines 30 and 35.

2. High: Assertion-based solver validation can be disabled in optimized Python mode.

- File: Irene/sdp.py:31
- Evidence: explicit runtime check now enforces solver validity and raises ValueError for unsupported inputs.
- Risk: Input validation is skipped with python -O.
- Suggested fix: Replace assert with explicit conditional + ValueError.
- Status: Implemented on 2026-03-12 in Irene/sdp.py.

3. High: SDP sparse writer uses exact float equality checks.

- File: Irene/sdp.py:175 and Irene/sdp.py:185
- Evidence: sparse writer now uses abs(value) > sparse_zero_tol threshold checks.
- Risk: Near-zero numerical noise is serialized as nonzero coefficients.
- Suggested fix: Use tolerance comparison, for example abs(x) > 1e-12.
- Status: Implemented on 2026-03-12 in Irene/sdp.py with regression coverage in tests/test_quality_plan.py.

4. High: CSDP subprocess errors are swallowed and execution continues.

- File: Irene/sdp.py:487-490
- Evidence: csdp() now uses subprocess.run(..., check=True) and raises RuntimeError on execution failure before read_csdp_out.
- Risk: Missing or partial output file parsing after failed solver invocation.
- Suggested fix: capture and report subprocess failure, then stop parsing path.
- Status: Implemented on 2026-03-12 in Irene/sdp.py with regression coverage in tests/test_quality_plan.py.

5. High: Parallel initialization in relaxations lacks explicit process cleanup.

- File: Irene/relaxations.py:576-710
- Evidence: parallel Calpha work now flows through a helper that joins started workers and terminates them on failure.
- Risk: Resource leakage and unstable behavior on exception paths.
- Suggested fix: Add deterministic process lifecycle management in each stage.
- Status: Implemented on 2026-03-12 in Irene/relaxations.py with regression coverage in tests/test_quality_plan.py.

6. High: State persistence methods use open without context managers.

- File: Irene/relaxations.py:824-849
- Evidence: Resume, SaveState, State open files directly and close inconsistently.
- Risk: file-handle leak and brittle interruption behavior.
- Suggested fix: Replace with with open(...) blocks and explicit error handling.
- Status: Implemented on 2026-03-12 in Irene/relaxations.py at lines 723, 828, 835, 842.

7. Medium: linear_combination assumes vertices and solve matrix are valid.

- File: Irene/program.py:492-496
- Evidence: direct access to self.vertices[0] and np.linalg.solve(A, point)
- Risk: IndexError or LinAlgError for degenerate/singular cases.
- Suggested fix: Add guards for empty vertices and singular systems.

8. Medium: base.LaTeX introduces tight runtime coupling via in-function imports.

- File: Irene/base.py:14
- Evidence: from Irene import SDPRelaxations, SDRelaxSol, Mom
- Risk: import-time coupling and hard-to-diagnose circular import behavior.
- Suggested fix: move to safer type checking strategy with narrower dependencies.

### Optimization and Readability Candidates (Draft)

1. Medium: Dense nested loops in sparse SDPA writer scale poorly.

- File: Irene/sdp.py:167-186
- Evidence: triple nested iteration over block entries with scalar checks.
- Opportunity: Iterate non-zero entries only or use sparse-aware traversal.
- Effort: Medium.

2. Medium: CvxOpt matrix assembly creates avoidable intermediates.

- File: Irene/sdp.py:414-433
- Evidence: list builds + matrix(Ablock).transpose() + reshape path.
- Opportunity: use direct ndarray construction and fewer temporary containers.
- Effort: Medium.

3. Medium: pInitSDP has duplicated stage logic and broad exception patterns.

- File: Irene/relaxations.py:576-710
- Evidence: repeated process spawn/gather/commit pattern in multiple stages.
- Opportunity: extract stage helper and unified commit/error path.
- Effort: Medium to Large.

4. Low: Solver discovery has repeated platform branches.

- File: Irene/base.py:64-92
- Evidence: near-duplicate solver checks in win32 vs non-win32 branches.
- Opportunity: centralize per-solver checks in a table-driven helper.
- Effort: Small.

5. Medium: linear_combination readability and robustness can improve together.

- File: Irene/program.py:492-496
- Evidence: implicit origin-special-case and direct solve path.
- Opportunity: make vertex filtering explicit and return actionable error context.
- Effort: Small.

### Prioritized Findings List (Phase 5 Draft)

Severity ordering: Critical, High, Medium, Low. Within each severity, order is based on expected runtime impact and fix urgency.

#### Critical

1. Mutable default argument in solver constructor.

- File: Irene/sdp.py:30
- Evidence: solver_path=None in __init__ signature and defensive copy via self.Path = dict(solver_path).
- Failure mode: cross-instance shared state may leak path updates between solver objects.
- Fix direction: change default to None and instantiate a fresh dict in __init__.
- Test recommendation: instantiate two sdp objects and mutate one path; verify the other remains unchanged.
- Implementation status: Completed in Irene/sdp.py with regression coverage in tests/test_quality_plan.py.

#### High

1. Validation implemented with assert, not runtime checks.

- File: Irene/sdp.py:31
- Evidence: constructor now validates solver via explicit conditional and raises ValueError.
- Failure mode: optimized Python (-O) strips assert and skips validation.
- Fix direction: replace with explicit conditional and ValueError.
- Test recommendation: invalid solver should always raise, independent of optimization flags.
- Implementation status: Completed in Irene/sdp.py with regression coverage in tests/test_quality_plan.py.

2. Subprocess failure swallowed in CSDP path.

- File: Irene/sdp.py:486-488
- Evidence: csdp() now wraps subprocess.run(..., check=True) and raises RuntimeError before parser execution.
- Failure mode: parser is executed even when solver call fails, causing misleading downstream errors.
- Fix direction: capture exception details and raise a domain-specific runtime error before parsing output.
- Test recommendation: mock subprocess.run to fail and assert a controlled runtime error while parser execution is skipped.
- Implementation status: Completed in Irene/sdp.py with regression coverage in tests/test_quality_plan.py.

3. No return-code validation in SDPA subprocess path.

- File: Irene/sdp.py:471
- Evidence: sdpa() now wraps subprocess.run(..., check=True) and raises RuntimeError before parser execution.
- Failure mode: failed SDPA execution can be treated as success and parsed anyway.
- Fix direction: use subprocess.run(..., check=True) or verify return code and handle failure explicitly.
- Test recommendation: mock subprocess return non-zero and assert graceful failure path.
- Implementation status: Completed in Irene/sdp.py with regression coverage in tests/test_quality_plan.py.

4. Float equality checks in sparse export path.

- File: Irene/sdp.py:171 and Irene/sdp.py:181
- Evidence: tolerance-based comparisons using abs(value) > sparse_zero_tol.
- Failure mode: numerical noise around zero inflates sparse output and can perturb solver behavior.
- Fix direction: compare against tolerance threshold.
- Test recommendation: matrix entries near 1e-14 should be treated as zero under configured tolerance.
- Implementation status: Completed in Irene/sdp.py with regression coverage in tests/test_quality_plan.py.

5. Parallel process lifecycle lacks explicit cleanup.

- File: Irene/relaxations.py:580-624 and Irene/relaxations.py:576-710
- Evidence: `_parallel_calpha_results()` now centralizes worker startup/collection and guarantees join-on-success plus terminate-and-join on exceptions.
- Failure mode: orphaned processes and unstable interrupt/error behavior.
- Fix direction: enforce join with timeout and terminate-on-failure cleanup in every stage.
- Test recommendation: run pInitSDP on a small case and assert all child processes are cleaned up on success and simulated failure.
- Implementation status: Completed in Irene/relaxations.py with regression coverage in tests/test_quality_plan.py.

6. File-based state persistence uses manual open/close patterns in critical paths.

- File: Irene/relaxations.py:723 and Irene/relaxations.py:828-849
- Evidence: open(...) used directly for save/resume/state operations.
- Failure mode: leaked handles and partial writes under interruption.
- Fix direction: move to with open(...) and add explicit exception-safe persistence behavior.
- Test recommendation: interrupt simulation around SaveState/Resume with temporary files and verify file integrity.
- Implementation status: Completed in Irene/relaxations.py (context-managed binary pickle I/O).

#### Medium

1. parse_solution_matrix termination logic relies on row state assumptions.

- File: Irene/sdp.py:199-217
- Evidence: parse_solution_matrix now validates row shape/counts and raises ValueError on incomplete or inconsistent matrix blocks.
- Failure mode: malformed iterator content can break parsing flow or return invalid partial matrices.
- Fix direction: guard row before startswith checks and add strict parser state validation.
- Test recommendation: feed truncated and malformed SDPA snippets and assert controlled parse errors.
- Implementation status: Completed in Irene/sdp.py with regression coverage in tests/test_quality_plan.py.

2. CSDP output parsing uses fragile whitespace splitting.

- File: Irene/sdp.py:359
- Evidence: read_csdp_out now uses split() tokenization and validates vector/row field counts before parsing.
- Failure mode: variable whitespace can produce empty tokens and parse failures.
- Fix direction: use split() without explicit delimiter and validate token lengths.
- Test recommendation: parse outputs with irregular spacing and trailing spaces.
- Implementation status: Completed in Irene/sdp.py with regression coverage in tests/test_quality_plan.py.

3. Constraint type check uses identity operator.

- File: Irene/relaxations.py:286
- Evidence: AddConstraint now uses isinstance-based relation checks (including equality) instead of identity comparison.
- Failure mode: identity check may fail for equivalent but non-identical objects.
- Fix direction: replace identity check with equality or explicit sympy relation check.
- Test recommendation: cover equivalent relation objects from separate construction paths.
- Implementation status: Completed in Irene/relaxations.py with regression coverage in tests/test_quality_plan.py.

4. Localized moment degree fallback swallows symbolic errors.

- File: Irene/relaxations.py:574-578, Irene/relaxations.py:637 and Irene/relaxations.py:658
- Evidence: localized moment code now routes degree extraction through _poly_total_degree_or_raise(...) and raises ValueError on non-polynomial localizers.
- Failure mode: malformed symbolic inputs silently become degree zero and corrupt downstream structure.
- Fix direction: catch specific exceptions, preserve context, and fail fast for invalid symbolic state.
- Test recommendation: invalid symbolic term should raise a typed error instead of silently continuing.
- Implementation status: Completed in Irene/relaxations.py with regression coverage in tests/test_quality_plan.py.

5. linear_combination assumes vertices exist and solve matrix is nonsingular.

- File: Irene/program.py:468-521
- Evidence: linear_combination now validates vertex availability, non-origin active vertices, point dimension, square solve matrix, and singular matrix failures.
- Failure mode: empty/degenerate vertex sets raise unhelpful IndexError or LinAlgError.
- Fix direction: precondition checks for vertex availability and rank; provide explicit user-facing error.
- Test recommendation: add degenerate polytope and empty-vertex cases.
- Implementation status: Completed in Irene/program.py with regression coverage in tests/test_quality_plan.py.

6. Runtime coupling in LaTeX helper via in-function imports.

- File: Irene/base.py:9-18
- Evidence: LaTeX now uses duck-typed __latex__ detection and SymPy Basic fallback with no runtime import from Irene package.
- Failure mode: tighter module coupling and potential circular import side effects.
- Fix direction: decouple type checks via protocol-like behavior or local lightweight checks.
- Test recommendation: validate LaTeX behavior with mocked object types and SymPy objects.
- Implementation status: Completed in Irene/base.py with regression coverage in tests/test_quality_plan.py.

#### Low

1. Repeated solver availability branching logic.

- File: Irene/base.py:62-96
- Evidence: solver detection now uses table-driven helper _solver_is_available with shared per-solver mapping across platforms.
- Failure mode: maintainability overhead and inconsistent future updates.
- Fix direction: table-driven solver check helper.
- Test recommendation: unit tests for solver discovery matrix (platform x solver).
- Implementation status: Completed in Irene/base.py with regression coverage in tests/test_quality_plan.py.

2. Readability and duplication in stage assembly code.

- File: Irene/relaxations.py:551-695
- Evidence: duplicated stage commit/interrupt blocks are now centralized via _commit_stage_state and reused across pInitSDP stages.
- Failure mode: difficult maintenance and review overhead.
- Fix direction: extract reusable stage executor helper and unify error handling.
- Test recommendation: ensure serial/parallel stage outputs are equivalent on small fixtures.
- Implementation status: Completed in Irene/relaxations.py with regression coverage in tests/test_quality_plan.py.

### Test Coverage Gaps Mapped to Risk

1. No interruption/persistence integrity tests for Irene/relaxations.py save/resume/state paths.
2. Geometry edge-case tests for Irene/program.py remain limited to linear_combination; convex decomposition paths still need broader coverage.

### Recommended Remediation Sequence

1. Correctness follow-up: persistence integrity tests and broader convex decomposition edge-case coverage.
2. Optional integration pass: solver-dependent validation on representative example scripts.

### Risks and Mitigations

| Risk                                                    | Impact | Likelihood | Mitigation                                                        | Owner   | Status |
| ------------------------------------------------------- | ------ | ---------- | ----------------------------------------------------------------- | ------- | ------ |
| Solver-dependent failures not exercised in this pass    | Medium | Medium     | Add explicit second-pass solver integration review after approval | Copilot | Open   |
| Parallel/IO error paths may need targeted repro scripts | High   | Medium     | Use lightweight focused probes and add tests for malformed inputs | Copilot | Open   |
| Large methods make findings triage noisy                | Medium | High       | Prioritize issues by user impact and reproducibility first        | Copilot | Open   |

### Sign-off

- Technical Owner: TBD
- Primary Reviewer: TBD
- Secondary Reviewer: TBD
- Final Approval Date: TBD

### References

- Source modules:
  - Irene/base.py
  - Irene/relaxations.py
  - Irene/sdp.py
  - Irene/program.py
- Existing tests:
  - tests/test_quality_plan.py
  - tests/test_sonc_section3.py
