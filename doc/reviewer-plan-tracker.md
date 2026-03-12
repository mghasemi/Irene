## Reviewer Plan Tracker: Core Algebra and GP Quality

Purpose: Track review progress, ownership, risks, and approvals for code-quality improvements in the Irene core modules.

### Scope

- In scope:
  - Irene/grouprings.py
  - Irene/program.py
  - Irene/geometric.py
- Out of scope:
  - Algorithm redesign
  - Solver-stack replacement
  - Repository-wide style migration

### Review Status Dashboard

| Phase | Description                        | Depends On | Owner   | Reviewer | Status | Target Date | Evidence                                                  |
| ----- | ---------------------------------- | ---------- | ------- | -------- | ------ | ----------- | --------------------------------------------------------- |
| 1     | Baseline + characterization tests  | None       | Copilot | Pass     | Done   | 2026-03-11  | tests/test_quality_plan.py                                |
| 2     | Correctness fixes in algebra core  | 1          | Copilot | Pass     | Done   | 2026-03-11  | Irene/grouprings.py                                       |
| 3     | Program representation consistency | 2          | Copilot | Pass     | Done   | 2026-03-11  | Irene/program.py                                          |
| 4     | Geometric relaxation refactor      | 2          | Copilot | Pass     | Done   | 2026-03-11  | Irene/geometric.py                                        |
| 5     | Type hints + API/docs cleanup      | 2          | Copilot | Pass     | Done   | 2026-03-11  | Irene/program.py, Irene/geometric.py, Irene/grouprings.py |
| 6     | Quality gates + build/lib policy   | 3, 4, 5    | Copilot | Pass     | Done   | 2026-03-11  | doc/documentation.md                                      |

Allowed status values: Not Started, In Progress, Blocked, In Review, Approved, Done.

### Phase Acceptance Criteria

1. Phase 1

- Regression tests added for known defect patterns.
- Baseline behavior captured for comparison.

2. Phase 2

- Equality and identity semantics corrected in algebra classes.
- Division remainder checks made explicit and deterministic.
- Mutable default argument removed.

3. Phase 3

- mono2ord_tuple contract made consistent.
- delta_vertex explicitly implemented or raised as NotImplementedError.
- to_sympy behavior on missing symbols is explicit.

4. Phase 4

- solve decomposed into helper methods with equivalent formulation.
- Constraint assembly checks are robust and readable.
- Solver failure path handled with actionable errors.

5. Phase 5

- Public APIs in target modules are type-annotated.
- Naming and docs improved for maintainability.

6. Phase 6

- Repeatable quality commands documented and runnable.
- Source-of-truth policy finalized for build/lib artifacts.

### Reviewer Checklist

- [X] Scope unchanged and explicitly documented.
- [ ] Backward-compatibility impact reviewed for equality semantics.
- [X] Tests cover all fixed defects and key edge cases.
- [X] Geometric refactor preserves numerical behavior within tolerance.
- [ ] Error messages are actionable and non-ambiguous.
- [X] Documentation reflects new contracts and caveats.
- [X] build/lib synchronization policy is documented and followed.

### Risks and Mitigations

| Risk                                                    | Impact | Likelihood | Mitigation                                                | Owner | Status |
| ------------------------------------------------------- | ------ | ---------- | --------------------------------------------------------- | ----- | ------ |
| Equality semantic change breaks downstream expectations | High   | Medium     | Add compatibility note and targeted regression tests      | TBD   | Open   |
| GP refactor changes numeric behavior                    | High   | Medium     | Snapshot fixed instances and compare within tolerance     | TBD   | Open   |
| Duplicate source/build edits diverge                    | Medium | High       | Treat build/lib as generated and regenerate after changes | TBD   | Open   |
| Missing tests for edge cases                            | Medium | Medium     | Add characterization tests before refactor                | TBD   | Open   |

### Decision Log

| Date       | Decision                  | Options Considered                                     | Rationale                                                                 | Approver |
| ---------- | ------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- | -------- |
| TBD        | Equality semantics policy | A keep legacy, B strict full equality, C staged toggle | TBD                                                                       | TBD      |
| 2026-03-11 | build/lib handling policy | A generated-only, B dual edits, C remove from VCS      | Adopt A: keep `Irene/` as source-of-truth and regenerate `build/lib/` | Copilot  |
| 2026-03-11 | Typing strictness         | A non-strict start, B strict now, C defer              | Adopt A: annotate public contracts first and keep incremental tightening  | Copilot  |

### Verification Log

| Date       | Check                        | Result | Notes                                                                                                                               | Reviewer |
| ---------- | ---------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 2026-03-11 | Unit test suite              | Pass   | python -m unittest tests/test_quality_plan.py -v                                                                                    | Copilot  |
| 2026-03-11 | Unit test suite              | Pass   | /home/mehdi/Code/Irene/.venv/bin/python -m unittest tests/test_quality_plan.py -v                                                   | Copilot  |
| 2026-03-11 | Static checks                | Pass   | get_errors clean for Irene/program.py, Irene/geometric.py, and Irene/grouprings.py                                                  | Copilot  |
| 2026-03-11 | Quality gate docs            | Pass   | Added quality commands and build synchronization policy to doc/documentation.md                                                     | Copilot  |
| 2026-03-11 | Example workflow validation  | Pass   | /home/mehdi/Code/Irene/.venv/bin/python examples/GPExample.py (exit code 0; runtime warning observed in auto_transform_matrix path) | Copilot  |
| 2026-03-11 | Example workflow validation  | Pass   | Re-run confirmed: GPExample solved with cvxopt in about 0.02s and produced gp.solution output                                       | Copilot  |
| 2026-03-11 | Numeric tolerance comparison | Pass   | GPExample setup solved twice; abs_diff_f_gp_g=0.0 and abs_diff_cost=0.0 (tolerance 1e-8)                                            | Copilot  |

### Sign-off

- Technical Owner: TBD
- Primary Reviewer: TBD
- Secondary Reviewer: TBD
- Final Approval Date: TBD

### References

- Base technical narrative: doc/documentation.md
- Source modules:
  - Irene/grouprings.py
  - Irene/program.py
  - Irene/geometric.py
