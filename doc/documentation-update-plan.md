# Documentation Update Plan (POP, Group-Rings, SDP/GP/SONC)

## Objective

Expand the documentation from an SDP-focused hierarchy to a unified constrained polynomial optimization (POP) guide that covers:

1. SDP relaxations (existing strength).
2. Geometric programming relaxations.
3. SONC relaxations.
4. The algebraic shift from polynomial-ring intuition to group-rings equipped with differential operators.

## Current Status

- [x] Planning complete.
- [x] Initial implementation started in Sphinx docs.
- [x] Theoretical expansion finalized.
- [x] Examples and validation workflow finalized.
- [x] Full editorial and build verification complete.

## Scope

### Included

- Documentation architecture and navigation.
- Theory-to-code mapping for `grouprings.py`, `program.py`, `geometric.py`, `sonc.py`.
- Method-selection guidance (when to use SDP vs GP vs SONC).
- API reference coverage expansion in `code.rst`.

### Excluded

- Algorithmic rewrites of optimization methods.
- Solver backend refactoring.

## Phased Plan

## Phase 1: Information Architecture

Deliverables:

1. Add a method overview chapter.
2. Add dedicated chapters for group-rings, problem representation, geometric POP, and SONC POP.
3. Update `index.rst` navigation to reflect the new conceptual flow.

Acceptance checks:

1. New chapters appear in the Sphinx toctree.
2. Reader can navigate from foundations to methods without leaving the main docs.

## Phase 2: Group-Ring Foundations and Problem Modeling

Deliverables:

1. Document `CommutativeSemigroup` and `SemigroupAlgebra` as core abstractions.
2. Explain derivation support (`add_derivative`, `derivative`, `diff`) and product-rule behavior.
3. Document `OptimizationProblem` data flow from symbolic representation to geometric/numeric routines.

Acceptance checks:

1. Core abstractions are described in narrative form and tied to code symbols.
2. Notation remains consistent with existing optimization chapters.

## Phase 3: Geometric and SONC Theory Expansion

Deliverables:

1. Add geometric-programming chapter based on Section 4 equation (3) implementation in `geometric.py`.
2. Add SONC chapter based on Section 3 constrained formulation and current implementation path in `sonc.py`.
3. Include theory-to-code mapping for key internal stages (`delta`, support points, barycentric weights, constraints, solve).

Acceptance checks:

1. Chapters reference both mathematical objects and corresponding implementation methods.
2. Example 3.3-style SONC workflow is documented and traceable.

## Phase 4: API Coverage and Onboarding

Deliverables:

1. Expand `code.rst` automodule coverage beyond `base`, `relaxations`, `sdp`.
2. Update installation/dependency guidance to clarify solver prerequisites and optional packages.
3. Add minimal validation sequence (imports, solver detection, and one runnable method per family).

Acceptance checks:

1. API docs include all active method families.
2. New users can run at least one SDP and one SONC/GP path with documented commands.

## Phase 5: Final Consistency and Verification

Deliverables:

1. Consistent notation across chapters (`K`, `G(mu)`, support and delta sets, lambda weights).
2. Sphinx build and warning cleanup.
3. Runtime verification with representative examples/tests.

Acceptance checks:

1. Documentation builds cleanly.
2. Example references align with actual behavior in the current codebase.

## Key Files

- `doc/index.rst`
- `doc/introduction.rst`
- `doc/optim.rst`
- `doc/sdp.rst`
- `doc/code.rst`
- `doc/grouprings_architecture.md`
- `doc/documentation.md`
- `Irene/grouprings.py`
- `Irene/program.py`
- `Irene/geometric.py`
- `Irene/sonc.py`
- `examples/Example01.py`
- `examples/GPExample.py`
- `examples/SONCExample.py`
- `examples/SONCExample33.py`
- `tests/test_quality_plan.py`
- `tests/test_sonc_section3.py`

## Implementation Log

- 2026-03-12: Added markdown plan and started Sphinx implementation by introducing new chapter skeletons and extending navigation/API coverage.
- 2026-03-12: Expanded theory chapters with method-selection/dependency matrices, SONC and GP equation-level mapping, and runnable examples documentation.
- 2026-03-12: Finalized theoretical expansion in algebra/program/geometric/sonc/optim chapters and verified warning-free Sphinx builds.
- 2026-03-12: Validation run completed with the following commands:
	- ``/home/mehdi/Code/Irene/.venv/bin/python examples/Example01.py`` (SDP path: success, optimal solver output observed).
	- ``/home/mehdi/Code/Irene/.venv/bin/python examples/GPExample.py`` (GP path: solved; runtime warning observed in transform ratio step).
	- ``/home/mehdi/Code/Irene/.venv/bin/python examples/SONCExample.py`` (SONC path: runtime infeasibility reported by GP model for this benchmark instance in current environment).
	- ``/home/mehdi/Code/Irene/.venv/bin/python -m unittest discover tests/`` (56 tests, all passed).