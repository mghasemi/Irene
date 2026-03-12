# Irene 1.2.5 Release Notes

Date: 2026-03-12

## Summary

Version 1.2.5 is a documentation-focused release that broadens Irene's presentation from an SDP-centric narrative to a unified constrained polynomial optimization (POP) framework across SDP, geometric programming, and SONC relaxations.

## Highlights

- Added an architecture chapter describing the relationship between algebra, problem formulation, relaxation methods, and solver backends.
- Added explicit group-ring and differential-operator foundations.
- Added an optimization problem representation chapter connecting symbolic and geometric data paths.
- Added dedicated geometric programming and SONC chapters with equation-level theory-to-code mapping.
- Added runnable examples and validation guidance for SDP, GP, and SONC workflows.
- Expanded API reference coverage to include `program`, `grouprings`, `geometric`, and `sonc` modules.
- Added dependency matrix and solver troubleshooting guidance in the introduction.

## New/Updated Documentation Files

- `doc/architecture.rst`
- `doc/algebra.rst`
- `doc/program.rst`
- `doc/geometric.rst`
- `doc/sonc.rst`
- `doc/examples.rst`
- `doc/documentation-update-plan.md`
- `doc/index.rst`
- `doc/introduction.rst`
- `doc/code.rst`
- `doc/optim.rst`
- `doc/rev.rst`
- `doc/conf.py`

## Notes

- This release does not introduce algorithmic behavior changes in optimization routines.
- Documentation was validated with a clean Sphinx build using the project virtual environment.
