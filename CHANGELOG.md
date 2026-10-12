# Changelog

All notable changes to Irene are documented here.
Versioning follows [Semantic Versioning](https://semver.org/).

## [1.5.0] — 2026-10-12

### Fixed

- **Newton-polytope over-pruning in relaxation bases** (decision #5): the pruner could silently drop monomials required for certification, producing invalid relaxations on some inputs. It now returns a certified-safe basis and falls back to the unpruned basis when no safe reduction is available. Verified: P3 values at orders 1–3 are finite and equal to baseline for Motzkin, Choi–Lam, and Robinson; full test suite passes (274 passed, 2 skipped, 1 xpassed).
- **Silent fallback in `newton_polytope()`**: non-ring inputs previously triggered an implicit fallback to the unpruned basis. The API now raises `TypeError`/`ValueError` with a clear message; internal pipeline call sites (which pass ring elements) are unaffected, and all benchmark cells remain byte-identical before/after the change.

### Changed

- Documentation examples in ``relaxation_api.rst`` and ``newton_polytope.rst`` were rewritten against measured output from the current code: estimated/fabricated numbers removed, and every documented value is now pinned by regression fixtures (``tests/test_documented_examples.py``) so CI fails if a documented number drifts.
- Lasserre citations corrected to Crossref-verified entries: *Convergent SDP-relaxations in polynomial optimization with sparsity*, SIAM J. Optim. 17(3), 822–843 (2006, DOI `10.1137/05064504x`); and *Global optimization with polynomials and the problem of moments*, SIAM J. Optim. 11(3), 796–817 (2001).

## [1.4.0]

Last release published to PyPI — see <https://pypi.org/project/Irene/> for the project page.
No detailed changelog was maintained before this file; refer to git history for changes up to 1.4.0.
