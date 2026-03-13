# Integration Plan: pyProximation into Irene

**Date:** 12 March 2026  
**Source:** `/home/mehdi/Code/pyProximation` (lean branch)  
**Target:** `/home/mehdi/Code/Irene`

---

## Decisions

| Decision | Choice |
|---|---|
| Integration method | Plain copy — `pyProximation/` folder placed beside `Irene/` at repo root |
| Documentation | Integrated as a new top-level section in the Irene Sphinx doc |
| `setup.py` | Merged — `pyProximation` added to `packages` list alongside `Irene` |
| PDF | Rebuild `doc/Irene.pdf` via `make latexpdf` |

---

## Phase 1 — Copy pyProximation package files

**Step 1.** Copy 6 Python source files from `/home/mehdi/Code/pyProximation/pyProximation/`
into a new `pyProximation/` directory at the repo root:

```
pyProximation/
  __init__.py
  base.py
  measure.py
  orthsys.py
  interpolation.py
  rational.py
```

---

## Phase 2 — Update setup.py

**Step 2.** In `setup.py`: add `'pyProximation'` to the `packages` list.  
`numpy`/`scipy`/`sympy` are already in `install_requires` — no further changes needed.

**Before:**
```python
packages=['Irene'],
```

**After:**
```python
packages=['Irene', 'pyProximation'],
```

---

## Phase 3 — Prepare documentation files

**Step 3.** Copy 5 `.rst` files from `/home/mehdi/Code/pyProximation/doc/` to `doc/`
with a `pyprox_` prefix to avoid name collisions with existing Irene docs:

| Source | Destination |
|---|---|
| `introduction.rst` | `doc/pyprox_intro.rst` |
| `measures.rst` | `doc/pyprox_measures.rst` |
| `hilbert.rst` | `doc/pyprox_hilbert.rst` |
| `interpolation.rst` | `doc/pyprox_interpolation.rst` |
| `code.rst` | `doc/pyprox_code.rst` |

The `code.rst` autodoc directives already use `pyProximation.xxx` module paths — no edits needed.

**Step 4.** Copy logo images from `/home/mehdi/Code/pyProximation/doc/images/` → `doc/images/`:
- `pyProxLogo.png`
- `pyProxLogoSmall.png`

**Step 5.** In `doc/appendix.rst`: remove the existing `pyProximationRef` stub section
(brief description + code snippet) and replace it with a `:ref:` cross-link pointing
to the new dedicated section, to avoid duplication.

---

## Phase 4 — Update Sphinx configuration

**Step 6.** In `doc/index.rst`: add a new toctree block for pyProximation, placed between
the `examples` entry and the `appendix`/`rev`/`todo` cluster:

```rst
.. toctree::
   :caption: pyProximation

   pyprox_intro
   pyprox_measures
   pyprox_hilbert
   pyprox_interpolation
   pyprox_code
```

**Step 7.** `doc/conf.py`: **no changes needed.**  
`sys.path` already points to the repo root (`os.path.abspath('..')`), covering both
`Irene/` and `pyProximation/`. `autodoc_mock_imports` is already complete —
pyProximation only uses numpy/scipy/sympy which are available.

---

## Phase 5 — Build and validate

**Step 8.** Run HTML build to surface any autodoc or cross-reference errors:
```bash
/home/mehdi/Code/Irene/.venv/bin/sphinx-build -b html doc doc/_build/html
```

**Step 9.** Fix any build warnings/errors surfaced by Step 8.

**Step 10.** Build the PDF:
```bash
cd /home/mehdi/Code/Irene/doc && make latexpdf
```
Output: `doc/_build/latex/Irene.pdf`

**Step 11.** Copy `doc/_build/latex/Irene.pdf` → `doc/Irene.pdf` to update the committed copy.

---

## Relevant Files (summary)

| File | Change |
|---|---|
| `setup.py` | Add `'pyProximation'` to `packages` |
| `pyProximation/` *(new)* | 6 source files copied from local clone |
| `doc/index.rst` | New toctree block for pyProximation section |
| `doc/appendix.rst` | Remove stub, add cross-ref |
| `doc/conf.py` | No changes required |
| `doc/pyprox_intro.rst` *(new)* | Copied + prefixed from pyProximation docs |
| `doc/pyprox_measures.rst` *(new)* | " |
| `doc/pyprox_hilbert.rst` *(new)* | " |
| `doc/pyprox_interpolation.rst` *(new)* | " |
| `doc/pyprox_code.rst` *(new)* | " |
| `doc/images/pyProxLogo.png` *(new)* | Logo for rendered docs |
| `doc/images/pyProxLogoSmall.png` *(new)* | Logo for LaTeX title page |
| `doc/Irene.pdf` | Rebuilt artifact |

---

## Verification Checklist

- [ ] `python -c "from pyProximation import Measure, OrthSystem"` — no error
- [ ] `python setup.py --version` succeeds; `packages` includes both `'Irene'` and `'pyProximation'`
- [ ] HTML build completes with zero errors
- [ ] All 5 `pyprox_*.rst` pages render with correct autodoc API tables
- [ ] `doc/Irene.pdf` table of contents includes the new pyProximation section
