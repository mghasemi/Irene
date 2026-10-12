# Contributing to Irene

Thank you for your interest in contributing! Irene is a Python package for
global optimization of polynomials (SDP/GP relaxations, SONC hierarchies).
Issues and pull requests are welcome.

## Development setup

Python >= 3.11 is required (the pinned scientific stack — numpy/scipy/cvxpy —
declares `Requires-Python >=3.11`).

```bash
git clone https://github.com/mghasemi/Irene.git
cd Irene
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

- `pip install -e ".[solvers]"` — SDP/GP solvers (Clarabel, SCS, OSQP).
  Clarabel and CVXOPT are the primary backends; others work where supported.
- `pip install -e ".[symengine]"` — SymEngine C++ symbolic backend (default
  when installed; falls back to SymPy). Switch at runtime with
  `Irene.symbolic_engine.set_symbolic_backend(...)` or the environment variable
  `IRENE_SYMBOLIC_BACKEND=sympy|symengine|auto`.

## Running tests

From the repository root:

```bash
python -m pytest Irene/tests/ tests/ --tb=short -q
```

This mirrors what CI runs (matrix: Python 3.11/3.12 × Clarabel/SCS). For a
Docker-based run matching CI exactly, see `docker-compose.yml` /
`ci_entrypoint.sh`:

```bash
docker compose run <service> test        # pytest with coverage
docker compose run <service> benchmark   # gallery quick-mode
```

## Submitting changes

1. Fork the repository and create a feature branch from `master`.
2. Keep the unified API contract intact: relaxation entry points return the
   documented result fields (`value`, `status`, `method`, `error_code`,
   `runtime`, `init_time`, `message`, `certificate`, `solver_info`). If you
   change that contract, update `doc/relaxation_api.rst` and the examples in
   the same pull request.
3. Add or adjust tests under `tests/`; documented-example fixtures pin the
   numbers shown in the docs, so keep them consistent.
4. Open a pull request against `master`.

## License

Irene is released under the MIT License (see `LICENSE`). By contributing, you
agree that your contributions will be licensed under the same terms.
