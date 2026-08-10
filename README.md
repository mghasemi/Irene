# Irene — Scientific Python Toolkit for Polynomial Optimization

A comprehensive toolkit for polynomial optimization, implementing SDP/SOS/SONC hierarchies, geometric programming relaxations, and circuit polynomial methods. Serves as the shared computational backend for the **Mean Polynomial** and **Differential SDP** research projects.

## Features

- **SDP/SOS Hierarchies**: Lasserre moment/SDP relaxation hierarchy
- **SONC Relaxations**: Constrained SONC via geometric programming with barycentric weights
- **SOSONC**: Combined SOS + SONC bounds
- **Geometric Programming**: GP-based relaxations with transformation matrices
- **Semigroup Algebras**: Commutative semigroups with derivation operator support
- **Invariant Theory**: Group-invariant polynomial optimization
- **Multiple Solvers**: CVXOPT (native), DSDP, SDPA, CSDP (via CLI)

## Modules

| Module | Role |
|--------|------|
| `grouprings.py` | Commutative semigroups & algebras (`CommutativeSemigroup`, `SemigroupAlgebra` with `add_derivative()`) |
| `program.py` | Optimization problem definition (`OptimizationProblem`) |
| `relaxations.py` | SDP moment/SOS hierarchy (`SDPRelaxations`) |
| `geometric.py` | Geometric programming relaxations (`GPRelaxations`) |
| `sonc.py` | Constrained SONC relaxations (`SONCRelaxations`) |
| `sosonc.py` | SOS + SONC combined bounds (`SOSONCRelaxations`) |
| `sdp.py` | SDP solver interface (`sdp`) |
| `matrices.py` | Moment/localizing matrix utilities |
| `invariant.py` | Group-invariant optimization extensions |
| `base.py` | LaTeX output helpers and solver discovery |

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify available solvers
python -c "from Irene.base import base; print(base().AvailableSDPSolvers())"
```

## Structure

```
Irene/
├── Irene/              ← package modules
├── MeansResearch/      ← Mean Polynomial experiments
├── pyProximation/      ← rational approximation, interpolation, orthosystems
├── tests/              ← test suite
├── examples/           ← example scripts
├── scripts/            ← utility scripts
└── doc/                ← API documentation (RST)
```

## Used By

- **Mean Polynomial (MP)**: Main research project — uses Irene for SDP/SONC/MP hierarchy experiments
- **Differential SDP (DSDP)**: Uses Irene's semigroup algebra with derivation support for ADE-constrained optimization (planned)
