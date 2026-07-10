# Irene Package — Quick Reference API

**Location**: `/home/mehdi/Code/Python/Irene/Irene/`

## Module Index

| Module | Role | Key Classes/Functions |
|--------|------|---------------------|
| `grouprings.py` | Commutative semigroups & algebras | `CommutativeSemigroup(['x','y'])`, `SemigroupAlgebra(sg)`, `.generators`, `.order()`, `.ord()` |
| `program.py` | Optimization problem definition | `OptimizationProblem(sga)`, `.set_objective(f)`, `.add_constraints([c1,c2])` |
| `relaxations.py` | SDP moment/SOS hierarchy (Lasserre) | `SDPRelaxations(prog).Minimize(objective_index=0, relaxation_order=2, solver='cvxopt')` → `.Primal`, `.Dual`, `.Status`, `.Message`, `.RunTime` |
| `sdp.py` | SDP solver interface (CVXOPT/DSDP/SDPA/CSDP) | `sdp(solver)`, `.SetObjective(b)`, `.AddConstraintBlock(A_i)`, `.AddConstantBlock(C_j)`, `.solve()` → `val, dual_val, status, message, runtime` |
| `sonc.py` | Constrained SONC relaxations (GPs) | `SONCRelaxations(prog).solve(verbosity=0)` — returns lower bound via GP/signomial |
| `sosonc.py` | SOS + SONC combined bounds | `SOSONCRelaxations(prog).globalMinSOS()`, `.globalMinSONC()`, `.globalMinSOSPSONC(first='sos'|'sonc')` |
| `geometric.py` | Geometric programming relaxations | `GPRelaxations(prog, gp=None, H=None)` → transformed GP via matrix $H$ |
| `base.py` | LaTeX output helpers | `LaTeX` class for formatted mathematical output |

## SemigroupAlgebra — Critical Conventions

```python
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
sg = CommutativeSemigroup(['x', 'y'])   # STRING names, NOT sympy Symbols!
sga = SemigroupAlgebra(sg)
x = sga['x']                             # Key access by string
y = sga['y']

# Build expressions via SGA arithmetic (NOT sympy):
f = 1 + 2 * x**2 * y**4 - 3 * x**2 * y**2

# Order methods:
sg.generators                            # list of generators
sg.order()                               # total order on semigroup
```

**⚠️ Pitfalls:**
- `CommutativeSemigroup([Symbol('x')])` → FAIL. Use strings: `'x'`.
- `set_objective()` takes ONE `SemigroupAlgebraElement`, not sympy expression.
- SDP default `relaxation_order=1` is insufficient for quartics — bump to 2+.

## OptimizationProblem Setup

```python
from Irene.program import OptimizationProblem
prog = OptimizationProblem(sga)
prog.set_objective(f)                    # unconstrained objective
# prog.Minimize() → DOES NOT EXIST. Use set_objective().
g1 = y - x**4 * y + y**5 - x**6 - y**6
prog.add_constraints([g1])               # optional constraints
```

## Solver Backends

Verify with: `from Irene.base import base; print(base().AvailableSDPSolvers())`

| Backend | Interface | Notes |
|---------|-----------|-------|
| CVXOPT | Native Python | Default, always available in venv |
| DSDP/SDPA/CSDP | CLI wrappers | Require system binaries installed (`sudo apt install`) |

## SONC Relaxed Minimum via GPs

The constrained SONC relaxed minimum equals the optimal value of:
$$\min \sum_{i=1}^{n} \lambda_0^{(i)} b^{(i)} \quad \text{s.t. constraints on } \{\lambda, b\}$$

Where each circuit polynomial has a unique decomposition $f_i = \alpha_i x^{\beta_i} + \sum_j c_{ij}^{1/\lambda_0^{(i)}} g_{ij}(x)$ with $\lambda_0^{(i)} > 0$, and the value is:
$$\lambda_0^{(i)} b^{(i)} (c_{ij}/b^{(i)})^{1/\lambda_0^{(i)}} \cdot (\lambda_j / \alpha_i)^{\lambda_j}$$

## SOSONC Relaxation Strategy

Three strategies for combining SOS and SONC bounds:
1. **SOSPSONC**: Minimize over the sum of a SOS element and a SONC element (jointly)
2. **SOS-first**: Fix SOS part to optimal, then optimize SONC on residual
3. **SONC-first**: Fix SONC part to optimal, then optimize SOS on residual

## Testing

```bash
cd /home/mehdi/Code/Python/Irene && python3 -m pytest tests/test_sosonc.py -v
```
