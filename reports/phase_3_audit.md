# Phase 3 Audit — Advanced Algebraic Reductions

**Date:** 2026-08-07  
**Task:** P3.1 Research and prototype border basis algorithm + sparsity insertion points

---

## Current Relaxation Pipeline (baseline)

```
SDPRelaxations(gens, relations)
    │
    ├── SetObjective(obj) → RedObjective via ReduceExp()
    ├── AddConstraint(cnst) → Constraints list
    ├── MomentsOrd(d) → MmntOrd = d
    ├── RelaxationDeg() → auto-compute required order
    │
    └── InitSDP()
            │
            ├── ExponentsVec(2*d) → all monomials up to degree 2d, reduced via Groebner
            ├── LocalizedMoment(p) for each constraint p
            │       └── ReducedMonomialBase(d - deg(p)/2) ← FULL basis, no pruning
            │           └── product(range(d+1), repeat=n) → ALL exponent tuples
            │               └── filter(sum ≤ d) → still exponential in n,d
            │
            ├── Build SDP matrices (self.SDP.SetObjective/AddConstraintBlock/AddConstantBlock)
            └── sdp.solve() → CVXPY CLARABEL/SCS backend
```

## Key Bottleneck: `ReducedMonomialBase(deg)` — Line 334–354

```python
all_monos = product(range(deg + 1), repeat=self.NumGenerators)  # (deg+1)^n tuples
req_monos = filter(lambda x: sum(x) <= deg, all_monos)         # still O(n^d) after filter
monos = [reduce(mul, [...], 1) for expn in req_monos]          # create symbolic monomials
for expr in monos:
    rexpr = self.ReduceExp(expr)                               # Groebner reduction per monomial
```

**Problem:** Generates the FULL monomial basis up to degree `deg`, then reduces each via Groebner. For n=3, d=4 this is 126 monomials; for n=5, d=6 it's 3003. The Groebner reduction step dominates runtime.

---

## Insertion Point A: Newton Polytope Pruning (P3.5)

### Where
Replace `product(range(deg+1), repeat=n)` in `ReducedMonomialBase()` with a filtered set based on the Newton polytope of the objective and constraints.

### How
```python
def _newton_support(self, expr):
    """Return the set of exponent vectors appearing in expr."""
    return set(engine.Poly(expr, *self.AuxSyms).as_dict().keys())

def _pruned_monomials(self, deg):
    """Only generate monomials inside the Minkowski sum of Newton polytopes."""
    # Union of supports from objective + constraints
    support = self._newton_support(self.RedObjective)
    for c in self.Constraints:
        support |= self._newton_support(c)
    # Generate only exponent tuples within the convex hull of 2·support ∩ [0,d]^n
    ...
```

### Expected impact
For sparse polynomials (common in polynomial optimization benchmarks), this can reduce basis size by 60–90%. The Motzkin polynomial x⁴y + xy⁴ - 3x²y² + 1 has only 4 terms but the full degree-4 basis in 2 variables has 15 monomials.

---

## Insertion Point B: Border Basis (P3.1–P3.2)

### Where
Replace Groebner-based reduction in `ReduceExp()` and `ReducedMonomialBase()`.

### Theory
Border bases generalize Gröbner bases to non-zero-dimensional ideals. For the moment problem, they provide a basis of the quotient algebra A/I that is numerically stable (unlike Gröbner bases which can be ill-conditioned). Key reference: Traverso (1990), Greuel & Pfister (2002) "A Border Basis Algorithm".

### Implementation plan
```python
# Irene/border_basis.py
class BorderBasis:
    """Border basis for the quotient algebra R[x]/I."""
    
    def __init__(self, generators, relations, degree):
        # Compute border basis via multiplication table method
        self.basis = [...]          # standard monomials (non-leading)
        self.border = [...]         # border = {b·x_i : b in basis, i=1..n} \ basis
        self.mult_tables = {...}    # multiplication maps for each variable
    
    def reduce(self, expr):
        """Reduce expression using border basis multiplication tables."""
        ...
    
    def moment_matrix_structure(self):
        """Return the block structure induced by the border basis."""
        ...
```

### Integration with relaxations.py
- `ReducedMonomialBase()` → use `BorderBasis.basis` instead of full product
- `ReduceExp()` → use `BorderBasis.reduce()` instead of Groebner reduction
- This is a **drop-in replacement** — the SDP construction code downstream doesn't change

---

## Insertion Point C: Correlative Sparsity (P3.4)

### Where
After `InitSDP()`, before `sdp.solve()`. Detect which variables appear together in constraints and partition the moment matrix into smaller blocks.

### How
```python
def _detect_correlative_sparsity(self):
    """Build a variable co-occurrence graph from objective + constraints."""
    # For each polynomial, find which variables actually appear (nonzero coeff)
    var_graph = nx.Graph()
    for expr in [self.RedObjective] + self.Constraints:
        support = set(engine.Poly(expr, *self.AuxSyms).as_dict().keys())
        active_vars = {i for exp in support if exp[i] > 0}
        for v1, v2 in product(active_vars, repeat=2):
            var_graph.add_edge(v1, v2)
    # Connected components give the sparsity pattern
    return list(nx.connected_components(var_graph))
```

### Expected impact
If variables {x₁,x₂} and {x₃,x₄} never appear together in any constraint, the moment matrix decomposes into two smaller blocks. For n=10 with 2 cliques of size 5, this reduces SDP block sizes from ~O(10^d) to ~O(5^d) each — exponential savings.

---

## Insertion Point D: Unified Relaxation API (P3.3)

### Current state
`SDPRelaxations` is the monolithic class handling everything. No way to swap in border basis vs Groebner, or enable/disable sparsity detection.

### Design
```python
class RelaxationConfig:
    """Configuration for relaxation construction."""
    reduction_method: str = "groebner"  # or "border_basis"
    monomial_pruning: bool = False      # Newton polytope pruning
    sparsity_detection: bool = False    # correlative sparsity
    solver: str = "CLARABEL"

class SDPRelaxations(base):
    def __init__(self, gens, relations=(), config=None):
        self.config = config or RelaxationConfig()
        ...
    
    def _build_monomial_basis(self, deg):
        """Dispatch to appropriate basis builder based on config."""
        if self.config.reduction_method == "border_basis":
            return self._border_basis_build(deg)
        else:
            return self.ReducedMonomialBase(deg)  # current Groebner path
```

---

## Recommended Phase 3 Execution Order

| Priority | Task | Rationale |
|----------|------|-----------|
| 1 | P3.5 Newton polytope pruning | Lowest effort, highest immediate impact — filters monomials before any expensive computation |
| 2 | P3.4 Correlative sparsity detection | Pure graph analysis on existing polynomial data; no new algebra needed |
| 3 | P3.3 Unified relaxation API | Enables clean integration of all reductions behind config flags |
| 4 | P3.1–P3.2 Border basis research + implementation | Highest theoretical value but most complex — do last after infrastructure is in place |

---

## Files to Modify for Phase 3

| File | Change | Tasks |
|------|--------|-------|
| `Irene/relaxations.py` | Add `_newton_support()`, `_pruned_monomials()`; wire into `ReducedMonomialBase()` | P3.5 |
| `Irene/relaxations.py` | Add `_detect_correlative_sparsity()`; partition SDP blocks | P3.4 |
| `Irene/border_basis.py` | New file — `BorderBasis` class with multiplication table method | P3.2 |
| `Irene/relaxations.py` | Add `RelaxationConfig`; dispatch in `_build_monomial_basis()` | P3.3 |
