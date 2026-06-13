# sosonc.py — Implementation Plan

## Overview
New Irene module implementing the SOS+SONC two-step optimization algorithms from Moritz Schick's PhD thesis,
translated from MATLAB (`globalMinSOSPSONC.m`, `globalMinSOS.m`, `globalMinSONC.m`) to Python.
Integrates with Irene's existing `SDPRelaxations`, `SONCRelaxations`, and `OptimizationProblem` classes.

## Class: `SOSONCRelaxations`

```
class SOSONCRelaxations(object):
    """Combined SOS+SONC relaxation framework for unconstrained polynomial optimization.

    Implements three methods corresponding to Schick's toolbox:
      - globalMinSOS: pure SOS relaxation (via SDP)
      - globalMinSONC: pure SONC relaxation (via signomial GP)
      - globalMinSOSPSONC: two-step SOS+SONC relaxation (Algorithm 4/5)

    Also provides preprocessing methods with configurable convex distance functions.
    """
```

## API Design

### Constructor
```python
def __init__(self, prog: OptimizationProblem, **kwargs):
    - prog: Irene OptimizationProblem instance
    - error_bound (1e-10): numerical zero tolerance
    - verbosity (1): logging level
    - solver: SDP solver name ('cvxopt', 'csdp', 'sdpa', 'dsdp')
    - use_local_solve (True): use signomial GP local solve for SONC portion
```

### Core Methods

1. **`globalMinSOS(self) -> SOSONCRelaxSol`**
   - Translates Schick's `globalMinSOS.m`
   - Uses Irene's existing `SDPRelaxations.Minimize()` internally
   - Computes `lambda* = sup{λ : f-λ ∈ Σ}` via Gram matrix / moment SDP
   - Returns solution with val, f_sos, error_code

2. **`globalMinSONC(self) -> SOSONCRelaxSol`**
   - Translates Schick's `globalMinSONC.m`
   - Uses Irene's existing `SONCRelaxations.solve()` internally
   - Computes `lambda* = sup{λ : f-λ ∈ C}` via signomial GP
   - Returns solution with val, f_sonc, certificate

3. **`globalMinSOSPSONC(self, method='two-step-sos-first') -> SOSONCRelaxSol`**
   - Translates Schick's `globalMinSOSpSONC.m`
   - Two strategies:
     a. `'two-step-sos-first'` (Algorithm 4): SOS preprocessing → SONC relaxation
     b. `'two-step-sonc-first'` (Algorithm 5): SONC preprocessing → SOS relaxation
     c. `'joint'` (direct): combine SOS + SONC certificates (uses relative entropy + Gram matrix — requires MOSEK/exponential cone solver; may fall back)
   - Default: two-step-sos-first (Algorithm 4)

### Preprocessing Methods (Algorithms 4 & 5)

4. **`_sos_preprocess(self, f, degree_bound) -> (f_minus_g, g_star)`**
   - Algorithm 4, Step 1: finds `g* ∈ Σ` minimizing `φ(f, g)`
   - `φ(f, g) = ||f - g||_2` over coefficients (convex distance)
   - Returns `f - g*` (for subsequent SONC relaxation) and `g*`

5. **`_sonc_preprocess(self, f, degree_bound) -> (f_minus_g, g_star)`**
   - Algorithm 5, Step 1: finds `g* ∈ C` minimizing `ψ(f, g)`
   - `ψ(f, g) = ||f - g||_2` over coefficients
   - Returns `f - g*` (for subsequent SOS relaxation) and `g*`

## Auxiliary Methods

6. **`newton_polytope(self, poly)`**
   - Translates Schick's `newtonPolytope.m`
   - Computes convex hull of exponent vectors, returns monomials and exponent matrix
   - Uses Irene's existing `prog.newton()` from `program.py`

7. **`half_newton_polytope(self, poly, exponents)`**
   - Translates Schick's `halfNewtonPolytope.m`
   - Returns monomials of degree ≤ deg(poly)/2 whose exponents are in the half-Newton polytope

8. **`_signomial_representative(self, poly, monomials, exponents)`**
   - Translates Schick's `sigRep.m`
   - Identifies which monomials are "inner terms" (odd exponents, even lattice points)
   - Returns the index set of inner terms

9. **`_const_sonc(self, coeff, exponents, inner_term_indices)`**
   - Translates Schick's `constSONC.m`
   - Builds the relative-entropy programming constraint set for SONC membership
   - Uses Irene's gpkit signomial infrastructure or CVXPY exponential cone

## Result Class: `SOSONCRelaxSol`

```python
class SOSONCRelaxSol(object):
    """Carries optimization results for SOS+SONC relaxations."""
    - val: float                # optimal λ* (lower bound on f*)
    - method: str               # 'sos', 'sonc', 'sos+sonc'
    - f_sos: Optional[Any]      # SOS certificate polynomial
    - f_sonc: Optional[Any]     # SONC certificate polynomial
    - sos_sol: SDRelaxSol       # SDP solution (if SOS was used)
    - sonc_sol: float           # SONC GP optimal value (if used)
    - status: str               # 'optimal', 'infeasible', 'error'
    - runtime: float
    - error_code: int           # 0=success, 1=infeasible, 2=error
```

## Implementation Strategy

### Phase 1: Global solutions via existing Irene infrastructure
- `globalMinSOS` → wrap `SDPRelaxations.Minimize()` 
- `globalMinSONC` → wrap `SONCRelaxations.solve()`
- These are straightforward delegations to existing code.

### Phase 2: Preprocessing algorithms
- `_sos_preprocess` → solve SDP to find g* ∈ Σ, then compute f-g*
- `_sonc_preprocess` → solve GP/signomial to find g* ∈ C, then compute f-g*
- The distance function φ/ψ: `||f - g||_2` over the coefficient vector, which is convex.

### Phase 3: Two-step SOS+SONC
- SOS-preprocess → SONC-relaxation (Algorithm 4)
- The lower bound: `(f-g*)_C*` where g* ∈ Σ, and f = g* + (f-g*) gives f ∈ Σ + C
- SONC-preprocess → SOS-relaxation (Algorithm 5)
- Both two-step methods return `SOSONCRelaxSol` with decomposed certificates.

### Phase 4: Joint SOS+SONC (optional, requires exponential cone)
- Combine Gram matrix (SDP) + relative entropy (exponential cone) constraints
- Requires solver support for both cone types (MOSEK). If unavailable, fall back to two-step.

## Dependencies
- Irene: `program.py` (OptimizationProblem), `sonc.py` (SONCRelaxations), `relaxations.py` (SDPRelaxations), `sdp.py`
- External: numpy, scipy, sympy, gpkit (for SONC), cvxopt (for SDP fallback)
- Optional: mosek (for relative entropy / exponential cone), csdp, cvxpy

## File Structure
```
Irene/
├── Irene/
│   ├── __init__.py          # Add: from .sosonc import SOSONCRelaxations, SOSONCRelaxSol
│   ├── sosonc.py            # NEW: this module
│   ├── sonc.py              # Existing
│   ├── sdp.py               # Existing
│   ├── relaxations.py       # Existing
│   ├── program.py           # Existing
│   └── ...
└── tests/
    └── test_sosonc.py       # NEW: tests
```

## Test Cases

### Test 1: Motzkin Polynomial (f = 1 + x⁴y² + x²y⁴ - 3x²y²)
- SOS: infeasible (λ* = -∞)
- SONC: λ* = 0 (Motzkin is SONC)
- SOS+SONC: λ* = 0 (exact), with g* ∈ Σ

### Test 2: Choi-Lam Form (f = x⁴y² + x²y⁴ + z⁶ - 3x²y²z²)
- SOS: infeasible
- SONC: positive bound
- SOS+SONC: should match SONC

### Test 3: Simple quadratic (f = x⁴ - x²)
- SOS: λ* = -0.25 (exact)
- SONC: approx -1.00
- SOS+SONC: should match SOS (-0.25)

### Test 4: Separating polynomial from Schick Lemma 4.2.7
- SOS: λ* = -∞ (infeasible)
- SONC: λ* ≈ -2.98
- SOS+SONC: λ* ≈ 0 (near-exact)
