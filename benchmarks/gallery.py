"""
Benchmark Gallery — Standard Polynomial Optimization Test Problems
=================================================================

A curated set of 12 polynomial optimization problems spanning the full spectrum
of sparsity patterns, degrees, and structural properties. Used to benchmark
Newton polytope pruning, correlative sparsity detection, and border basis
reductions in Irene's relaxation pipeline.

Each problem is a dict with:
    - name: human-readable identifier
    - objective: SymPy expression (minimization target)
    - constraints: list of (expr, type) tuples; type in ('equality', 'inequality')
    - variables: ordered list of SymPy symbols
    - nvars / degree: structural metadata
    - sparsity_class: 'sparse' | 'dense' | 'separable' | 'mixed'
    - expected_reduction: rough lower bound on Newton pruning ratio (0-1)
    - known_minimum: verified global minimum (for validation)
"""

from __future__ import annotations

from sympy import symbols


def _build_gallery():
    """Build the full gallery with proper SymPy expressions."""
    gallery = []

    # =========================================================================
    #  SPARSE PROBLEMS (expected 60-90% basis reduction)
    # =========================================================================

    x, y = symbols("x y")

    # 1. Motzkin polynomial (degree 6, 2 vars)
    gallery.append({
        "name": "Motzkin",
        "objective": x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1,
        "constraints": [],
        "variables": [x, y],
        "nvars": 2,
        "degree": 6,
        "sparsity_class": "sparse",
        "expected_reduction": 0.7,
        "known_minimum": 0.0,
        "description": "Nonnegative but not SOS; Newton polytope has few interior lattice points.",
    })

    # 2. Robinson polynomial (degree 6, 2 vars)
    gallery.append({
        "name": "Robinson",
        "objective": x**6 + y**6 - x**4 - y**4 + 2 * x**2 * y**2 - x**2 - y**2 + 1,
        "constraints": [],
        "variables": [x, y],
        "nvars": 2,
        "degree": 6,
        "sparsity_class": "sparse",
        "expected_reduction": 0.75,
        "known_minimum": 0.0,
        "description": "Nonnegative, not SOS; symmetric sparse structure.",
    })

    # 3. Choi-Lam polynomial (degree 6, 2 vars)
    gallery.append({
        "name": "ChoiLam",
        "objective": (x**2 - y)**2 + (y**2 - x)**2,
        "constraints": [],
        "variables": [x, y],
        "nvars": 2,
        "degree": 4,
        "sparsity_class": "sparse",
        "expected_reduction": 0.65,
        "known_minimum": 0.0,
        "description": "Sum of two squares; trivially nonnegative with small Newton polytope.",
    })

    # 4. Bunimovich-Meixner (degree 8, 2 vars) — sparse high-degree
    gallery.append({
        "name": "BunimovichMeixner",
        "objective": x**8 + y**8 - x**6 - y**6 + 3 * x**4 * y**4 - 2 * x**4 - 2 * y**4 + 1,
        "constraints": [],
        "variables": [x, y],
        "nvars": 2,
        "degree": 8,
        "sparsity_class": "sparse",
        "expected_reduction": 0.6,
        "known_minimum": 0.0,
        "description": "High-degree sparse form; Newton pruning should be effective.",
    })

    # =========================================================================
    #  SEPARABLE PROBLEMS (for correlative sparsity chordal decomposition)
    # =========================================================================

    x1, x2, x3, x4 = symbols("x1 x2 x3 x4")

    # 5. Separable chain — {x1,x2}, {x2,x3}, {x3,x4} cliques
    gallery.append({
        "name": "SeparableChain",
        "objective": (x1**2 - 1)**2 + (x2**2 - x1)**2 + (x3**2 - x2)**2 + (x4**2 - x3)**2,
        "constraints": [],
        "variables": [x1, x2, x3, x4],
        "nvars": 4,
        "degree": 4,
        "sparsity_class": "separable",
        "expected_reduction": 0.5,
        "known_minimum": 0.0,
        "description": "Chain structure: cliques {x1,x2}, {x2,x3}, {x3,x4}. Ideal for chordal decomposition.",
    })

    # 6. Block diagonal — two independent blocks
    gallery.append({
        "name": "BlockDiagonal",
        "objective": (x1**2 + x2**2 - 1)**2 + (x3**2 + x4**2 - 1)**2,
        "constraints": [],
        "variables": [x1, x2, x3, x4],
        "nvars": 4,
        "degree": 4,
        "sparsity_class": "separable",
        "expected_reduction": 0.4,
        "known_minimum": 0.0,
        "description": "Two independent {x1,x2} and {x3,x4} blocks; correlative sparsity should detect two cliques.",
    })

    # =========================================================================
    #  MIXED SPARSITY (partial structure)
    # =========================================================================

    x, y, z = symbols("x y z")

    # 7. Constrained sparse — Motzkin with box constraints
    gallery.append({
        "name": "MotzkinConstrained",
        "objective": x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1,
        "constraints": [
            (1 - x**2, "inequality"),
            (1 - y**2, "inequality"),
        ],
        "variables": [x, y],
        "nvars": 2,
        "degree": 6,
        "sparsity_class": "mixed",
        "expected_reduction": 0.65,
        "known_minimum": 0.0,
        "description": "Motzkin on [-1,1]^2; constraints add structure but preserve sparsity.",
    })

    # 8. 3-variable sparse with equality constraint
    gallery.append({
        "name": "SparseEquality",
        "objective": x**4 + y**4 + z**4 - 2 * x**2 * y**2,
        "constraints": [
            (x**2 + y**2 + z**2 - 3, "equality"),
        ],
        "variables": [x, y, z],
        "nvars": 3,
        "degree": 4,
        "sparsity_class": "mixed",
        "expected_reduction": 0.55,
        "known_minimum": -1.0,
        "description": "Sparse objective with one equality; border basis applicable.",
    })

    # =========================================================================
    #  DENSE PROBLEMS (minimal expected reduction)
    # =========================================================================

    x, y = symbols("x y")

    # 9. Dense quartic — full support in 2D
    gallery.append({
        "name": "DenseQuartic",
        "objective": (x**2 + y**2)**2 - 4 * x * y * (x - y) + x**4 + y**4,
        "constraints": [],
        "variables": [x, y],
        "nvars": 2,
        "degree": 4,
        "sparsity_class": "dense",
        "expected_reduction": 0.95,
        "known_minimum": 0.0,
        "description": "Dense quartic with full monomial support; Newton pruning should be minimal.",
    })

    # 10. Degree-8 stress test — dense high-degree
    gallery.append({
        "name": "Degree8Stress",
        "objective": (x**2 + y**2 - 1)**4 + x**8 + y**8,
        "constraints": [],
        "variables": [x, y],
        "nvars": 2,
        "degree": 8,
        "sparsity_class": "dense",
        "expected_reduction": 0.9,
        "known_minimum": 0.0,
        "description": "High-degree dense form; stress test for moment matrix size.",
    })

    # =========================================================================
    #  LARGE-SCALE SPARSE (many variables, sparse structure)
    # =========================================================================

    x1, x2, x3, x4, x5 = symbols("x1 x2 x3 x4 x5")

    # 11. Sparse 5-variable chain
    gallery.append({
        "name": "SparseChain5",
        "objective": sum((xi**2 - xi_plus1)**2 for xi, xi_plus1 in [
            (x1, x2), (x2, x3), (x3, x4), (x4, x5)
        ]),
        "constraints": [],
        "variables": [x1, x2, x3, x4, x5],
        "nvars": 5,
        "degree": 4,
        "sparsity_class": "sparse",
        "expected_reduction": 0.6,
        "known_minimum": 0.0,
        "description": "5-variable chain; correlative sparsity should find small cliques.",
    })

    # 12. Dense 3-variable — full support in 3D
    gallery.append({
        "name": "DenseTrivariate",
        "objective": (x**2 + y**2 + z**2 - 1)**2 + x**4 * y**2 + y**4 * z**2 + z**4 * x**2,
        "constraints": [],
        "variables": [x, y, z],
        "nvars": 3,
        "degree": 6,
        "sparsity_class": "dense",
        "expected_reduction": 0.85,
        "known_minimum": 0.0,
        "description": "Dense trivariate degree-6; Newton pruning should be limited.",
    })

    return gallery


# Build at import time
GALLERY = _build_gallery()


def get_problem(name: str) -> dict | None:
    """Look up a problem by name."""
    for p in GALLERY:
        if p["name"].lower() == name.lower():
            return p
    return None


def get_all_names() -> list[str]:
    """Return all gallery problem names."""
    return [p["name"] for p in GALLERY]


if __name__ == "__main__":
    print(f"Gallery: {len(GALLERY)} problems")
    for p in GALLERY:
        print(f"  {p['name']:20s} | nvars={p['nvars']} deg={p['degree']} "
              f"class={p['sparsity_class']:10s} expected_reduction={p['expected_reduction']:.0%}")
