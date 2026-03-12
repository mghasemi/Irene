
# Documentation: `geometric.py`, `grouprings.py`, `program.py`, and `sonc.py`

## Reviewer Tracking

For review workflow and sign-off tracking, use:

- [reviewer-plan-tracker.md](reviewer-plan-tracker.md)
- [reviewer-plan-tracker-sonc.md](reviewer-plan-tracker-sonc.md)

## Quality Gates and Build Policy

The repository should treat `Irene/` as the source of truth for Python modules. Files under `build/lib/Irene/` are generated artifacts and should not be edited manually.

### Recommended quality checks

Run these commands from the repository root:

```bash
/home/mehdi/Code/Irene/.venv/bin/python -m unittest tests/test_quality_plan.py -v
/home/mehdi/Code/Irene/.venv/bin/python examples/GPExample.py
```

`examples/GPExample.py` exercises `GPRelaxations.solve()` end-to-end through `Irene/geometric.py`.
Depending on the matrix structure used in `auto_transform_matrix`, NumPy may emit a runtime warning during intermediate ratio evaluation; this does not necessarily prevent the model from solving when the final transformation is well-defined.

Optional regression checks for package build consistency:

```bash
/home/mehdi/Code/Irene/.venv/bin/python setup.py build
```

### Source/build synchronization workflow

1. Make all code changes in `Irene/*.py`.
2. Run the quality checks listed above.
3. Regenerate `build/lib/Irene/` via `setup.py build` when a distributable build is needed.
4. Review generated diffs separately from source edits.

This document outlines the relationship between the files `geometric.py`, `grouprings.py`, and `program.py`, and explains how they can be used to complete the implementation of `sonc.py`. The goal of `sonc.py` is to implement the SONC (Sum of Non-negative Circuit polynomials) relaxation for polynomial optimization problems, as described by the formulation in `prog32.png`.

## File Descriptions and Relationships

### `grouprings.py`

This file provides the foundational algebraic structures for the entire project. It defines classes for:

*   **`CommutativeSemigroup`**: Represents a commutative semigroup, which is a set with an associative and commutative binary operation.
*   **`SemigroupAlgebra`**: Represents a semigroup algebra, which is a vector space over a field with a basis consisting of the elements of a semigroup.
*   **`AtomicSGElement` and `SemigroupAlgebraElement`**: Represent elements within the semigroup algebra, effectively allowing for the creation and manipulation of polynomials and monomials.

In essence, `grouprings.py` provides the tools to represent the mathematical objects (polynomials) that are central to the optimization problems being solved.

### `program.py`

This file builds upon the structures in `grouprings.py` to define a formal optimization problem. The key class is:

*   **`OptimizationProblem`**: This class encapsulates a polynomial optimization problem. It takes a `SemigroupAlgebra` object and allows for the definition of an objective function and a set of constraints.

This file acts as a bridge between the abstract algebraic structures in `grouprings.py` and the concrete optimization problems that are solved in other parts of the codebase. It provides a structured way to define a problem that can then be passed to a solver.

### `geometric.py`

This file implements a specific type of solver for polynomial optimization problems.

*   **`GPRelaxations`**: This class takes an `OptimizationProblem` object and constructs a Geometric Program (GP) relaxation of it. The `solve` method of this class uses the `gpkit` library to solve the GP.

This file demonstrates how to take a problem defined in `program.py` and use an external library (`gpkit`) to find a solution.

### `sonc.py` (Incomplete)

This file is intended to implement the SONC (Sum of Non-negative Circuit polynomials) relaxation. The image `prog32.png` provides the mathematical formulation for this relaxation, which is a geometric program.

## Completing `sonc.py`

To complete `sonc.py`, you need to implement the optimization problem (3.2) from `prog32.png`. This will involve the following steps:

1.  **Define the `SONCRelaxations` class**: This class will be similar in structure to the `GPRelaxations` class in `geometric.py`. It should take an `OptimizationProblem` object in its constructor.

2.  **Define the GP Variables**: The optimization problem in `prog32.png` has several variables: `μ`, `a_β,j`, and `b_β`. These can be defined using `gpkit`'s `VectorVariable` and `Variable` classes.

3.  **Construct the Objective Function**: The objective function `p(μ, {(a_β, b_β): β ∈ Δ(G)})` needs to be constructed as a `gpkit` expression. This will involve summing up the terms as defined in the image.

4.  **Construct the Constraints**: The four constraints of the optimization problem need to be translated into `gpkit` constraints.
    *   Constraint (1): `Σ a_β,j ≤ G(μ)_α(j)`
    *   Constraint (2): `Π (a_β,j / λ_j^(β))^(λ_j^(β)) ≥ b_β`
    *   Constraint (3): `G(μ)_β^+ ≤ b_β`
    *   Constraint (4): `G(μ)_β^- ≤ b_β`

5.  **Solve the GP**: Once the objective function and constraints are defined, you can create a `gpkit` `Model` and call the `solve()` method to find the solution. The `geometric.py` file provides a good example of how to do this.

By following the structure of `geometric.py` and using the classes and functions from `grouprings.py` and `program.py`, you can complete the implementation of `sonc.py` to solve the SONC relaxation.

## SONC Examples

Two runnable examples are available under `examples/`:

- `SONCExample.py`
    - A 1D constrained benchmark that reports either a computed SONC bound or an explicit runtime status.
- `SONCExample33.py`
    - Uses Section 3, Example 3.3 from the paper:
        - `f = 1 + 2*x^2*y^4 + (1/2)*x^3*y^2`
        - `g1 = 1/3 - x^6*y^2`
    - Prints extracted barycentric coordinates for `beta = (3, 2)` and then solves with `SONCRelaxations`.

Run them with:

```bash
/home/mehdi/Code/Irene/.venv/bin/python examples/SONCExample.py
/home/mehdi/Code/Irene/.venv/bin/python examples/SONCExample33.py
```
