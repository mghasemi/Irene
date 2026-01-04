
# Documentation: `geometric.py`, `grouprings.py`, `program.py`, and `sonc.py`

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
