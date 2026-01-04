import numpy as np
import sympy as sp
import cvxpy as cp
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key

def get_gram_matrix(polynomial):
    """
    Computes a symmetric Gram matrix Q for a given SymPy polynomial p
    such that p = Z.T * Q * Z, where Z is the vector of monomials.

    Args:
        polynomial (sympy.Expr): A sympy polynomial expression.

    Returns:
        Q (sympy.Matrix): The Gram matrix.
        Z (sympy.Matrix): The monomial basis vector.
    """
    # 1. Extract variables and ensure it is a polynomial
    poly = sp.Poly(polynomial)
    vars = poly.gens
    degree = poly.total_degree()

    # 2. Gram matrices typically require an even degree (2d)
    if degree % 2 != 0:
        raise ValueError(f"Polynomial must have an even total degree. Current degree: {degree}")
    
    half_degree = degree // 2

    # 3. Generate the basis Z (monomials up to degree d)
    # We sort them to ensure the matrix is deterministic and organized
    monoms = sorted(list(itermonomials(vars, half_degree)), 
                    key=monomial_key('grlex', vars))
    
    Z = sp.Matrix(monoms)
    n = len(Z)
    Q = sp.zeros(n, n)

    # 4. Map monomials in p to matrix indices (i, j) that produce them
    # We create a map: product_monomial -> list of (i, j) pairs
    product_map = {}
    
    for i in range(n):
        for j in range(n):
            prod = Z[i] * Z[j]
            if prod not in product_map:
                product_map[prod] = []
            product_map[prod].append((i, j))

    # 5. Fill the Matrix Q
    # We iterate through the terms of the input polynomial
    # and distribute the coefficient equally among all (i, j) pairs that form that term.
    terms = poly.as_expr().as_coefficients_dict()
    
    for monom, coeff in terms.items():
        # Handle constant term explicitly if it's '1' (sympy treats it differently sometimes)
        if monom == 1:
            monom = sp.Integer(1)
            
        if monom in product_map:
            pairs = product_map[monom]
            num_pairs = len(pairs)
            
            # Distribute coefficient equally
            value = coeff / num_pairs
            
            for (i, j) in pairs:
                Q[i, j] += value

    return np.array(Q.evalf(), dtype=np.float64), Q

def is_psd_numeric(matrix, tol=1e-8):
    """
    Checks if a matrix is Positive Semidefinite using NumPy.
    
    Args:
        matrix (np.ndarray): The input matrix.
        tol (float): Tolerance for floating point errors.
        
    Returns:
        bool: True if PSD, False otherwise.
    """
    # 1. Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # 2. Check if matrix is symmetric (A == A.T)
    # We use allclose to handle floating point noise
    if not np.allclose(matrix, matrix.T, atol=tol):
        return False

    # 3. Calculate eigenvalues
    # 'eigvalsh' is preferred over 'eigvals' for symmetric matrices
    # as it is faster and more numerically stable.
    eigenvalues = np.linalg.eigvalsh(matrix)

    # 4. Check if all eigenvalues are >= -tolerance
    # A theoretically 0 eigenvalue might appear as -1e-15 due to noise.
    return np.all(eigenvalues >= -tol)

def is_psd_symbolic(matrix):
    """
    Checks if a SymPy matrix is Positive Semidefinite.
    """
    # SymPy has a built-in method for this
    return matrix.is_positive_semidefinite

def find_psd_gram_matrix(polynomial):
    """
    Uses Convex Optimization (SDP) to find a Positive Semidefinite (PSD) 
    Gram matrix for the given polynomial.
    """
    # 1. Setup SymPy polynomial and Basis
    poly = sp.Poly(polynomial)
    vars = poly.gens
    degree = poly.total_degree()
    
    if degree % 2 != 0:
        print("Polynomial has odd degree. Cannot be SOS.")
        return None

    half_degree = degree // 2
    # Create basis vector Z
    basis = sorted(list(itermonomials(vars, half_degree)), 
                   key=monomial_key('grlex', vars))
    n = len(basis)
    
    print(f"Polynomial: {polynomial}")
    print(f"Basis Z: {basis}")

    # 2. Create the Optimization Variable (The Matrix Q)
    # We explicitly tell cvxpy this matrix must be PSD
    Q = cp.Variable((n, n), PSD=True)

    # 3. Create Constraints
    # We must ensure Z.T * Q * Z == polynomial
    # This means matching the coefficients of every monomial.
    constraints = []
    
    # We need a map to store which Q_ij contribute to which monomial
    coeff_map = {}
    
    # Loop over the matrix Q indices (i, j)
    for i in range(n):
        for j in range(n):
            # Calculate the monomial resulting from basis[i] * basis[j]
            prod = basis[i] * basis[j]
            
            if prod not in coeff_map:
                coeff_map[prod] = []
            
            # Store the cvxpy variable reference Q[i, j]
            coeff_map[prod].append(Q[i, j])

    # Now, equate the sums of Q variables to the actual polynomial coefficients
    poly_coeffs = poly.as_expr().as_coefficients_dict()
    
    # We must check ALL possible monomials produced by the basis,
    # not just the ones currently in the polynomial (some might need to sum to 0).
    for monom, q_vars in coeff_map.items():
        # Get actual coefficient from polynomial (0 if term is missing)
        target_coeff = float(poly_coeffs.get(monom, 0))
        
        # Constraint: Sum of Q entries for this monomial == Target Coefficient
        constraints.append(cp.sum(q_vars) == target_coeff)

    # 4. Solve the Problem
    # We just want *feasibility* (any valid PSD matrix), so we minimize constant 0.
    prob = cp.Problem(cp.Minimize(0), constraints)
    
    try:
        prob.solve()
    except cp.SolverError:
        print("Solver failed.")
        return None

    # 5. Check results
    if prob.status in ["infeasible", "unbounded"]:
        print("No PSD Gram matrix exists. The polynomial is NOT a Sum of Squares.")
        return None
    else:
        print("Found a PSD Gram matrix!")
        # Convert result to a clean numpy array (rounding to remove solver noise)
        Q_value = np.round(Q.value, 5)
        return Q_value
    
def numpy_to_latex(matrix, precision=3, env='bmatrix'):
    """
    Converts a numpy matrix or array into LaTeX code.
    
    Args:
        matrix (np.ndarray): The input matrix.
        precision (int): Decimal places for floating point numbers.
        env (str): The LaTeX environment (bmatrix, pmatrix, vmatrix, etc.)
        
    Returns:
        str: The generated LaTeX string.
    """
    if len(matrix.shape) == 1:
        matrix = matrix.reshape(1, -1)
        
    lines = []
    
    # Define the format string for floats based on precision
    float_fmt = f"{{:.{precision}f}}"
    
    for row in matrix:
        line = []
        for val in row:
            if isinstance(val, (int, np.integer)):
                line.append(str(val))
            else:
                # Format floats, removing trailing zeros if cleaner look is desired
                s = float_fmt.format(val)
                if '.' in s:
                    s = s.rstrip('0').rstrip('.')
                line.append(s)
                
        # Join the row values with ' & ' and end with '\\'
        lines.append(" & ".join(line) + " \\\\")
        
    # Join all lines and wrap in the environment
    body = "\n".join(lines)
    latex_code = f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"
    
    return latex_code