import numpy as np
import sympy as sp
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

    return Q, Z

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

# --- Example ---

# Define a matrix symbolically
Q = sp.Matrix([[2, 1], 
               [1, 2]])

is_psd = is_psd_symbolic(Q)

print("Matrix Q:")
sp.pprint(Q)
print(f"Is PSD? {is_psd}")

# You can even use it on matrices with variables if conditions are met
x = sp.symbols('x')
M_var = sp.Matrix([[x**2, 0], 
                   [0, x**2]])
print(f"\nIs M_var (with variables) PSD? {is_psd_symbolic(M_var)}")

# --- Examples ---

# Case A: Clearly PSD
A = np.array([[2, 0], 
              [0, 2]])
print(f"Matrix A is PSD: {is_psd_numeric(A)}")

# Case B: Not PSD (Negative eigenvalue)
B = np.array([[2, 0], 
              [0, -2]])
print(f"Matrix B is PSD: {is_psd_numeric(B)}")

# Case C: PSD (with floating point noise)
# Mathematically this is [1 1; 1 1], eigenvalues are 0 and 2.
C = np.array([[1.0000000001, 1], 
              [1, 1]])
print(f"Matrix C is PSD: {is_psd_numeric(C)}")

# --- Example Usage ---

# Define variables
x, y = sp.symbols('x y')

# Example 1: A simple univariate polynomial: x^4 + 2x^2 + 1
p1 = x**4 + 2*x**2 + 1
print(f"--- Processing: {p1} ---")
Q1, Z1 = get_gram_matrix(p1)
print("Basis Z:", Z1.T)
print("Gram Matrix Q:")
sp.pprint(Q1)
print("\nVerification (Z.T * Q * Z):", sp.expand((Z1.T * Q1 * Z1)[0]))

print("-" * 30)

# Example 2: A multivariate polynomial: x^2 + y^2 + 4xy + 2
p2 = x**2 + y**2 + 4*x*y + 2
print(f"--- Processing: {p2} ---")
Q2, Z2 = get_gram_matrix(p2)
print("Basis Z:", Z2.T)
print("Gram Matrix Q:")
sp.pprint(Q2)
print("\nVerification (Z.T * Q * Z):", sp.expand((Z2.T * Q2 * Z2)[0]))
# Convert the symbolic Gram matrix to a NumPy array for numeric PSD check
print(is_psd_numeric(np.array(Q2.evalf(), dtype=np.float64)))
print("-:-"*20)
print(np.array(Q2.evalf(), dtype=np.float64))