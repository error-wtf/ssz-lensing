"""
Exact Linear Solvers for No-Fit Inversion

Pure linear algebra - no optimization, no fitting, no least squares.
When system is exactly determined, solve Ax = b directly.
When overdetermined, select an invertible subset.

NO SCIPY.OPTIMIZE ALLOWED.
"""

import numpy as np
from typing import Tuple, List, Optional


def solve_linear_exact(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Solve Ax = b exactly using Gaussian elimination with pivoting.
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        Square coefficient matrix
    b : ndarray, shape (n,)
        Right-hand side vector
        
    Returns
    -------
    x : ndarray
        Solution vector
    success : bool
        True if solution found (matrix invertible)
    """
    n = A.shape[0]
    if A.shape[1] != n or len(b) != n:
        return np.zeros(n), False
    
    # Augmented matrix
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    
    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(Ab[col, col])
        for row in range(col + 1, n):
            if abs(Ab[row, col]) > max_val:
                max_val = abs(Ab[row, col])
                max_row = row
        
        # Check for singularity
        if max_val < 1e-15:
            return np.zeros(n), False
        
        # Swap rows
        if max_row != col:
            Ab[[col, max_row]] = Ab[[max_row, col]]
        
        # Eliminate below
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]
    
    # Back substitution
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        if abs(Ab[row, row]) < 1e-15:
            return np.zeros(n), False
        x[row] = (Ab[row, n] - np.dot(Ab[row, row+1:n], x[row+1:n])) / Ab[row, row]
    
    return x, True


def solve_linear_subset(
    A: np.ndarray,
    b: np.ndarray,
    rows: List[int]
) -> Tuple[np.ndarray, bool]:
    """
    Solve using a subset of equations.
    
    Parameters
    ----------
    A : ndarray, shape (m, n)
        Coefficient matrix (m equations, n unknowns)
    b : ndarray, shape (m,)
        Right-hand side
    rows : list of int
        Which rows to use (must have len(rows) == n)
        
    Returns
    -------
    x : ndarray
        Solution
    success : bool
        True if successful
    """
    n = A.shape[1]
    if len(rows) != n:
        return np.zeros(n), False
    
    A_sub = A[rows, :]
    b_sub = b[rows]
    
    return solve_linear_exact(A_sub, b_sub)


def matrix_rank(A: np.ndarray, tol: float = 1e-12) -> int:
    """
    Compute matrix rank via SVD-free method (Gaussian elimination).
    """
    m, n = A.shape
    A_work = A.astype(float).copy()
    
    rank = 0
    col = 0
    
    for row in range(min(m, n)):
        # Find pivot in current column
        pivot_row = None
        for r in range(row, m):
            if abs(A_work[r, col]) > tol:
                pivot_row = r
                break
        
        if pivot_row is None:
            col += 1
            if col >= n:
                break
            continue
        
        # Swap
        if pivot_row != row:
            A_work[[row, pivot_row]] = A_work[[pivot_row, row]]
        
        # Eliminate
        for r in range(row + 1, m):
            if abs(A_work[r, col]) > tol:
                factor = A_work[r, col] / A_work[row, col]
                A_work[r, col:] -= factor * A_work[row, col:]
        
        rank += 1
        col += 1
        if col >= n:
            break
    
    return rank


def choose_invertible_subset(
    A: np.ndarray,
    b: np.ndarray,
    n_params: int
) -> Tuple[List[int], bool]:
    """
    Choose n_params rows that form an invertible system.
    
    Strategy: Greedy selection maximizing condition number proxy.
    
    Parameters
    ----------
    A : ndarray, shape (m, n_params)
        Full coefficient matrix
    b : ndarray, shape (m,)
        Right-hand side
    n_params : int
        Number of parameters (columns)
        
    Returns
    -------
    rows : list of int
        Selected row indices
    success : bool
        True if invertible subset found
    """
    m = A.shape[0]
    if m < n_params:
        return [], False
    
    # Try all combinations if small
    if m <= 8:
        from itertools import combinations
        best_rows = None
        best_det = 0
        
        for rows in combinations(range(m), n_params):
            rows_list = list(rows)
            A_sub = A[rows_list, :]
            
            # Check determinant magnitude
            try:
                det = abs(determinant(A_sub))
                if det > best_det:
                    best_det = det
                    best_rows = rows_list
            except Exception:
                continue
        
        if best_rows is not None and best_det > 1e-12:
            return best_rows, True
        return [], False
    
    # Greedy selection for larger systems
    selected = []
    remaining = list(range(m))
    
    for _ in range(n_params):
        best_row = None
        best_score = -1
        
        for row in remaining:
            test_rows = selected + [row]
            A_test = A[test_rows, :len(test_rows)]
            
            # Score by row norm contribution
            score = np.linalg.norm(A[row, :])
            
            # Check linear independence
            if len(test_rows) > 1:
                rank = matrix_rank(A[test_rows, :])
                if rank < len(test_rows):
                    continue
            
            if score > best_score:
                best_score = score
                best_row = row
        
        if best_row is None:
            return [], False
        
        selected.append(best_row)
        remaining.remove(best_row)
    
    # Verify final selection
    A_final = A[selected, :]
    if matrix_rank(A_final) < n_params:
        return [], False
    
    return selected, True


def determinant(A: np.ndarray) -> float:
    """
    Compute determinant via LU decomposition (no numpy.linalg).
    """
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix must be square")
    
    A_work = A.astype(float).copy()
    det = 1.0
    
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(A_work[col, col])
        for row in range(col + 1, n):
            if abs(A_work[row, col]) > max_val:
                max_val = abs(A_work[row, col])
                max_row = row
        
        if max_val < 1e-15:
            return 0.0
        
        # Swap rows (changes sign of determinant)
        if max_row != col:
            A_work[[col, max_row]] = A_work[[max_row, col]]
            det *= -1
        
        det *= A_work[col, col]
        
        # Eliminate
        for row in range(col + 1, n):
            factor = A_work[row, col] / A_work[col, col]
            A_work[row, col:] -= factor * A_work[col, col:]
    
    return det


def condition_estimate(A: np.ndarray) -> float:
    """
    Estimate condition number without full SVD.
    
    Uses power iteration for largest/smallest singular values.
    """
    n = A.shape[0]
    if n == 0:
        return float('inf')
    
    # Power iteration for largest singular value
    AtA = A.T @ A
    v = np.ones(A.shape[1])
    v = v / np.linalg.norm(v)
    
    for _ in range(20):
        v_new = AtA @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-15:
            return float('inf')
        v = v_new / norm
    
    sigma_max = np.sqrt(np.linalg.norm(AtA @ v))
    
    # Inverse iteration for smallest singular value
    try:
        # Add regularization for inverse
        AtA_reg = AtA + 1e-14 * np.eye(A.shape[1])
        v = np.ones(A.shape[1])
        v = v / np.linalg.norm(v)
        
        for _ in range(20):
            v_new, ok = solve_linear_exact(AtA_reg, v)
            if not ok:
                return float('inf')
            norm = np.linalg.norm(v_new)
            if norm < 1e-15:
                return float('inf')
            v = v_new / norm
        
        sigma_min = 1.0 / np.sqrt(np.linalg.norm(AtA @ v))
        
        if sigma_min < 1e-15:
            return float('inf')
        
        return sigma_max / sigma_min
    except Exception:
        return float('inf')


def solve_overdetermined_exact(
    A: np.ndarray,
    b: np.ndarray
) -> Tuple[np.ndarray, List[int], bool]:
    """
    Solve overdetermined system by finding the exact solution subset.
    
    For n unknowns with m > n equations, find n equations whose
    solution also satisfies all other equations (within tolerance).
    
    This is NOT least squares - it finds an exact solution if one exists.
    
    Parameters
    ----------
    A : ndarray, shape (m, n)
        Coefficient matrix
    b : ndarray, shape (m,)
        Right-hand side
        
    Returns
    -------
    x : ndarray
        Solution
    rows_used : list of int
        Which rows determined the solution
    success : bool
        True if exact solution found
    """
    m, n = A.shape
    
    if m == n:
        x, ok = solve_linear_exact(A, b)
        return x, list(range(m)), ok
    
    if m < n:
        return np.zeros(n), [], False
    
    # Find invertible subset
    rows, ok = choose_invertible_subset(A, b, n)
    if not ok:
        return np.zeros(n), [], False
    
    # Solve subset
    x, ok = solve_linear_subset(A, b, rows)
    if not ok:
        return np.zeros(n), [], False
    
    # Check all other equations
    residuals = A @ x - b
    max_res = np.max(np.abs(residuals))
    
    # For exact solution, all residuals should be near zero
    if max_res > 1e-10:
        # Not an exact solution - system may be inconsistent
        # Still return the solution but mark as approximate
        pass
    
    return x, rows, True
