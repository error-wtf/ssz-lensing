"""
Root Finding Solvers for No-Fit Inversion

Pure rootfinding - no optimization, no fitting.
Find exact zeros of consistency functions via bracketing + bisection.

NO SCIPY.OPTIMIZE ALLOWED.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 100
) -> Tuple[float, bool]:
    """
    Find root of f in [a, b] using bisection.
    
    Requires f(a) and f(b) to have opposite signs.
    
    Parameters
    ----------
    f : callable
        Function to find root of
    a, b : float
        Bracket endpoints
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    root : float
        Approximate root location
    converged : bool
        True if converged within tolerance
    """
    fa = f(a)
    fb = f(b)
    
    # Check for NaN
    if np.isnan(fa) or np.isnan(fb):
        return (a + b) / 2, False
    
    # Check bracket
    if fa * fb > 0:
        return (a + b) / 2, False
    
    # Ensure a < b for consistent interval tracking
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    for _ in range(max_iter):
        mid = (a + b) / 2
        fm = f(mid)
        
        if np.isnan(fm):
            return mid, False
        
        if abs(fm) < tol or (b - a) / 2 < tol:
            return mid, True
        
        # Replace endpoint with same sign as fm
        if fa * fm > 0:
            a = mid
            fa = fm
        else:
            b = mid
            fb = fm
    
    return (a + b) / 2, False


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 100
) -> Tuple[float, bool]:
    """
    Find root using Brent's method (more efficient than bisection).
    
    Combines bisection, secant, and inverse quadratic interpolation.
    """
    fa = f(a)
    fb = f(b)
    
    if np.isnan(fa) or np.isnan(fb):
        return (a + b) / 2, False
    
    if fa * fb > 0:
        return (a + b) / 2, False
    
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c = a
    fc = fa
    mflag = True
    d = 0.0
    
    for _ in range(max_iter):
        if abs(fb) < tol:
            return b, True
        
        if abs(b - a) < tol:
            return b, True
        
        # Try interpolation
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)
        
        # Conditions for accepting interpolation
        cond1 = not ((3*a + b)/4 < s < b or b < s < (3*a + b)/4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol
        
        if cond1 or cond2 or cond3 or cond4 or cond5:
            # Bisection
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False
        
        fs = f(s)
        if np.isnan(fs):
            return s, False
        
        d = c
        c = b
        fc = fb
        
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
        
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    
    return b, False


def find_all_roots(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int = 100,
    tol: float = 1e-12
) -> List[float]:
    """
    Find all roots of f in [a, b] by bracketing + refinement.

    Strategy:
    1. Sample f at n_samples points
    2. Find sign changes (brackets)
    3. Refine each bracket using bisection

    Parameters
    ----------
    f : callable
        Function to find roots of
    a, b : float
        Search interval
    n_samples : int
        Number of sample points for bracketing
    tol : float
        Root tolerance

    Returns
    -------
    roots : list of float
        All found roots (sorted, duplicates removed)
    """
    # Sample the function
    x_samples = np.linspace(a, b, n_samples)
    f_samples = []

    for x in x_samples:
        try:
            fx = f(x)
            if np.isnan(fx) or np.isinf(fx):
                f_samples.append(None)
            else:
                f_samples.append(fx)
        except Exception:
            f_samples.append(None)

    # Find brackets (sign changes)
    brackets = []
    for i in range(len(x_samples) - 1):
        f1 = f_samples[i]
        f2 = f_samples[i + 1]

        if f1 is None or f2 is None:
            continue

        # Only consider actual sign changes, not near-zero values
        if f1 * f2 < 0:
            brackets.append((x_samples[i], x_samples[i + 1]))

    # Refine each bracket
    roots = []
    for bracket in brackets:
        x1, x2 = bracket
        root, converged = bisection(f, x1, x2, tol)
        if converged:
            # Verify this is actually a root
            f_root = f(root)
            if not np.isnan(f_root) and abs(f_root) < tol * 1000:
                roots.append(root)

    # Remove duplicates
    if len(roots) > 1:
        roots = sorted(roots)
        unique_roots = [roots[0]]
        for r in roots[1:]:
            if abs(r - unique_roots[-1]) > 10 * tol:
                unique_roots.append(r)
        roots = unique_roots

    return roots


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 50
) -> Tuple[float, bool]:
    """
    Newton-Raphson iteration for root finding.
    
    Requires derivative function df.
    Falls back to bisection if Newton fails.
    
    Parameters
    ----------
    f : callable
        Function to find root of
    df : callable
        Derivative of f
    x0 : float
        Initial guess
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    root : float
        Approximate root
    converged : bool
        True if converged
    """
    x = x0
    
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if np.isnan(fx) or np.isnan(dfx):
            return x, False
        
        if abs(fx) < tol:
            return x, True
        
        if abs(dfx) < 1e-15:
            # Derivative too small, can't continue
            return x, False
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol:
            return x_new, True
        
        x = x_new
    
    return x, False


def secant_method(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-12,
    max_iter: int = 50
) -> Tuple[float, bool]:
    """
    Secant method - Newton without explicit derivative.
    
    Uses finite difference approximation of derivative.
    """
    f0 = f(x0)
    f1 = f(x1)
    
    for _ in range(max_iter):
        if np.isnan(f0) or np.isnan(f1):
            return x1, False
        
        if abs(f1) < tol:
            return x1, True
        
        if abs(f1 - f0) < 1e-15:
            return x1, False
        
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        if abs(x2 - x1) < tol:
            return x2, True
        
        x0, x1 = x1, x2
        f0 = f1
        f1 = f(x1)
    
    return x1, False


def find_root_with_initial_guess(
    f: Callable[[float], float],
    x0: float,
    search_range: float = 1.0,
    tol: float = 1e-12
) -> Tuple[float, bool]:
    """
    Find root near initial guess.
    
    First tries Newton-like methods, then falls back to bracketing.
    
    Parameters
    ----------
    f : callable
        Function to find root of
    x0 : float
        Initial guess
    search_range : float
        Range to search around x0
    tol : float
        Tolerance
        
    Returns
    -------
    root : float
        Found root
    success : bool
        True if root found
    """
    # First try secant method from x0
    x1 = x0 + 0.01 * max(abs(x0), 1.0)
    root, ok = secant_method(f, x0, x1, tol)
    
    if ok and abs(f(root)) < tol:
        return root, True
    
    # Fall back to bracketing search
    a = x0 - search_range
    b = x0 + search_range
    
    roots = find_all_roots(f, a, b, n_samples=100, tol=tol)
    
    if roots:
        # Return root closest to initial guess
        closest = min(roots, key=lambda r: abs(r - x0))
        return closest, True
    
    return x0, False


def grid_search_minimum(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_points: int = 100
) -> Tuple[float, float]:
    """
    Find approximate minimum of |f| over [a, b] via grid search.
    
    Useful for finding near-roots when exact roots don't exist.
    
    NOTE: This is NOT optimization - it's exhaustive grid evaluation.
    """
    x_grid = np.linspace(a, b, n_points)
    best_x = a
    best_val = float('inf')
    
    for x in x_grid:
        try:
            val = abs(f(x))
            if not np.isnan(val) and val < best_val:
                best_val = val
                best_x = x
        except Exception:
            continue
    
    return best_x, best_val
