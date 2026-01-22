"""
Root Finding Utilities for Lens Inversion

Safe, robust root finding without scipy.optimize dependencies.
Uses only bisection and bracketing - deterministic and exact.

Authors: Carmen N. Wrede, Lino P. Casu
License: ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import numpy as np
from typing import List, Callable, Optional, Tuple


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 100
) -> Optional[float]:
    """
    Simple bisection root-finding.
    
    Finds x such that f(x) = 0 in [a, b], assuming f(a) and f(b) 
    have opposite signs.
    
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
    root : float or None
        Root if found, None otherwise
    """
    try:
        fa, fb = f(a), f(b)
    except Exception:
        return None
    
    if np.isnan(fa) or np.isnan(fb):
        return None
    if np.isinf(fa) or np.isinf(fb):
        return None
    
    if fa * fb > 0:
        return None  # No sign change
    
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        
        try:
            fm = f(mid)
        except Exception:
            return None
        
        if np.isnan(fm):
            return None
        
        if abs(fm) < tol or (b - a) < tol:
            return mid
        
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    
    return 0.5 * (a + b)


def find_all_roots(
    f: Callable[[float], float],
    x_min: float,
    x_max: float,
    n_samples: int = 500,
    tol: float = 1e-12
) -> np.ndarray:
    """
    Find all roots of f in [x_min, x_max] by scanning for sign changes.
    
    Parameters
    ----------
    f : callable
        Function to find roots of
    x_min, x_max : float
        Search interval
    n_samples : int
        Number of sample points for bracket detection
    tol : float
        Root tolerance
    
    Returns
    -------
    roots : ndarray
        Array of root locations
    """
    x_test = np.linspace(x_min, x_max, n_samples)
    f_vals = []
    
    for x in x_test:
        try:
            val = f(x)
            if np.isnan(val) or np.isinf(val):
                f_vals.append(None)
            else:
                f_vals.append(val)
        except Exception:
            f_vals.append(None)
    
    roots = []
    for i in range(len(x_test) - 1):
        if f_vals[i] is None or f_vals[i+1] is None:
            continue
        if f_vals[i] * f_vals[i+1] < 0:
            root = bisection(f, x_test[i], x_test[i+1], tol=tol)
            if root is not None:
                roots.append(root)
    
    return np.array(roots)


def find_all_roots_safe(
    f: Callable[[float], float],
    x_min: float,
    x_max: float,
    n_samples: int = 200,
    tol: float = 1e-10
) -> List[float]:
    """
    Safe version that returns list and handles exceptions gracefully.
    """
    roots = find_all_roots(f, x_min, x_max, n_samples, tol)
    return list(roots)


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 50
) -> Tuple[Optional[float], bool]:
    """
    Newton-Raphson root finding with derivative.
    
    Parameters
    ----------
    f : callable
        Function
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
    root : float or None
        Root if converged
    converged : bool
        Whether converged
    """
    x = x0
    
    for _ in range(max_iter):
        try:
            fx = f(x)
            dfx = df(x)
        except Exception:
            return None, False
        
        if abs(dfx) < 1e-15:
            return None, False
        
        dx = fx / dfx
        x_new = x - dx
        
        if abs(dx) < tol:
            return x_new, True
        
        x = x_new
    
    return x, False


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 100
) -> Optional[float]:
    """
    Brent's method - combines bisection, secant, and inverse quadratic.
    
    More efficient than pure bisection for smooth functions.
    """
    try:
        fa, fb = f(a), f(b)
    except Exception:
        return None
    
    if np.isnan(fa) or np.isnan(fb):
        return None
    
    if fa * fb > 0:
        return None
    
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c = a
    fc = fa
    mflag = True
    
    for _ in range(max_iter):
        if abs(fb) < tol:
            return b
        
        if abs(b - a) < tol:
            return b
        
        # Try inverse quadratic interpolation
        if fa != fc and fb != fc:
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)
        
        # Check conditions for bisection
        cond1 = not ((3*a + b) / 4 < s < b or b < s < (3*a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2 if 'd' in dir() else False
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol if 'd' in dir() else False
        
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False
        
        try:
            fs = f(s)
        except Exception:
            s = (a + b) / 2
            fs = f(s)
        
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
    
    return b


def find_minimum_bracketed(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, float]:
    """
    Find minimum of f in [a, b] using golden section search.
    
    Returns
    -------
    x_min : float
        Location of minimum
    f_min : float
        Value at minimum
    """
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    
    f1 = f(x1)
    f2 = f(x2)
    
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)
    
    x_min = (a + b) / 2
    return x_min, f(x_min)
