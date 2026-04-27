"""
Numerical inverse Laplace transform using the de Hoog, Knight & Stokes (1982) algorithm,
implemented entirely in NumPy float64.

This is a direct translation of the `deHoog` class in mpmath
(mpmath/calculus/inverselaplace.py) into NumPy. The algorithm structure,
parameter choices, and index arithmetic are identical to the mpmath version;
only the arithmetic library changes (mpmath → numpy complex128).

Key difference from mpmath's invertlaplace:
  - The callable F is called ONCE per time point with all (2M+1) sample points
    as a numpy complex128 array, instead of (2M+1) separate scalar calls.
  - This makes numpy-compatible RELAP callables evaluate their vectorized
    expressions (exp, sqrt, etc.) in a single C-level call.

Reference:
    de Hoog, F.R., Knight, J.H., Stokes, A.N. (1982).
    An improved method for numerical inversion of Laplace transforms.
    SIAM Journal on Scientific and Statistical Computing, 3(3), 357-366.
    https://doi.org/10.1137/0903022
"""

import numpy as np


def dehoog_invert(F_batch, t_points, M=10, dps=8):
    """
    Numerically invert the Laplace transform F(s) at each value in t_points.

    Parameters
    ----------
    F_batch : callable
        The Laplace-domain function F(s). Must accept a 1-D complex128 numpy
        array of shape (2*M+1,) and return a complex128 array of the same shape.
    t_points : array-like
        1-D array of positive time values at which to invert.
    M : int
        Number of terms in the approximation. Default 10 matches mpmath at dps=8
        (mpmath sets M = max(10, int(dps * 1.36)) = max(10, 10) = 10).
    dps : int
        Target decimal digits of precision. Used only to compute the shift
        parameter gamma (same formula as mpmath). Does not affect float64
        arithmetic precision. Default 8 matches the existing code.

    Returns
    -------
    result : ndarray, shape (len(t_points),)
        Approximated time-domain values f(t) for each t in t_points.

    Notes
    -----
    The algorithm evaluates F at the complex abscissa:
        s_k = gamma + i * k * pi / T,   k = 0, 1, ..., 2M
    where T = 2*t (per time point, matching mpmath default scale=2) and
        gamma = alpha - log(tol) / (2*T)
        alpha = 10^(-dps_goal),   tol = 10*alpha,   dps_goal = int(dps * 1.36).
    It then applies the QD (quotient-difference) rhombus rule and a Pade
    recurrence to accelerate the Fourier series, following de Hoog et al.
    """
    t_points = np.asarray(t_points, dtype=float)
    result = np.empty(len(t_points))

    dps_goal = int(dps * 1.36)
    alpha = 10.0 ** (-dps_goal)
    tol = 10.0 * alpha
    np_terms = 2 * M + 1   # total number of sample points

    for idx, t in enumerate(t_points):
        T = 2.0 * t                                     # half-period (scale=2)
        gamma = alpha - np.log(tol) / (2.0 * T)

        # --- Sample points in the Laplace domain ----------------------------
        k = np.arange(np_terms)
        s_k = gamma + 1j * k * np.pi / T               # shape (2M+1,) complex128

        # ONE batched call replaces (2M+1) scalar calls
        fp = np.asarray(F_batch(s_k), dtype=complex)   # shape (2M+1,)

        # --- QD table: rhombus rule -----------------------------------------
        # Translated directly from deHoog.calc_time_domain_solution in mpmath.
        e = np.zeros((np_terms, M + 1), dtype=complex)
        q = np.zeros((2 * M, M), dtype=complex)

        # First column of q (initialisation)
        q[0, 0] = fp[1] / (fp[0] / 2.0)
        for i in range(1, 2 * M):
            if fp[i] != 0.0:
                q[i, 0] = fp[i + 1] / fp[i]

        # Fill remaining columns via the rhombus rule
        for r in range(1, M + 1):
            mr = 2 * (M - r) + 1
            # e column r depends on q column r-1 and e column r-1
            e[0:mr, r] = (q[1:mr + 1, r - 1]
                          - q[0:mr,     r - 1]
                          + e[1:mr + 1, r - 1])
            if r != M:
                rq = r + 1
                mr2 = 2 * (M - rq) + 3      # same as mr for the next r; see notes
                for i in range(mr2):
                    denom = e[i, rq - 1]
                    if abs(denom) > 0.0:
                        q[i, rq - 1] = (q[i + 1, rq - 2]
                                        * e[i + 1, rq - 1]
                                        / denom)

        # --- Continued-fraction coefficients d ------------------------------
        d = np.zeros(np_terms, dtype=complex)
        d[0] = fp[0] / 2.0
        for r in range(1, M + 1):
            d[2 * r - 1] = -q[0, r - 1]   # even-indexed d terms
            d[2 * r]     = -e[0, r]        # odd-indexed  d terms

        # --- Pade recurrence (A / B) ----------------------------------------
        A = np.zeros(np_terms + 1, dtype=complex)
        B = np.ones(np_terms + 1, dtype=complex)
        A[0] = 0.0 + 0.0j
        A[1] = d[0]
        B[0] = 1.0 + 0.0j   # B[1] already 1 from np.ones

        z = np.exp(1j * np.pi * t / T)    # base of the power series

        for i in range(1, 2 * M):
            A[i + 1] = A[i] + d[i] * A[i - 1] * z
            B[i + 1] = B[i] + d[i] * B[i - 1] * z

        # --- Improved remainder (mpmath powm1 analogue) ---------------------
        # rem = brem * (sqrt(1 + d[2M]*z/brem) - 1)
        # which equals brem * powm1(1 + d[2M]*z/brem, 1/2) in mpmath
        brem = (1.0 + (d[2 * M - 1] - d[2 * M]) * z) / 2.0
        if abs(brem) > 0.0:
            x = d[2 * M] * z / brem
            rem = brem * (np.sqrt(1.0 + x) - 1.0)
        else:
            rem = 0.0 + 0.0j

        A[np_terms] = A[2 * M] + rem * A[2 * M - 1]
        B[np_terms] = B[2 * M] + rem * B[2 * M - 1]

        # --- Final value ----------------------------------------------------
        result[idx] = float(
            np.exp(gamma * t) / T * (A[np_terms] / B[np_terms]).real
        )

    return result
