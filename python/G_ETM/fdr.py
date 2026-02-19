"""
This function computes false discovery rate corrected for multiple comparisons.

File from Kim et al.'s paper available at:
http://www.neurostat.mit.edu/gcpp

Converted from MATLAB to Python.
"""

import numpy as np
from scipy.stats import chi2


def fdr(D, p, ht):
    """Compute false discovery rate corrected for multiple comparisons.

    Parameters
    ----------
    D : np.ndarray
        Matrix of deviance ratio statistics, shape (CHN, CHN).
    p : float
        FDR threshold (e.g., 0.05).
    ht : array-like
        Degrees of freedom for chi-squared distribution, one per row.

    Returns
    -------
    GCMAP : np.ndarray
        Binary connectivity matrix after FDR correction, shape (CHN, CHN).
    """
    CHN = D.shape[0]

    # Number of multiple hypothesis tests
    m = CHN * CHN

    # Compute p-values using chi-squared distribution
    P = np.zeros_like(D, dtype=float)
    for n in range(CHN):
        P[n, :] = 1.0 - chi2.cdf(D[n, :], ht[n])

    # Sort p-values (column-major order to match MATLAB behavior)
    Ps_flat = P.flatten(order='F')
    sort_order = np.argsort(Ps_flat, kind='stable')
    Ps = Ps_flat[sort_order]

    # Find largest k such that Ps[k] <= (k+1)/m * p  (Benjamini-Hochberg)
    # The loop variable k holds the index of the first p-value exceeding the
    # threshold (or m-1 if none exceeds it).  No decrement is applied here so
    # that range(k) correctly iterates over the k significant tests, matching
    # MATLAB's post-decrement "for ii = 1:k" loop behaviour.
    k = 0
    for k in range(m):
        if Ps[k] > (k + 1) / m * p:
            break
    else:
        # Loop completed without breaking: all k indices 0..m-2 are significant.
        # range(k) below will cover 0..m-2, matching MATLAB's m-1 iterations.
        pass

    GCMAP = np.zeros_like(D)

    for ii in range(k):
        # Convert flat (column-major) index to 2D index
        row, col = np.unravel_index(sort_order[ii], (CHN, CHN), order='F')
        GCMAP[row, col] = 1

    return GCMAP
