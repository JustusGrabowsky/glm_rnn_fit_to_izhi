"""
logexp1.py - Faithful translation of logexp1.m

Computes the function:
    f(x) = log(1+exp(x))
and returns first and second derivatives.
"""

import numpy as np


def logexp1(x, nargout=1):
    """
    [f, df, ddf] = logexp1(x)

    Computes the function:
        f(x) = log(1+exp(x))
    and returns first and second derivatives.

    Parameters:
    -----------
    x : array_like
        Input values
    nargout : int
        Number of outputs to return (1, 2, or 3)

    Returns:
    --------
    f : ndarray
        Function values f(x) = log(1 + exp(x))
    df : ndarray (optional)
        First derivative f'(x) = exp(x) / (1 + exp(x))
    ddf : ndarray (optional)
        Second derivative f''(x) = exp(x) / (1 + exp(x))^2
    """
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    f = np.log(1 + np.exp(x))

    if nargout > 1:
        df = np.exp(x) / (1 + np.exp(x))

    if nargout > 2:
        ddf = np.exp(x) / (1 + np.exp(x)) ** 2

    # Check for small values
    iix = x < -20
    if np.any(iix):
        f[iix] = np.exp(x[iix])
        if nargout > 1:
            df[iix] = f[iix]
        if nargout > 2:
            ddf[iix] = f[iix]

    # Check for large values
    iix = x > 500
    if np.any(iix):
        f[iix] = x[iix]
        if nargout > 1:
            df[iix] = 1
        if nargout > 2:
            ddf[iix] = 0

    if scalar_input:
        f = f[0]
        if nargout > 1:
            df = df[0]
        if nargout > 2:
            ddf = ddf[0]

    if nargout == 1:
        return f
    elif nargout == 2:
        return f, df
    else:
        return f, df, ddf
