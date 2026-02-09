"""
normalizecols.py - Faithful translation of normalizecols.m

Normalizes the columns of a matrix, so each is a unit vector.
"""

import numpy as np


def normalizecols(A):
    """
    B = normalizecols(A)

    Normalizes the columns of a matrix, so each is a unit vector.

    Parameters:
    -----------
    A : ndarray
        Input matrix

    Returns:
    --------
    B : ndarray
        Matrix with normalized columns
    """
    # B = A./repmat(sqrt(sum(A.^2)), size(A,1), 1)
    norms = np.sqrt(np.sum(A ** 2, axis=0))
    B = A / norms
    return B
