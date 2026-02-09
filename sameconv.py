"""
sameconv.py - Faithful translation of sameconv.m

Causally filters A with B, giving a column vector with same height as A.
(B not flipped as in standard convolution).

Convolution performed efficiently in (zero-padded) Fourier domain.
"""

import numpy as np
from numpy.fft import fft, ifft


def sameconv(A, B):
    """
    G = sameconv(A, B)

    Causally filters A with B, giving a column vector with same height as A.
    (B not flipped as in standard convolution).

    Convolution performed efficiently in (zero-padded) Fourier domain.

    Parameters:
    -----------
    A : ndarray
        Input signal (1D or 2D column vector)
    B : ndarray
        Filter kernel (1D or 2D column vector)

    Returns:
    --------
    G : ndarray
        Filtered signal, same length as A
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    # Ensure column vectors
    if A.shape[0] == 1:
        A = A.T
    if B.shape[0] == 1:
        B = B.T

    am, an = A.shape
    bm, bn = B.shape
    nn = am + bm - 1

    # G = ifft(sum(fft(A,nn).*fft(flipud(B),nn),2))
    G = ifft(np.sum(fft(A, nn, axis=0) * fft(np.flipud(B), nn, axis=0), axis=1))
    G = np.real(G[:am])

    return G.flatten()
