"""
makeBasis_StimKernel.py - Faithful translation of makeBasis_StimKernel.m

Generates a basis consisting of raised cosines and several columns of
identity matrix vectors for temporal structure of stimulus kernel.
"""

import numpy as np
from normalizecols import normalizecols


def makeBasis_StimKernel(kbasprs, nkt=None):
    """
    [kbas, kbasis] = makeBasis_StimKernel(kbasprs, nkt)

    Generates a basis consisting of raised cosines and several columns of
    identity matrix vectors for temporal structure of stimulus kernel.

    Parameters:
    -----------
    kbasprs : dict
        Structure with fields:
            neye : number of identity basis vectors at front
            ncos : number of vectors that are raised cosines
            kpeaks : 2-element list, with peak position of 1st and last vector,
                     relative to start of cosine basis vectors (e.g. [0, 10])
            b : offset for nonlinear scaling. larger values -> more linear
                scaling of vectors. b must be >= 0
    nkt : int, optional
        Number of time samples in basis

    Returns:
    --------
    kbas : ndarray
        Orthogonal basis (same as kbasis in this implementation)
    kbasis : ndarray
        Standard (non-orth) basis
    """
    neye = kbasprs['neye']
    ncos = kbasprs['ncos']
    kpeaks = kbasprs['kpeaks']
    b = kbasprs['b']

    kdt = 1  # spacing of x axis must be in units of 1

    # nonlinearity for stretching x axis (and its inverse)
    def nlin(x):
        return np.log(x + 1e-20)

    def invnl(x):
        return np.exp(x) - 1e-20

    # Generate basis of raised cosines
    yrnge = nlin(np.array(kpeaks) + b)
    db = np.diff(yrnge)[0] / (ncos - 1)  # spacing between raised cosine peaks
    ctrs = np.arange(yrnge[0], yrnge[1] + db / 2, db)  # centers for basis vectors
    mxt = invnl(yrnge[1] + 2 * db) - b  # maximum time bin
    kt0 = np.arange(0, mxt + kdt, kdt).reshape(-1, 1)
    nt = len(kt0)  # number of points in iht

    # raised cosine basis vector
    def ff(x, c, dc):
        return (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x - c) * np.pi / dc / 2))) + 1) / 2

    kbasis0 = ff(np.tile(nlin(kt0 + b), (1, ncos)), np.tile(ctrs, (nt, 1)), db)

    # Concatenate identity-vectors
    nkt0 = kt0.shape[0]
    # kbasis = [[eye(neye); zeros(nkt0,neye)] [zeros(neye, ncos); kbasis0]]
    top_left = np.eye(neye) if neye > 0 else np.zeros((0, neye))
    bottom_left = np.zeros((nkt0, neye)) if neye > 0 else np.zeros((nkt0, 0))
    left_block = np.vstack([top_left, bottom_left]) if neye > 0 else np.zeros((nkt0, 0))

    top_right = np.zeros((neye, ncos)) if neye > 0 else np.zeros((0, ncos))
    right_block = np.vstack([top_right, kbasis0])

    if neye > 0:
        kbasis = np.hstack([left_block, right_block])
    else:
        kbasis = right_block

    kbasis = np.flipud(kbasis)  # flip so fine timescales are at the end
    nkt0 = kbasis.shape[0]

    if nkt is not None:
        if nkt0 < nkt:
            # Padding basis with zeros
            kbasis = np.vstack([np.zeros((nkt - nkt0, ncos + neye)), kbasis])
        elif nkt0 > nkt:
            # Removing rows from basis
            kbasis = kbasis[-(nkt):, :]

    kbasis = normalizecols(kbasis)
    # kbas = orth(kbasis)
    kbas = kbasis

    return kbas, kbasis
