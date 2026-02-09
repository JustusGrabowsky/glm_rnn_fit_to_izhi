"""
makeBasis_PostSpike.py - Faithful translation of makeBasis_PostSpike.m

Make nonlinearly stretched basis consisting of raised cosines.
"""

import numpy as np
from scipy.linalg import orth


def makeBasis_PostSpike(ihprs, dt, iht0=None):
    """
    [iht, ihbas, ihbasis] = makeBasis_PostSpike(ihprs, dt, iht0)

    Make nonlinearly stretched basis consisting of raised cosines.

    Parameters:
    -----------
    ihprs : dict
        Param structure with fields:
            ncols : number of basis vectors
            hpeaks : 2-element list containing [1st_peak, last_peak], the peak
                     location of first and last raised cosine basis vectors
            b : offset for nonlinear stretching of x axis: y = log(x+b)
                (larger b -> more nearly linear stretching)
            absref : absolute refractory period (optional)
    dt : float
        Grid of time points for representing basis
    iht0 : ndarray, optional
        Cut off time (or extend) basis so it matches this

    Returns:
    --------
    iht : ndarray
        Time lattice on which basis is defined
    ihbas : ndarray
        Orthogonalized basis
    ihbasis : ndarray
        Original (non-orthogonal) basis

    Example:
    --------
    ihbasprs = {'ncols': 5, 'hpeaks': [0.1, 2], 'b': 0.5, 'absref': 0.1}
    iht, ihbas, ihbasis = makeBasis_PostSpike(ihbasprs, dt)
    """
    ncols = ihprs['ncols']
    b = ihprs['b']
    hpeaks = ihprs['hpeaks']

    if 'absref' in ihprs:
        absref = ihprs['absref']
    else:
        absref = 0

    # Check input values
    if hpeaks[0] + b < 0:
        raise ValueError('b + first peak location: must be greater than 0')

    if absref >= dt:  # use one fewer "cosine-shaped basis vector"
        ncols = ncols - 1
    elif absref > 0:
        import warnings
        warnings.warn('Refractory period too small for time-bin sizes')

    # nonlinearity for stretching x axis (and its inverse)
    def nlin(x):
        return np.log(x + 1e-20)

    def invnl(x):
        return np.exp(x) - 1e-20

    # Generate basis of raised cosines
    yrnge = nlin(np.array(hpeaks) + b)
    db = np.diff(yrnge)[0] / (ncols - 1)  # spacing between raised cosine peaks
    ctrs = np.arange(yrnge[0], yrnge[1] + db / 2, db)  # centers for basis vectors
    mxt = invnl(yrnge[1] + 2 * db) - b  # maximum time bin
    iht = np.arange(0, mxt + dt, dt).reshape(-1, 1)
    nt = len(iht)  # number of points in iht

    # raised cosine basis vector
    def ff(x, c, dc):
        return (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x - c) * np.pi / dc / 2))) + 1) / 2

    ihbasis = ff(np.tile(nlin(iht + b), (1, ncols)), np.tile(ctrs, (nt, 1)), db)

    # set first cosine basis vector bins (before 1st peak) to 1
    ii = np.where(iht.flatten() <= hpeaks[0])[0]
    ihbasis[ii, 0] = 1

    # create first basis vector as step-function for absolute refractory period
    if absref >= dt:
        ii = np.where(iht.flatten() < absref)[0]
        ih0 = np.zeros((ihbasis.shape[0], 1))
        ih0[ii, 0] = 1
        ihbasis[ii, :] = 0
        ihbasis = np.hstack([ih0, ihbasis])

    ihbas = orth(ihbasis)  # use orthogonalized basis

    if iht0 is not None:
        if np.abs(np.diff(iht0[:2])[0] - dt) > 1e-10:
            raise ValueError('iht passed in has different time binsize')
        niht = len(iht0)
        if iht[-1, 0] > iht0[-1]:  # Truncate basis
            iht = iht0.reshape(-1, 1)
            ihbasis = ihbasis[:niht, :]
            ihbas = ihbas[:niht, :]
        elif iht[-1, 0] < iht0[-1]:  # Extend basis
            nextra = niht - len(iht)
            iht = iht0.reshape(-1, 1)
            ihbasis = np.vstack([ihbasis, np.zeros((nextra, ihbasis.shape[1]))])
            ihbas = np.vstack([ihbas, np.zeros((nextra, ihbas.shape[1]))])

    return iht.flatten(), ihbas, ihbasis
