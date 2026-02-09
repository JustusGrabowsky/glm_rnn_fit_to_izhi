"""
fit_glm.py - Faithful translation of fit_glm.m

This code fits a Poisson GLM to given data, using basis vectors to
characterize the stimulus and post-spike filters.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import warnings

from makeBasis_StimKernel import makeBasis_StimKernel
from makeBasis_PostSpike import makeBasis_PostSpike
from sameconv import sameconv
from negloglike_glm_basis import negloglike_glm_basis
from negloglike_glm_basis_softRect import negloglike_glm_basis_softRect
from logexp1 import logexp1


def fit_glm(x, y, dt, nkt=None, kbasprs=None, ihbasprs=None, prs=None,
            softRect=None, plotFlag=None, maxIter=None, tolFun=None, L2pen=None):
    """
    [k, h, dc, prs, kbasis, hbasis] = fit_glm(x, y, dt, nkt, kbasprs, ihbasprs, prs, softRect, plotFlag, maxIter, tolFun, L2pen)

    This code fits a Poisson GLM to given data, using basis vectors to
    characterize the stimulus and post-spike filters.

    Parameters:
    -----------
    x : ndarray
        Stimulus
    y : ndarray
        Spiking data, vector of 0s and 1s
    dt : float
        Time step of x and y in ms
    nkt : int, optional
        Number of ms in stimulus filter (default: 100)
    kbasprs : dict, optional
        Structure containing parameters of stimulus filter basis vectors
            neye : number of "identity" basis vectors near time of spike
            ncos : number of raised-cosine vectors to use
            kpeaks : position of first and last bump (relative to identity bumps)
            b : how nonlinear to make spacings (larger -> more linear)
    ihbasprs : dict, optional
        Structure containing parameters of post-spike filter basis vectors
            ncols : number of basis vectors for post-spike kernel
            hpeaks : peak location for first and last vectors
            b : how nonlinear to make spacings (larger -> more linear)
            absref : absolute refractory period, in ms
    prs : ndarray, optional
        Vector to initialize fit parameters
    softRect : int, optional
        0 uses exponential nonlinearity; 1 uses soft-rectifying nonlinearity (default: 0)
    plotFlag : int, optional
        0 or 1, plot simulated data (default: 0)
    maxIter : int, optional
        Maximum number of iterations for fitting (default: 100)
    tolFun : float, optional
        Function tolerance for fitting (default: 1e-8)
    L2pen : float, optional
        Size of L2 penalty on coefficients in prs (default: 0)

    Returns:
    --------
    k : ndarray
        Stimulus filter
    h : ndarray
        Post-spike filter
    dc : float
        DC offset
    prs : ndarray
        Full set of coefficients for basis vectors, [k_coeffs, h_coeffs, dc]
    kbasis : ndarray
        Basis vectors for stimulus filter
    hbasis : ndarray
        Basis vectors for post-spike filters
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # set defaults
    if nkt is None:
        nkt = 100

    if kbasprs is None:
        # basis functions for stimulus filter
        kbasprs = {
            'neye': 0,  # number of "identity" basis vectors near time of spike
            'ncos': 3,  # number of raised-cosine vectors to use
            'kpeaks': [1, round(nkt / 2)],  # position of first and last bump
            'b': 10  # how nonlinear to make spacings (larger -> more linear)
        }

    if ihbasprs is None:
        # basis functions for post-spike kernel
        ihbasprs = {
            'ncols': 2,  # number of basis vectors for post-spike kernel
            'hpeaks': [1, 100],  # peak location for first and last vectors, in ms
            'b': 10,  # how nonlinear to make spacings (larger -> more linear)
            'absref': 0  # absolute refractory period, in ms
        }

    if softRect is None:
        softRect = 0

    if plotFlag is None:
        plotFlag = 0

    if maxIter is None:
        maxIter = 100

    if tolFun is None:
        tolFun = 1e-8

    if L2pen is None:
        L2pen = 0  # penalty on L2 norm

    refreshRate = 1000 / dt  # stimulus in ms, sampled at dt

    # create basis functions and initialize parameters
    kbasisTemp = makeBasis_StimKernel(kbasprs, nkt)[1]
    nkb = kbasisTemp.shape[1]
    lenkb = kbasisTemp.shape[0]

    # Interpolate to match dt
    kbasis = np.zeros((int(lenkb / dt), nkb))
    for bNum in range(nkb):
        interp_func = interp1d(np.arange(1, lenkb + 1), kbasisTemp[:, bNum], kind='linear', fill_value='extrapolate')
        kbasis[:, bNum] = interp_func(np.linspace(1, lenkb, int(lenkb / dt)))

    ht, hbas, hbasis = makeBasis_PostSpike(ihbasprs, dt)
    hbasis = np.vstack([np.zeros((1, ihbasprs['ncols'])), hbasis])  # enforce causality

    nkbasis = kbasis.shape[1]  # number of basis functions for k
    nhbasis = hbasis.shape[1]  # number of basis functions for h

    if prs is None:
        prs = np.zeros(nkbasis + nhbasis + 1)  # initialize parameters

    # Pre-calculate convolutions
    xconvki = np.zeros((len(y), nkbasis))
    yconvhi = np.zeros((len(y), nhbasis))

    for knum in range(nkbasis):
        xconvki[:, knum] = sameconv(x, kbasis[:, knum])

    for hnum in range(nhbasis):
        yconvhi[:, hnum] = sameconv(y, np.flipud(hbasis[:, hnum]))

    # minimization
    if softRect:
        NL = logexp1

        def fneglogli(prs):
            nll, grad = negloglike_glm_basis_softRect(prs, NL, xconvki, yconvhi, y, 1, refreshRate)
            return nll, grad

        # Use L-BFGS-B for soft-rect (no Hessian)
        res = minimize(
            fneglogli,
            prs,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxIter, 'ftol': tolFun, 'disp': True}
        )
    else:
        NL = np.exp

        def fneglogli(prs):
            nll, grad, hess = negloglike_glm_basis(prs, NL, xconvki, yconvhi, y, 1, refreshRate, L2pen)
            return nll, grad

        def fneglogli_hess(prs):
            nll, grad, hess = negloglike_glm_basis(prs, NL, xconvki, yconvhi, y, 1, refreshRate, L2pen)
            return hess

        # Use trust-ncg for exponential (has Hessian)
        res = minimize(
            fneglogli,
            prs,
            method='trust-ncg',
            jac=True,
            hess=fneglogli_hess,
            options={'maxiter': maxIter, 'gtol': tolFun, 'disp': True}
        )

    prs = res.x

    # calculate filters from basis fcns/weights
    k = kbasis @ prs[:nkbasis]  # k basis functions weighted by given parameters
    h = hbasis @ prs[nkbasis:-1]  # h basis functions weighted by given parameters
    dc = prs[-1]  # dc current (accounts for mean spike rate)

    # plot results
    if plotFlag:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Stimulus basis
        ax = axes[0, 0]
        for i in range(kbasis.shape[1]):
            ax.plot(kbasis[:, i])
        ax.set_xlim([0, len(k)])
        ax.set_title('Stimulus basis')

        # Stimulus filter
        ax = axes[1, 0]
        ax.plot(k)
        ax.set_xlim([0, len(k)])
        ax.set_xlabel('time (ms)')
        ax.set_title('stimulus filter')

        # Post-spike basis
        ax = axes[0, 1]
        for i in range(hbasis.shape[1]):
            ax.plot(hbasis[:, i])
        ax.set_xlim([0, len(h)])
        ax.set_title('Post-spike basis')

        # Post-spike filter
        ax = axes[1, 1]
        ax.plot(h)
        ax.set_xlim([0, len(h)])
        ax.set_xlabel('time (ms)')
        ax.set_title('post-spike filter')

        plt.tight_layout()
        plt.close()

    return k, h, dc, prs, kbasis, hbasis
