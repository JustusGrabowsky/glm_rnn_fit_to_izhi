"""
negloglike_glm_basis.py - Faithful translation of negloglike_glm_basis.m

Calculate negative log likelihood, gradient, and Hessian for GLM.
"""

import numpy as np


def negloglike_glm_basis(prs, NL, xconvki, yconvhi, y, dt, refreshRate, L2pen):
    """
    [negloglike, dL, H] = negloglike_glm_basis(prs, NL, xconvki, yconvhi, y, dt, refreshRate, L2pen)

    Parameters:
    -----------
    prs : ndarray
        Vector of parameters, coefficients for basis functions in order
        [kprs (stim filter); hprs (post-spike filter); dc]
    NL : callable
        Function handle for nonlinearity (e.g., np.exp)
    xconvki : ndarray
        Stimulus convolved with each filter basis vector,
        upsampled to match response sampling rate
    yconvhi : ndarray
        Response vector convolved with each filter basis vector
    y : ndarray
        Response vector (zeros and ones)
    dt : float
        Time scale of y (in frames/stimulus frame)
    refreshRate : float
        Refresh rate of stimulus (frames/sec)
    L2pen : float
        Penalty on L2 norm of prs vector

    Returns:
    --------
    negloglike : float
        Negative log likelihood
    dL : ndarray
        Gradient vector
    H : ndarray
        Hessian matrix
    """
    prs = np.asarray(prs).flatten()
    y = np.asarray(y).flatten()

    # calculate negative log likelihood
    nkbasis = xconvki.shape[1]  # number of basis functions for k

    kprs = prs[:nkbasis]  # k basis functions weighted by given parameters
    hprs = prs[nkbasis:-1]  # h basis functions weighted by given parameters
    dc = prs[-1]  # dc current (accounts for mean spike rate)

    xconvk_dc = xconvki @ kprs + dc

    yconvh = yconvhi @ hprs  # same as output of spikeconv_mex

    g = xconvk_dc + yconvh  # g = loglambda for NL = @exp
    lambda_val = NL(g)

    negloglike = -np.dot(y, g) + dt * np.sum(lambda_val) / refreshRate + L2pen * np.dot(prs, prs)

    # calculate negative gradient
    dL = np.zeros_like(prs)
    prsMat = np.hstack([xconvki, yconvhi, np.ones((xconvki.shape[0], 1))])
    for pr in range(len(prs)):
        dL[pr] = -np.sum(prsMat[y.astype(bool), pr]) + dt / refreshRate * np.sum(prsMat[:, pr] * lambda_val) + L2pen * 2 * prs[pr]

    # calculate negative Hessian
    H = np.zeros((len(prs), len(prs)))

    for pr1 in range(len(prs)):
        for pr2 in range(pr1, len(prs)):
            H[pr1, pr2] = dt / refreshRate * np.sum(prsMat[:, pr1] * prsMat[:, pr2] * lambda_val) + L2pen * 2 * (pr1 == pr2)
            H[pr2, pr1] = H[pr1, pr2]

    return negloglike, dL, H
