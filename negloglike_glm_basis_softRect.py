"""
negloglike_glm_basis_softRect.py - Faithful translation of negloglike_glm_basis_softRect.m

Calculate negative log likelihood and gradient for GLM with soft-rectification.
"""

import numpy as np


def negloglike_glm_basis_softRect(prs, NL, xconvki, yconvhi, y, dt, refreshRate):
    """
    [negloglike, dL] = negloglike_glm_basis_softRect(prs, NL, xconvki, yconvhi, y, dt, refreshRate)

    Parameters:
    -----------
    prs : ndarray
        Vector of parameters, coefficients for basis functions in order
        [kprs (stim filter); hprs (post-spike filter); dc]
    NL : callable
        Function handle for nonlinearity (e.g., logexp1)
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

    Returns:
    --------
    negloglike : float
        Negative log likelihood
    dL : ndarray
        Gradient vector

    Notes:
    ------
    Gives same output as negloglike_glm_basis_old, neglogli_GLM, and Loss_GLM_logli
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

    loglambda = np.log(lambda_val)

    negloglike = -np.dot(y, loglambda) + dt * np.sum(lambda_val) / refreshRate

    # calculate negative gradient
    dL = np.zeros_like(prs)

    # kterms
    # trm1all = repmat(exp(g)./lambda./exp(lambda),1,size(xconvki,2)).*xconvki
    exp_g = np.exp(g)
    exp_lambda = np.exp(lambda_val)
    factor1 = exp_g / lambda_val / exp_lambda

    trm1all = np.tile(factor1.reshape(-1, 1), (1, xconvki.shape[1])) * xconvki
    trm1 = trm1all.T @ y  # sum of above term at t=spike

    factor2 = exp_g / exp_lambda
    trm2all = np.tile(factor2.reshape(-1, 1), (1, xconvki.shape[1])) * xconvki
    trm2 = np.sum(trm2all, axis=0) * dt / refreshRate
    dL[:nkbasis] = trm2 - trm1  # negative gradient

    # hterms
    trm1all = np.tile(factor1.reshape(-1, 1), (1, yconvhi.shape[1])) * yconvhi
    trm1 = trm1all.T @ y  # sum of above term at t=spike

    trm2all = np.tile(factor2.reshape(-1, 1), (1, yconvhi.shape[1])) * yconvhi
    trm2 = np.sum(trm2all, axis=0) * dt / refreshRate
    dL[nkbasis:-1] = trm2 - trm1  # negative gradient

    # dc
    dL[-1] = -np.dot(y, factor1) + np.sum(factor2) * dt / refreshRate  # negative gradient

    return negloglike, dL
