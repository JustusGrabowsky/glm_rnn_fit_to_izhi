"""
simulate_glm.py - Faithful translation of simulate_glm.m

This code simulates responses from a fitted Poisson GLM.
"""

import numpy as np

from sameconv import sameconv
from logexp1 import logexp1


def simulate_glm(x, dt, k, h, dc, runs=None, softRect=None, plotFlag=None):
    """
    [y, stimcurr, hcurr, r] = simulate_glm(x, dt, k, h, dc, runs, softRect, plotFlag)

    This code simulates responses from a fitted Poisson GLM.

    Parameters:
    -----------
    x : ndarray
        Stimulus
    dt : float
        Time step of x and y in ms
    k : ndarray
        Stimulus filter
    h : ndarray
        Post-spike filter
    dc : float
        DC offset
    runs : int, optional
        Number of trials to simulate (default: 5)
    softRect : int, optional
        0 uses exponential nonlinearity; 1 uses soft-rectifying nonlinearity (default: 0)
    plotFlag : int, optional
        0 or 1, plot simulated data (default: 0)

    Returns:
    --------
    y : ndarray
        Spike train (0s and 1s), shape (nTimePts, runs)
    stimcurr : ndarray
        Output of stimulus filter (without DC current added)
    hcurr : ndarray
        Output of post-spike filter, shape (nTimePts, runs)
    r : ndarray
        Firing rate (stimcurr + hcurr + dc passed through nonlinearity), shape (nTimePts, runs)
    """
    x = np.asarray(x).flatten()
    k = np.asarray(k).flatten()
    h = np.asarray(h).flatten()

    # set defaults
    if runs is None:
        runs = 5

    if softRect is None:
        softRect = 0

    if plotFlag is None:
        plotFlag = 0

    # generate data with fitted GLM
    nTimePts = len(x)
    refreshRate = 1000 / dt  # stimulus in ms, sampled at dt

    if softRect:
        NL = logexp1
    else:
        NL = np.exp

    g = np.zeros((nTimePts + len(h), runs))  # filtered stimulus + dc
    y = np.zeros((nTimePts, runs))  # initialize response vector
    r = np.zeros((nTimePts + len(h) - 1, runs))  # firing rate
    hcurr = np.zeros_like(g)

    stimcurr = sameconv(x, k)
    Iinj = stimcurr + dc

    for runNum in range(runs):
        g[:, runNum] = np.concatenate([Iinj, np.zeros(len(h))])  # injected current includes DC drive

        # loop to get responses, incorporate post-spike filter
        for t in range(nTimePts):
            r[t, runNum] = NL(g[t, runNum])  # firing rate
            if np.random.rand() < (1 - np.exp(-r[t, runNum] / refreshRate)):  # 1-P(0 spikes)
                y[t, runNum] = 1
                g[t:t + len(h), runNum] = g[t:t + len(h), runNum] + h  # add post-spike filter
                hcurr[t:t + len(h), runNum] = hcurr[t:t + len(h), runNum] + h

    hcurr = hcurr[:nTimePts, :]  # trim zero padding
    r = r[:nTimePts, :]  # trim zero padding

    # plot
    if plotFlag:
        import matplotlib.pyplot as plt

        minT = int(1 / dt)
        maxT = len(x)

        tIdx = np.arange(minT, maxT)
        t = (tIdx - minT) * dt

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        # stimulus
        ax = axes[0]
        ax.plot(t, x[tIdx], linewidth=2)
        ax.set_xlim([t.min(), t.max()])
        ax.set_ylim([x[tIdx].min() - 0.05 * np.abs(x[tIdx].min()),
                     x[tIdx].max() + 0.05 * np.abs(x[tIdx].max())])
        ax.set_title('stimulus')

        # filter outputs
        ax = axes[1]
        ax.plot(t, Iinj[tIdx], linewidth=1.5, label='stim filter')
        ax.plot(t, hcurr[tIdx, 0], 'r', linewidth=1.5, label='post-spike filter')
        ax.set_xlim([t.min(), t.max()])
        ax.legend()
        ax.set_title('filter outputs')

        # prob(firing)
        ax = axes[2]
        ax.semilogy(t, NL(hcurr[tIdx, 0] + Iinj[tIdx]), color=[0.5, 0.5, 0.5], linewidth=1.5)
        ax.set_xlim([t.min(), t.max()])
        ax.set_title('prob(firing)')

        # GLM spikes
        ax = axes[3]
        spikeHeight = 0.7

        for i in range(y.shape[1]):  # for each run of glm simulation
            spt = np.where(y[tIdx, i])[0]
            for spikeNum in range(len(spt)):
                ax.plot([spt[spikeNum] * dt, spt[spikeNum] * dt],
                        [i - 0.5, i - 0.5 + spikeHeight],
                        color=[0.5, 0.5, 0.5], linewidth=1.25)

        ax.set_xlim([0, t.max() - t.min()])
        ax.set_ylim([0, runs + spikeHeight])
        ax.set_xlabel('time (ms)')
        ax.set_title('spikes')

        plt.tight_layout()
        plt.close()

    return y, stimcurr, hcurr, r
