"""
run_all.py - Integrated GLM fitting and RNN training pipeline

This script:
1. Fits GLM to Izhikevich neuron data
2. Trains three RNN models (GRU, Vanilla RNN, LSTM) on trial-based data
3. Trains extended RNN variants with smoothing and history features
4. Evaluates ALL models on the EXACT same stimulus
5. Creates integrated comparison plots
6. Saves results to result_plots/<neuron_type>/

Usage:
    python run_all.py           # Run all neuron types
    python run_all.py all       # Explicit all neuron types
    python run_all.py 4         # Run single neuron type (e.g., phasic bursting)
    python run_all.py 1 2 3 4   # Run multiple specific neuron types
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Optional, Tuple

from generate_izhikevich_stim import generate_izhikevich_stim
from simulate_izhikevich import simulate_izhikevich
from fit_glm import fit_glm
from simulate_glm import simulate_glm
from rnn_models import TorchRNNRegressor, TorchVanillaRNNRegressor, TorchLSTMRegressor
from rnn_data_generator import RNNDataGenerator
from izhikevich_configs import cids, index_to_name


# =============================================================================
# Configuration
# =============================================================================

NEURON_TYPES = {
    1: 'tonic_spiking',
    2: 'phasic_spiking',
    3: 'tonic_bursting',
    4: 'phasic_bursting',
    5: 'mixed_mode',
    6: 'spike_frequency_adaptation',
    7: 'class_1',
    8: 'class_2',
    9: 'spike_latency',
    11: 'resonator',
    12: 'integrator',
    13: 'rebound_spike',
    14: 'rebound_burst',
    15: 'threshold_variability',
    16: 'bistability',
    18: 'accomodation',
    19: 'inhibition_induced_spiking',
    20: 'inhibition_induced_bursting',
    21: 'bistability_2',
}

# GLM Configuration
GLM_MAX_ITER = 1000
GLM_TOL_FUN = 1e-12
GLM_L2_PEN = 0
GLM_N_RUNS = 5  # Number of GLM simulation runs to show

# RNN Configuration
RNN_N_TRIALS = 2000
RNN_N_EPOCHS = 10
RNN_HIDDEN_DIM = 64
RNN_BIN_SIZE_MS = 1.0

# Extended RNN Configuration
RNN_SMOOTH_SIGMA_MS = 3.0  # Gaussian smoothing sigma for targets
TRAIN_EXTENDED_MODELS = True  # Whether to train extended model variants

# Advanced GRU Configuration
TRAIN_ADVANCED_MODELS = True  # Whether to train advanced GRU variants
ADVANCED_SPARSITY_WEIGHT = 10.0  # Weight for spike bins in weighted loss
ADVANCED_SCHEDULED_SAMPLING_DECAY = 0.99  # Decay rate for scheduled sampling


def logexp1(x):
    """Numerically stable log(1 + exp(x))."""
    return np.logaddexp(0, x)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_data(X, y, save_path, n_examples=5, title="RNN Training Data"):
    """Visualize training data samples with raw spike counts."""
    fig, axes = plt.subplots(n_examples, 2, figsize=(14, 3 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    indices = np.random.choice(len(X), size=min(n_examples, len(X)), replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i, 0]
        t = np.arange(X.shape[1])
        ax.plot(t, X[idx, :, 0], 'b', linewidth=1.5)
        ax.set_ylabel('Current')
        if i == 0:
            ax.set_title('Stimulus (I)')
        if i == n_examples - 1:
            ax.set_xlabel('Time (ms)')
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.bar(t, y[idx], width=1.0, color='k', alpha=0.7)
        ax.set_ylabel('Spike count')
        if i == 0:
            ax.set_title('Target (spike counts)')
        if i == n_examples - 1:
            ax.set_xlabel('Time (ms)')
        ax.grid(True, alpha=0.3)

        n_spikes = y[idx].sum()
        axes[i, 0].text(0.02, 0.95, f'Trial {idx} ({n_spikes:.0f} spikes)',
                        transform=axes[i, 0].transAxes, fontsize=9, va='top')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_smoothed_training_data(X, y_smoothed, save_path, n_examples=5,
                                 title="RNN Training Data (Smoothed Targets)"):
    """Visualize training data samples with smoothed firing rates."""
    fig, axes = plt.subplots(n_examples, 2, figsize=(14, 3 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    indices = np.random.choice(len(X), size=min(n_examples, len(X)), replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i, 0]
        t = np.arange(X.shape[1])
        ax.plot(t, X[idx, :, 0], 'b', linewidth=1.5)
        ax.set_ylabel('Current')
        if i == 0:
            ax.set_title('Stimulus (I)')
        if i == n_examples - 1:
            ax.set_xlabel('Time (ms)')
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.plot(t, y_smoothed[idx], 'r', linewidth=1.5)
        ax.fill_between(t, 0, y_smoothed[idx], alpha=0.3, color='r')
        ax.set_ylabel('Smoothed rate')
        if i == 0:
            ax.set_title('Target (smoothed firing rate)')
        if i == n_examples - 1:
            ax.set_xlabel('Time (ms)')
        ax.grid(True, alpha=0.3)

        total_rate = y_smoothed[idx].sum()
        axes[i, 0].text(0.02, 0.95, f'Trial {idx} (sum={total_rate:.1f})',
                        transform=axes[i, 0].transAxes, fontsize=9, va='top')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_losses(train_losses_dict, save_path):
    """Plot training losses for base RNN models only."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'GRU': 'tab:blue', 'Vanilla RNN': 'tab:orange', 'LSTM': 'tab:green'}

    for model_name, losses in train_losses_dict.items():
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, color=colors[model_name], linewidth=2,
                marker='o', markersize=4, label=model_name, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Poisson NLL Loss')
    ax.set_title('RNN Training Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_losses_extended(train_losses_dict: Dict[str, List[float]], save_path: str):
    """
    Plot training losses for all model variations.

    Args:
        train_losses_dict: Dictionary with model names as keys, loss lists as values.
                          Keys can include variants like 'GRU (Smoothed)', 'LSTM (Spike Hist)', etc.
                          Also includes advanced models like 'GRU_Derivative', 'BiGRU', etc.
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for different model types (base + advanced)
    model_colors = {
        # Base models
        'GRU': 'tab:blue',
        'Vanilla RNN': 'tab:orange',
        'LSTM': 'tab:green',
        # Advanced GRU models
        'GRU_Derivative': 'tab:cyan',
        'GRU_Weighted': 'tab:purple',
        'GRU_ClosedLoop': 'tab:pink',
        'BiGRU': 'darkblue',
        'GRU_Attention': 'mediumvioletred',
    }

    variant_styles = {
        '': '-',           # Standard
        'Smoothed': '--',  # Smoothed targets
        'Spike Hist': ':',  # Spike history
        'Input Hist': '-.',  # Input history
        'Full Hist': (0, (3, 1, 1, 1)),  # Full history (dash-dot-dot)
    }

    for model_name, losses in train_losses_dict.items():
        epochs = np.arange(1, len(losses) + 1)

        # Check if it's an advanced model (exact match)
        if model_name in model_colors:
            color = model_colors[model_name]
            linestyle = '-'
            marker = 's'  # Square marker for advanced models
        else:
            # Parse model name to get base and variant (for extended models)
            base_name = model_name.split(' (')[0]
            variant = ''
            if '(' in model_name:
                variant = model_name.split('(')[1].rstrip(')')

            color = model_colors.get(base_name, 'gray')
            linestyle = variant_styles.get(variant, '-')
            marker = 'o' if variant == '' else None

        ax.plot(epochs, losses, color=color, linewidth=2, linestyle=linestyle,
                marker=marker, markersize=3,
                label=model_name, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Poisson NLL Loss', fontsize=12)
    ax.set_title('RNN Training Loss (All Variants)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spike_raster_comparison(
    I_binned: np.ndarray,
    y_true: np.ndarray,
    all_predictions: Dict[str, np.ndarray],
    rnn_bin_size_ms: float,
    save_path: str,
    cell_name: str,
    cid: str,
    max_time_ms: float = 1500.0
):
    """
    Create a comprehensive spike raster comparison plot.

    Shows Izhikevich ground truth and all RNN model predictions on clean step input.
    Only shows the first portion of data (default: first 1500ms) for clarity.

    Args:
        I_binned: Binned stimulus (for reference)
        y_true: True binned spike counts
        all_predictions: Dict of model_name -> predicted rates
        rnn_bin_size_ms: RNN bin size in ms
        save_path: Path to save the plot
        cell_name: Cell type name for title
        cid: Cell ID for title
        max_time_ms: Maximum time to display (default 1500ms, ~2 pulses)
    """
    # Limit data to first max_time_ms (for clearer visualization)
    max_bins = int(max_time_ms / rnn_bin_size_ms)
    max_bins = min(max_bins, len(I_binned))  # Don't exceed data length

    I_binned = I_binned[:max_bins]
    y_true = y_true[:max_bins]
    all_predictions = {k: v[:max_bins] for k, v in all_predictions.items()}

    n_models = len(all_predictions) + 1  # +1 for ground truth

    fig, axes = plt.subplots(n_models + 1, 1, figsize=(14, 2 + 1.2 * n_models),
                              sharex=True, gridspec_kw={'height_ratios': [1] + [1] * n_models})

    t = np.arange(len(I_binned)) * rnn_bin_size_ms

    # Panel 0: Stimulus
    ax = axes[0]
    ax.plot(t, I_binned, 'b', linewidth=1.5)
    ax.set_ylabel('Current')
    ax.set_title(f'{cell_name} ({cid}) - Spike Raster Comparison (First {max_time_ms:.0f}ms)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 1: Ground truth (Izhikevich)
    ax = axes[1]
    spike_times = t[y_true > 0]
    ax.eventplot([spike_times], colors=['k'], linewidths=1.5, linelengths=0.8)
    ax.set_ylabel('Izhi', fontsize=10)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Define colors for model types (base models and advanced variants)
    model_colors = {
        # Base models
        'GRU': 'tab:blue',
        'Vanilla RNN': 'tab:orange',
        'LSTM': 'tab:green',
        # Advanced GRU models (different shades of blue/purple)
        'GRU_Derivative': 'tab:cyan',
        'GRU_Weighted': 'tab:purple',
        'GRU_ClosedLoop': 'tab:pink',
        'BiGRU': 'darkblue',
        'GRU_Attention': 'mediumvioletred',
    }

    # Remaining panels: Model predictions
    for i, (model_name, pred) in enumerate(all_predictions.items()):
        ax = axes[i + 2]

        # Sample spikes from Poisson distribution
        np.random.seed(42 + i)
        sampled_spikes = np.random.poisson(np.clip(pred, 0, 10))
        spike_times_model = t[sampled_spikes > 0]

        # Get color based on model name (try exact match first, then base name)
        if model_name in model_colors:
            color = model_colors[model_name]
        else:
            # Fall back to base model name (for extended variants like "GRU (Smoothed)")
            base_name = model_name.split(' (')[0]
            color = model_colors.get(base_name, 'gray')

        ax.eventplot([spike_times_model], colors=[color], linewidths=1.5, linelengths=0.8)

        # Short label
        label = model_name
        if len(label) > 15:
            label = label.replace('Vanilla RNN', 'V-RNN')
        ax.set_ylabel(label, fontsize=8, rotation=0, ha='right', va='center')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axes[-1].set_xlabel('Time (ms)', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(
    cell_type, I_iz, spikes, v, dt, k, h, dc,
    rnn_models, rnn_predictions_binned, rnn_bin_size_ms,
    soft_rect=0, save_dir='plots'
):
    """
    Create comprehensive comparison plot of GLM + RNN vs Izhikevich neuron.

    Shows:
    - Left: Time series (stimulus, voltage, filter outputs, conditional intensity, spike rasters)
    - Right: Filters (k, h)
    - Spike rasters include: Izhikevich, 5 GLM runs, 3 RNN models (base only)
    """
    # Time windows for each cell type
    Ts = np.array([
        [400, 1150], [400, 1150], [400, 1150], [400, 1150], [400, 1150], [400, 1150],
        [9250, 15250], [8250, 14250], [750, 1050], [200, 1100], [1900, 3000],
        [600, 900], [600, 900], [600, 900], [1400, 2200], [125, 225], [200, 1100],
        [300, 2200], [200, 1100], [200, 1100], [525, 725]
    ])

    cid_idx = cell_type - 1
    cid = cids[cid_idx]

    NL = logexp1 if soft_rect else np.exp

    # Simulate GLM
    y_glm, stimcurr, hcurr, r = simulate_glm(I_iz, dt, k, h, dc, GLM_N_RUNS, soft_rect, plotFlag=0)

    # Setup time indices
    minT = max(Ts[cid_idx, 0], 1)
    maxT = Ts[cid_idx, 1]
    minT_idx = int(minT / dt)
    maxT_idx = int(maxT / dt)
    tIdx = np.arange(minT_idx, maxT_idx + 1)
    tIdx = tIdx[tIdx < len(spikes)]
    t = (tIdx - minT_idx) * dt

    # Plotting parameters
    axisLabelFontSize = 11
    axisTickLabelFontSize = 10
    axisWidth = 1
    izColor = 'k'
    glmColor = [0.5, 0.5, 0.5]
    rnnColors = {'GRU': 'tab:blue', 'Vanilla RNN': 'tab:orange', 'LSTM': 'tab:green'}

    cell_name = index_to_name.get(cell_type, f"Cell {cell_type}")
    fig = plt.figure(figsize=(12, 14))
    fig.suptitle(f'{cell_name} - GLM + RNN vs Izhikevich', fontsize=14, fontweight='bold', y=0.995)

    # ==========================================================================
    # RIGHT SIDE: Filters
    # ==========================================================================
    n_samples_100ms = int(100 / dt)

    # Stimulus Filter k
    pos_k = [0.72, 0.55, 0.22, 0.22]
    ax_k = fig.add_axes(pos_k)
    k_100ms = k[-n_samples_100ms:] if len(k) >= n_samples_100ms else k
    k_time = np.linspace(-100, 0, len(k_100ms))
    ax_k.plot([-100, 0], [0, 0], 'k--', linewidth=1)
    ax_k.plot(k_time, k_100ms, 'b', linewidth=2)
    ax_k.set_xlim([-100, 0])
    ax_k.set_xticks([-100, -50, 0])
    ax_k.set_xlabel('time (ms)', fontsize=axisLabelFontSize)
    ax_k.set_title('Stimulus Filter (k)', fontsize=axisLabelFontSize, fontweight='bold')
    mu_val = np.round(dc * 10) / 10
    ax_k.text(-90, np.max(k_100ms) * 0.8, fr'$\mu = {mu_val}$', fontsize=axisLabelFontSize)
    ax_k.tick_params(direction='out', width=axisWidth, labelsize=axisTickLabelFontSize)
    ax_k.spines['top'].set_visible(False)
    ax_k.spines['right'].set_visible(False)

    # Post-Spike Filter h
    pos_h = [0.72, 0.22, 0.22, 0.22]
    ax_h = fig.add_axes(pos_h)
    h_100ms = h[1:n_samples_100ms+1] if len(h) > n_samples_100ms else h[1:]
    h_time = np.linspace(0, 100, len(h_100ms))
    ax_h.plot([0, 100], [0, 0], 'k--', linewidth=1)
    ax_h.plot(h_time, h_100ms, 'r', linewidth=2)
    ax_h.set_xlim([0, 100])
    ax_h.set_xticks([0, 50, 100])
    ax_h.set_xlabel('time (ms)', fontsize=axisLabelFontSize)
    ax_h.set_title('Post-Spike Filter (h)', fontsize=axisLabelFontSize, fontweight='bold')
    ax_h.tick_params(direction='out', width=axisWidth, labelsize=axisTickLabelFontSize)
    ax_h.spines['top'].set_visible(False)
    ax_h.spines['right'].set_visible(False)

    # ==========================================================================
    # LEFT SIDE: Time series (5 panels)
    # ==========================================================================
    t_max = np.max(t)
    xticks = np.arange(0, t_max + 1, 100)

    I_slice = I_iz[tIdx]
    v_slice = v[tIdx]
    Iinj = stimcurr + dc
    Iinj_slice = Iinj[tIdx]
    hcurr_slice = hcurr[tIdx, 0] if hcurr.ndim > 1 else hcurr[tIdx]

    panel_height = 0.12
    panel_gap = 0.03
    left_margin = 0.08
    panel_width = 0.58
    bottom_start = 0.06

    # Panel 1: Stimulus
    pos1 = [left_margin, bottom_start + 4*(panel_height + panel_gap), panel_width, panel_height]
    ax1 = fig.add_axes(pos1)
    ax1.plot(t, I_slice, 'b', linewidth=1.5)
    ax1.set_xlim([0, t_max])
    ax1.set_ylabel('Input', fontsize=axisLabelFontSize)
    ax1.set_title('Stimulus', fontsize=axisLabelFontSize, fontweight='bold')
    ax1.set_xticks([])
    ax1.tick_params(direction='out', width=axisWidth, labelsize=axisTickLabelFontSize)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    # Panel 2: Izhikevich Voltage
    pos2 = [left_margin, bottom_start + 3*(panel_height + panel_gap), panel_width, panel_height]
    ax2 = fig.add_axes(pos2)
    ax2.plot(t, v_slice, 'k', linewidth=1)
    ax2.set_xlim([0, t_max])
    ax2.set_ylabel('V (mV)', fontsize=axisLabelFontSize)
    ax2.set_title('Izhikevich Neuron Response', fontsize=axisLabelFontSize, fontweight='bold')
    ax2.set_xticks([])
    ax2.tick_params(direction='out', width=axisWidth, labelsize=axisTickLabelFontSize)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # Panel 3: Filter Outputs
    pos3 = [left_margin, bottom_start + 2*(panel_height + panel_gap), panel_width, panel_height]
    ax3 = fig.add_axes(pos3)
    ax3.plot(t, Iinj_slice, 'b', linewidth=1.5, label=r'$\vec{k}$ output')
    ax3.plot(t, hcurr_slice, 'r', linewidth=1.5, label=r'$\vec{h}$ output')
    ax3.set_xlim([0, t_max])
    ax3.set_ylabel('Filter Outputs', fontsize=axisLabelFontSize)
    ax3.legend(loc='lower left', fontsize=9, frameon=False)
    ax3.set_xticks([])
    ax3.tick_params(direction='out', width=axisWidth, labelsize=axisTickLabelFontSize)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)

    # Panel 4: Conditional Intensity
    pos4 = [left_margin, bottom_start + 1*(panel_height + panel_gap), panel_width, panel_height]
    ax4 = fig.add_axes(pos4)
    prob_val = NL(hcurr_slice + Iinj_slice)
    ax4.semilogy(t, prob_val, color=glmColor, linewidth=1.5)
    ax4.set_xlim([0, t_max])
    ax4.set_ylim([0.1, 10**6])
    ax4.set_ylabel(r'$\lambda$ (spikes/s)', fontsize=axisLabelFontSize)
    ax4.set_title('Conditional Intensity', fontsize=axisLabelFontSize, fontweight='bold')
    ax4.set_yticks([1, 10**3, 10**6])
    ax4.set_xticks([])
    ax4.tick_params(direction='out', width=axisWidth, labelsize=axisTickLabelFontSize)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)

    # Panel 5: Spike Rasters (GLM + RNN + Izhikevich)
    pos5 = [left_margin, bottom_start, panel_width, panel_height + 0.02]
    ax5 = fig.add_axes(pos5)

    spikeHeight = 0.7
    if y_glm.ndim == 1:
        y_glm = y_glm[:, np.newaxis]

    # Row positions: GLM runs at bottom, then RNNs, then Izhikevich at top
    row_idx = 0

    # Plot GLM spikes (5 runs)
    for i in range(GLM_N_RUNS):
        current_y_slice = y_glm[tIdx, i]
        spt = np.where(current_y_slice)[0]
        y_base = (i + 1) - 0.5

        for spike_idx in spt:
            time_ms = spike_idx * dt
            ax5.plot([time_ms, time_ms], [y_base, y_base + spikeHeight], color=glmColor, linewidth=1)

    row_idx = GLM_N_RUNS

    # Plot RNN spikes (base models only)
    minT_bin = int(minT / rnn_bin_size_ms)
    maxT_bin = int(maxT / rnn_bin_size_ms)
    rnn_tIdx = np.arange(minT_bin, min(maxT_bin + 1, len(list(rnn_predictions_binned.values())[0])))

    for model_name in ['GRU', 'Vanilla RNN', 'LSTM']:
        if model_name in rnn_predictions_binned:
            pred = rnn_predictions_binned[model_name]
            # Sample spikes from Poisson
            np.random.seed(42 + row_idx)
            pred_slice = pred[rnn_tIdx] if len(rnn_tIdx) > 0 else pred
            sampled_spikes = np.random.poisson(np.clip(pred_slice, 0, 10))
            spt = np.where(sampled_spikes > 0)[0]
            y_base = row_idx + 0.5

            for spike_idx in spt:
                time_ms = spike_idx * rnn_bin_size_ms
                ax5.plot([time_ms, time_ms], [y_base, y_base + spikeHeight],
                        color=rnnColors[model_name], linewidth=1.2)
            row_idx += 1

    # Plot Izhikevich spikes (top)
    spikes_slice = spikes[tIdx]
    spt_iz = np.where(spikes_slice)[0]
    y_base_iz = row_idx + 0.5

    for spike_idx in spt_iz:
        time_ms = spike_idx * dt
        ax5.plot([time_ms, time_ms], [y_base_iz, y_base_iz + spikeHeight], color=izColor, linewidth=1.5)

    ax5.set_xlim([0, t_max])
    ax5.set_ylim([0, row_idx + 2])
    ax5.set_xlabel('time (ms)', fontsize=axisLabelFontSize)
    ax5.set_xticks(xticks)

    # No y-axis, just legend
    ax5.set_yticks([])
    ax5.spines['left'].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=izColor, linewidth=2, label='Izhikevich'),
        Line2D([0], [0], color=rnnColors['LSTM'], linewidth=2, label='LSTM'),
        Line2D([0], [0], color=rnnColors['Vanilla RNN'], linewidth=2, label='Vanilla RNN'),
        Line2D([0], [0], color=rnnColors['GRU'], linewidth=2, label='GRU'),
        Line2D([0], [0], color=glmColor, linewidth=2, label='GLM repeats')
    ]
    ax5.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=False)

    ax5.tick_params(direction='out', width=axisWidth, labelsize=axisTickLabelFontSize)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Save plot
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plot_filename = f"comparison_{cid}.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    return plot_path


# =============================================================================
# Training Functions
# =============================================================================

def train_base_rnn_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True
) -> Dict[str, object]:
    """
    Train the three base RNN models (GRU, Vanilla RNN, LSTM).

    Args:
        X_train: Input sequences (n_trials, seq_len, n_features)
        y_train: Target spike counts (n_trials, seq_len)
        verbose: Print progress

    Returns:
        Dictionary of trained models
    """
    rnn_models = {}

    if verbose:
        print("\n  Training GRU...")
    gru = TorchRNNRegressor(hidden_dim=RNN_HIDDEN_DIM, n_epochs=RNN_N_EPOCHS,
                             batch_size=32, learning_rate=0.001, verbose=verbose)
    gru.fit(X_train, y_train)
    rnn_models['GRU'] = gru

    if verbose:
        print("\n  Training Vanilla RNN...")
    vanilla_rnn = TorchVanillaRNNRegressor(hidden_dim=RNN_HIDDEN_DIM, n_epochs=RNN_N_EPOCHS,
                                            batch_size=32, learning_rate=0.001, verbose=verbose)
    vanilla_rnn.fit(X_train, y_train)
    rnn_models['Vanilla RNN'] = vanilla_rnn

    if verbose:
        print("\n  Training LSTM...")
    lstm = TorchLSTMRegressor(hidden_dim=RNN_HIDDEN_DIM, n_epochs=RNN_N_EPOCHS,
                               batch_size=32, learning_rate=0.001, verbose=verbose)
    lstm.fit(X_train, y_train)
    rnn_models['LSTM'] = lstm

    return rnn_models


def train_extended_rnn_models(
    generator: RNNDataGenerator,
    cellType: int,
    verbose: bool = True
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    """
    Train extended RNN model variants with smoothing and history features.

    Trains for each architecture (GRU, Vanilla RNN, LSTM):
    - Smoothed: Gaussian smoothed targets
    - Spike Hist: Input includes S_{t-1}
    - Input Hist: Input includes I_{t-1}
    - Full Hist: Input includes I_{t-1} and S_{t-1}

    Args:
        generator: RNNDataGenerator instance
        cellType: Izhikevich neuron type
        verbose: Print progress

    Returns:
        Tuple of (extended_models_dict, X_base, y_base)
    """
    if verbose:
        print(f"\n  Generating extended training data...")

    # Generate base data with smoothed targets
    X_base, y_base, y_smoothed = generator.generate_training_data_extended(
        cellType,
        n_trials=RNN_N_TRIALS,
        verbose=verbose,
        smooth_sigma_ms=RNN_SMOOTH_SIGMA_MS,
        history_mode='none'  # Base features only
    )

    extended_models = {}

    # Define model classes
    model_classes = {
        'GRU': TorchRNNRegressor,
        'Vanilla RNN': TorchVanillaRNNRegressor,
        'LSTM': TorchLSTMRegressor
    }

    # Train smoothed variants (using base X, smoothed y)
    if verbose:
        print("\n  Training smoothed target variants...")
    for base_name, ModelClass in model_classes.items():
        model_name = f"{base_name} (Smoothed)"
        if verbose:
            print(f"    Training {model_name}...")
        model = ModelClass(hidden_dim=RNN_HIDDEN_DIM, n_epochs=RNN_N_EPOCHS,
                          batch_size=32, learning_rate=0.001, verbose=False)
        model.fit(X_base, y_smoothed)
        extended_models[model_name] = model

    # Train spike history variants
    if verbose:
        print("\n  Training spike history variants...")
    X_spike_hist = generator.add_history_features(X_base, y_base, mode='spike')
    for base_name, ModelClass in model_classes.items():
        model_name = f"{base_name} (Spike Hist)"
        if verbose:
            print(f"    Training {model_name}...")
        model = ModelClass(hidden_dim=RNN_HIDDEN_DIM, n_epochs=RNN_N_EPOCHS,
                          batch_size=32, learning_rate=0.001, verbose=False)
        model.fit(X_spike_hist, y_base)
        extended_models[model_name] = model

    # Train input history variants
    if verbose:
        print("\n  Training input history variants...")
    X_input_hist = generator.add_history_features(X_base, y_base, mode='input')
    for base_name, ModelClass in model_classes.items():
        model_name = f"{base_name} (Input Hist)"
        if verbose:
            print(f"    Training {model_name}...")
        model = ModelClass(hidden_dim=RNN_HIDDEN_DIM, n_epochs=RNN_N_EPOCHS,
                          batch_size=32, learning_rate=0.001, verbose=False)
        model.fit(X_input_hist, y_base)
        extended_models[model_name] = model

    # Train full history variants
    if verbose:
        print("\n  Training full history variants...")
    X_full_hist = generator.add_history_features(X_base, y_base, mode='full')
    for base_name, ModelClass in model_classes.items():
        model_name = f"{base_name} (Full Hist)"
        if verbose:
            print(f"    Training {model_name}...")
        model = ModelClass(hidden_dim=RNN_HIDDEN_DIM, n_epochs=RNN_N_EPOCHS,
                          batch_size=32, learning_rate=0.001, verbose=False)
        model.fit(X_full_hist, y_base)
        extended_models[model_name] = model

    return extended_models, X_base, y_base


def train_advanced_gru_models(
    generator: RNNDataGenerator,
    cellType: int,
    X_base: np.ndarray,
    y_base: np.ndarray,
    verbose: bool = True
) -> Dict[str, object]:
    """
    Train 5 advanced GRU model variants.

    Models trained:
    1. GRU_Derivative: Standard GRU with derivative (dI/dt) feature
    2. GRU_Weighted: Standard GRU with sparsity-weighted loss
    3. GRU_ClosedLoop: GRU with spike history + scheduled sampling
    4. BiGRU: Bidirectional GRU
    5. GRU_Attention: GRU with self-attention

    Args:
        generator: RNNDataGenerator instance
        cellType: Izhikevich neuron type
        X_base: Base training input (n_trials, seq_len, 1)
        y_base: Target spike counts (n_trials, seq_len)
        verbose: Print progress

    Returns:
        Dictionary of trained models with metadata for evaluation
    """
    advanced_models = {}

    # =========================================================================
    # 1. GRU_Derivative: Add derivative (dI/dt) as feature
    # =========================================================================
    if verbose:
        print("\n  Training GRU_Derivative (with dI/dt feature)...")

    # Generate data with derivative feature
    X_deriv, y_deriv = generator.generate_training_data(
        cellType,
        n_trials=RNN_N_TRIALS,
        verbose=False,
        add_derivative=True
    )

    model_deriv = TorchRNNRegressor(
        hidden_dim=RNN_HIDDEN_DIM,
        n_epochs=RNN_N_EPOCHS,
        batch_size=32,
        learning_rate=0.001,
        verbose=False
    )
    model_deriv.fit(X_deriv, y_deriv)
    advanced_models['GRU_Derivative'] = {
        'model': model_deriv,
        'needs_derivative': True,
        'history_mode': 'none'
    }

    # =========================================================================
    # 2. GRU_Weighted: Standard GRU with sparsity-weighted loss
    # =========================================================================
    if verbose:
        print("  Training GRU_Weighted (sparsity_weight={})...".format(ADVANCED_SPARSITY_WEIGHT))

    model_weighted = TorchRNNRegressor(
        hidden_dim=RNN_HIDDEN_DIM,
        n_epochs=RNN_N_EPOCHS,
        batch_size=32,
        learning_rate=0.001,
        sparsity_weight=ADVANCED_SPARSITY_WEIGHT,
        verbose=False
    )
    model_weighted.fit(X_base, y_base)
    advanced_models['GRU_Weighted'] = {
        'model': model_weighted,
        'needs_derivative': False,
        'history_mode': 'none'
    }

    # =========================================================================
    # 3. GRU_ClosedLoop: GRU with spike history + scheduled sampling
    # =========================================================================
    if verbose:
        print("  Training GRU_ClosedLoop (scheduled_sampling_decay={})...".format(
            ADVANCED_SCHEDULED_SAMPLING_DECAY))

    # Add spike history to input
    X_spike_hist = generator.add_history_features(X_base, y_base, mode='spike')

    model_closed = TorchRNNRegressor(
        hidden_dim=RNN_HIDDEN_DIM,
        n_epochs=RNN_N_EPOCHS,
        batch_size=32,
        learning_rate=0.001,
        scheduled_sampling_decay=ADVANCED_SCHEDULED_SAMPLING_DECAY,
        verbose=False
    )
    model_closed.fit(X_spike_hist, y_base)
    advanced_models['GRU_ClosedLoop'] = {
        'model': model_closed,
        'needs_derivative': False,
        'history_mode': 'spike'
    }

    # =========================================================================
    # 4. BiGRU: Bidirectional GRU
    # =========================================================================
    if verbose:
        print("  Training BiGRU (bidirectional)...")

    model_bi = TorchRNNRegressor(
        hidden_dim=RNN_HIDDEN_DIM,
        n_epochs=RNN_N_EPOCHS,
        batch_size=32,
        learning_rate=0.001,
        bidirectional=True,
        verbose=False
    )
    model_bi.fit(X_base, y_base)
    advanced_models['BiGRU'] = {
        'model': model_bi,
        'needs_derivative': False,
        'history_mode': 'none'
    }

    # =========================================================================
    # 5. GRU_Attention: GRU with self-attention
    # =========================================================================
    if verbose:
        print("  Training GRU_Attention (with self-attention)...")

    model_attn = TorchRNNRegressor(
        hidden_dim=RNN_HIDDEN_DIM,
        n_epochs=RNN_N_EPOCHS,
        batch_size=32,
        learning_rate=0.001,
        use_attention=True,
        attention_heads=4,
        verbose=False
    )
    model_attn.fit(X_base, y_base)
    advanced_models['GRU_Attention'] = {
        'model': model_attn,
        'needs_derivative': False,
        'history_mode': 'none'
    }

    if verbose:
        print("  Done training 5 advanced GRU variants.")

    return advanced_models


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(cellType: int, results_dir: str = 'result_plots', verbose: bool = True):
    """
    Run integrated GLM + RNN pipeline for a single neuron type.

    Args:
        cellType: Izhikevich neuron type
        results_dir: Base directory for results
        verbose: Print progress
    """
    neuron_name = NEURON_TYPES.get(cellType, f'type_{cellType}')
    cid = cids[cellType - 1]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {neuron_name} (type {cellType})")
        print('='*60)

    # Create output directory
    output_dir = os.path.join(results_dir, neuron_name)
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================================================
    # STEP 1: Generate stimulus and simulate Izhikevich neuron
    # ==========================================================================
    T = 10000
    if cellType in [7, 8]:
        T = 20000

    if verbose:
        print(f"\n[1/7] Generating stimulus (T={T}ms)...")

    I, dt = generate_izhikevich_stim(cellType, T)
    if I is None:
        print(f"  Skipping {neuron_name}: subthreshold behavior")
        return None

    v, u, spikes, _ = simulate_izhikevich(cellType, I, dt, jitter=0, plotFlag=0, saveFlag=0, fid='')
    n_spikes = int(np.sum(spikes))

    if verbose:
        print(f"  Generated {n_spikes} spikes")

    # ==========================================================================
    # STEP 2: Fit GLM
    # ==========================================================================
    if verbose:
        print(f"\n[2/7] Fitting GLM (maxIter={GLM_MAX_ITER})...")

    nkt = 100
    kbasprs = {'neye': 0, 'ncos': 7, 'kpeaks': [0.1, round(nkt / 1.2)], 'b': 10}
    ihbasprs = {'ncols': 7, 'hpeaks': [0.1, 100], 'b': 10, 'absref': 1}
    softRect = 0

    try:
        k, h, dc, prs, kbasis, hbasis = fit_glm(
            I, spikes, dt, nkt, kbasprs, ihbasprs, None,
            softRect, plotFlag=0, maxIter=GLM_MAX_ITER, tolFun=GLM_TOL_FUN, L2pen=GLM_L2_PEN
        )
    except Exception as e:
        print(f"  Error fitting GLM: {e}")
        return None

    if verbose:
        print(f"  GLM fit complete. DC offset: {dc:.4f}")

    # ==========================================================================
    # STEP 3: Generate RNN training data and train BASE models
    # ==========================================================================
    if verbose:
        print(f"\n[3/7] Training BASE RNN models ({RNN_N_TRIALS} trials, {RNN_N_EPOCHS} epochs)...")

    generator = RNNDataGenerator(
        bin_size_ms=RNN_BIN_SIZE_MS,
        trial_duration_ms=250.0,
        seed=42
    )

    try:
        X_train, y_train = generator.generate_training_data(cellType, n_trials=RNN_N_TRIALS, verbose=verbose)
    except Exception as e:
        print(f"  Error generating RNN training data: {e}")
        return None

    # Save training data plot
    training_data_plot = os.path.join(output_dir, 'rnn_training_data.png')
    plot_training_data(X_train, y_train, training_data_plot, n_examples=5,
                       title=f'{neuron_name} - RNN Training Data ({RNN_N_TRIALS} trials)')

    # Train base RNN models
    rnn_models = train_base_rnn_models(X_train, y_train, verbose=verbose)

    # Collect base model losses
    train_losses_dict = {
        'GRU': rnn_models['GRU'].train_losses,
        'Vanilla RNN': rnn_models['Vanilla RNN'].train_losses,
        'LSTM': rnn_models['LSTM'].train_losses,
    }

    # Plot base training losses
    losses_plot = os.path.join(output_dir, 'rnn_training_losses.png')
    plot_training_losses(train_losses_dict, losses_plot)

    # ==========================================================================
    # STEP 4: Train EXTENDED RNN models (if enabled)
    # ==========================================================================
    extended_models = {}
    all_train_losses = dict(train_losses_dict)  # Copy base losses

    if TRAIN_EXTENDED_MODELS:
        if verbose:
            print(f"\n[4/7] Training EXTENDED RNN models...")

        try:
            extended_models, X_ext, y_ext = train_extended_rnn_models(generator, cellType, verbose=verbose)

            # Add extended model losses
            for model_name, model in extended_models.items():
                all_train_losses[model_name] = model.train_losses

            # Generate smoothed targets for visualization
            y_smoothed = generator.smooth_spikes(y_train, sigma_ms=RNN_SMOOTH_SIGMA_MS)

            # Save smoothed training data plot
            smoothed_data_plot = os.path.join(output_dir, 'rnn_training_data_smoothed.png')
            plot_smoothed_training_data(X_train, y_smoothed, smoothed_data_plot, n_examples=5,
                                        title=f'{neuron_name} - RNN Training Data (Smoothed, σ={RNN_SMOOTH_SIGMA_MS}ms)')

            # Plot extended training losses
            extended_losses_plot = os.path.join(output_dir, 'rnn_training_losses_extended.png')
            plot_training_losses_extended(all_train_losses, extended_losses_plot)

        except Exception as e:
            print(f"  Error training extended models: {e}")
            import traceback
            traceback.print_exc()
    else:
        if verbose:
            print(f"\n[4/7] Skipping extended RNN models (disabled)")

    # ==========================================================================
    # STEP 5: Train ADVANCED GRU models (if enabled)
    # ==========================================================================
    advanced_models = {}

    if TRAIN_ADVANCED_MODELS:
        if verbose:
            print(f"\n[5/7] Training ADVANCED GRU models...")

        try:
            advanced_models = train_advanced_gru_models(
                generator, cellType, X_train, y_train, verbose=verbose
            )

            # Add advanced model losses to all_train_losses
            for model_name, model_info in advanced_models.items():
                all_train_losses[model_name] = model_info['model'].train_losses

        except Exception as e:
            print(f"  Error training advanced models: {e}")
            import traceback
            traceback.print_exc()
    else:
        if verbose:
            print(f"\n[5/7] Skipping advanced GRU models (disabled)")

    # ==========================================================================
    # STEP 6: Evaluate RNNs on EXACT same stimulus as GLM
    # ==========================================================================
    if verbose:
        print(f"\n[6/7] Evaluating RNNs on GLM stimulus...")

    # Bin the GLM stimulus for RNN evaluation
    samples_per_bin = int(RNN_BIN_SIZE_MS / dt)
    n_bins = len(I) // samples_per_bin
    n_samples = n_bins * samples_per_bin

    I_trimmed = I[:n_samples]
    I_binned = I_trimmed.reshape(n_bins, samples_per_bin).mean(axis=1)

    # Get binned spikes for history features
    spike_times = np.where(spikes[:n_samples])[0] * dt
    bin_edges = np.arange(0, n_bins * RNN_BIN_SIZE_MS + RNN_BIN_SIZE_MS, RNN_BIN_SIZE_MS)
    y_binned, _ = np.histogram(spike_times, bins=bin_edges)
    y_binned = y_binned[:n_bins].astype(np.float64)

    # Get base RNN predictions
    X_test = generator.prepare_rnn_input(I_binned)
    rnn_predictions = {}
    for model_name, model in rnn_models.items():
        pred = model.predict_rate(X_test, return_sequence=True)
        rnn_predictions[model_name] = pred[0]  # Remove batch dimension

    # Get extended RNN predictions (for raster comparison)
    all_predictions = dict(rnn_predictions)  # Copy base predictions

    if TRAIN_EXTENDED_MODELS and extended_models:
        # Prepare inputs with different history features
        X_test_spike_hist = generator.prepare_rnn_input(I_binned, y_binned, history_mode='spike')
        X_test_input_hist = generator.prepare_rnn_input(I_binned, y_binned, history_mode='input')
        X_test_full_hist = generator.prepare_rnn_input(I_binned, y_binned, history_mode='full')

        for model_name, model in extended_models.items():
            # Select appropriate input based on model type
            if 'Smoothed' in model_name:
                X_eval = X_test  # Smoothed models use base input
            elif 'Spike Hist' in model_name:
                X_eval = X_test_spike_hist
            elif 'Input Hist' in model_name:
                X_eval = X_test_input_hist
            elif 'Full Hist' in model_name:
                X_eval = X_test_full_hist
            else:
                X_eval = X_test

            pred = model.predict_rate(X_eval, return_sequence=True)
            all_predictions[model_name] = pred[0]

    # Get advanced GRU model predictions
    if TRAIN_ADVANCED_MODELS and advanced_models:
        # Prepare special inputs for advanced models
        X_test_deriv = generator.prepare_rnn_input(I_binned, y_binned, add_derivative=True)
        X_test_spike_hist = generator.prepare_rnn_input(I_binned, y_binned, history_mode='spike')

        for model_name, model_info in advanced_models.items():
            model = model_info['model']
            needs_deriv = model_info['needs_derivative']
            hist_mode = model_info['history_mode']

            # Select appropriate input
            if needs_deriv:
                X_eval = X_test_deriv
            elif hist_mode == 'spike':
                X_eval = X_test_spike_hist
            else:
                X_eval = X_test

            pred = model.predict_rate(X_eval, return_sequence=True)
            all_predictions[model_name] = pred[0]

    if verbose:
        print(f"    Binned stimulus to {n_bins} bins @ {RNN_BIN_SIZE_MS}ms for RNN evaluation")

    # ==========================================================================
    # STEP 7: Create plots
    # ==========================================================================
    if verbose:
        print(f"\n[7/7] Creating comparison plots...")

    # Main comparison plot (with base RNNs only)
    plot_comparison(
        cellType, I, spikes, v, dt, k, h, dc,
        rnn_models, rnn_predictions, RNN_BIN_SIZE_MS,
        soft_rect=softRect, save_dir=output_dir
    )

    # Extended raster comparison (with all models including advanced)
    # Show this plot if we have any extended or advanced models
    if (TRAIN_EXTENDED_MODELS and extended_models) or (TRAIN_ADVANCED_MODELS and advanced_models):
        cell_name = index_to_name.get(cellType, f"Cell {cellType}")
        raster_plot = os.path.join(output_dir, 'spike_raster_comparison.png')
        plot_spike_raster_comparison(
            I_binned, y_binned, all_predictions, RNN_BIN_SIZE_MS,
            raster_plot, cell_name, cid
        )

        # Also plot the combined training losses
        combined_losses_plot = os.path.join(output_dir, 'rnn_training_losses_all.png')
        plot_training_losses_extended(all_train_losses, combined_losses_plot)

    # ==========================================================================
    # Save results
    # ==========================================================================
    np.savez(os.path.join(output_dir, 'results.npz'),
             cellType=cellType,
             neuron_name=neuron_name,
             cid=cid,
             dt=dt,
             T=T,
             I=I,
             v=v,
             u=u,
             spikes=spikes,
             n_spikes=n_spikes,
             k=k,
             h=h,
             dc=dc,
             prs=prs,
             kbasis=kbasis,
             hbasis=hbasis,
             kbasprs=kbasprs,
             ihbasprs=ihbasprs,
             softRect=softRect,
             rnn_n_trials=RNN_N_TRIALS,
             rnn_n_epochs=RNN_N_EPOCHS,
             rnn_train_losses=train_losses_dict,
             train_extended_models=TRAIN_EXTENDED_MODELS)

    if verbose:
        print(f"\n  Done! Results saved to: {output_dir}")

    return {
        'cellType': cellType,
        'neuron_name': neuron_name,
        'n_spikes': n_spikes,
        'dc': dc,
        'output_dir': output_dir,
        'rnn_final_losses': {name: losses[-1] for name, losses in train_losses_dict.items()},
        'n_extended_models': len(extended_models) if extended_models else 0
    }


def run_all(results_dir: str = 'result_plots'):
    """Run pipeline for all available neuron types."""
    print(f"\nRunning integrated GLM + RNN pipeline for all neuron types")
    print(f"Results will be saved to: {results_dir}")

    results = []
    failed = []

    for cellType in sorted(NEURON_TYPES.keys()):
        try:
            result = run_pipeline(cellType, results_dir=results_dir, verbose=True)
            if result is not None:
                results.append(result)
            else:
                failed.append(cellType)
        except Exception as e:
            print(f"\nError processing {NEURON_TYPES[cellType]}: {e}")
            failed.append(cellType)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Successfully processed: {len(results)}/{len(NEURON_TYPES)} neuron types")

    if results:
        print("\nResults:")
        print(f"{'Type':<5} {'Name':<30} {'Spikes':<10} {'DC':<10}")
        print('-'*55)
        for r in results:
            print(f"{r['cellType']:<5} {r['neuron_name']:<30} {r['n_spikes']:<10} {r['dc']:<10.4f}")

    if failed:
        print(f"\nFailed/Skipped: {[NEURON_TYPES.get(c, c) for c in failed]}")

    print(f"\nAll results saved to: {results_dir}")

    return results


if __name__ == '__main__':
    results_dir = 'result_plots'

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'all':
            # Explicit 'all' argument
            run_all(results_dir)
        elif len(sys.argv) > 2:
            # Multiple neuron types: python run_all.py 1 2 3 4
            cell_types = [int(arg) for arg in sys.argv[1:]]
            print(f"\nRunning pipeline for neuron types: {cell_types}")
            print(f"Results will be saved to: {results_dir}")

            results = []
            failed = []
            for cellType in cell_types:
                if cellType not in NEURON_TYPES:
                    print(f"\nWarning: Unknown neuron type {cellType}, skipping")
                    failed.append(cellType)
                    continue
                try:
                    result = run_pipeline(cellType, results_dir=results_dir, verbose=True)
                    if result is not None:
                        results.append(result)
                    else:
                        failed.append(cellType)
                except Exception as e:
                    print(f"\nError processing type {cellType}: {e}")
                    failed.append(cellType)

            # Summary
            print(f"\n{'='*60}")
            print("SUMMARY")
            print('='*60)
            print(f"Successfully processed: {len(results)}/{len(cell_types)} neuron types")
            if failed:
                print(f"Failed/Skipped: {failed}")
        else:
            # Single neuron type: python run_all.py 4
            cellType = int(sys.argv[1])
            run_pipeline(cellType, results_dir)
    else:
        # Default: run all neuron types
        run_all(results_dir)
