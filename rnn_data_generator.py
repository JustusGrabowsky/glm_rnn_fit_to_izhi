"""
rnn_data_generator.py - Generate training data for RNN models

This module generates trial-based training data for RNNs from Izhikevich neurons.
Each trial is a short sequence (~250ms) with:
- Variable silence-step-silence pattern
- Amplitude jitter for data augmentation
- Optional OU noise for generalization
- Binned at coarse resolution (1.0ms) for RNN input

Extended features:
- Gaussian smoothing of spike targets
- History-dependent input augmentation (spike history, input history)
- Derivative feature augmentation (rate of change of current)
"""

import numpy as np
from typing import Tuple, List, Optional, Literal
from scipy.ndimage import gaussian_filter1d

from generate_izhikevich_stim import generate_izhikevich_stim
from simulate_izhikevich import simulate_izhikevich


# Type alias for history modes
HistoryMode = Literal['none', 'spike', 'input', 'full']


class RNNDataGenerator:
    """
    Generate trial-based training data for RNN models.

    Creates multiple short trials with variable timing and amplitude,
    binned at coarse resolution for RNN training.

    Extended features:
    - OU noise injection for generalization
    - Gaussian smoothing of spike targets
    - History-dependent input features
    - Derivative feature augmentation (rate of change)
    """

    def __init__(
        self,
        bin_size_ms: float = 1.0,
        trial_duration_ms: float = 250.0,
        seed: Optional[int] = None
    ):
        """
        Args:
            bin_size_ms: Bin size for RNN input (default: 1.0ms)
            trial_duration_ms: Duration of each trial in ms
            seed: Random seed for reproducibility
        """
        self.bin_size_ms = bin_size_ms
        self.trial_duration_ms = trial_duration_ms
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def get_default_amplitude(self, cellType: int) -> float:
        """Get default stimulus amplitude for each cell type."""
        # From generate_izhikevich_stim.py
        amplitudes = {
            1: 14,      # tonic spiking
            2: 0.5,     # phasic spiking
            3: 10,      # tonic bursting
            4: 0.6,     # phasic bursting
            5: 10,      # mixed mode
            6: 20,      # spike frequency adaptation
            7: 25,      # Class 1
            8: 0.5,     # Class 2
            9: 3.49,    # spike latency
            11: 0.3,    # resonator
            12: 27.4,   # integrator
            13: -5,     # rebound spike
            14: -5,     # rebound burst
            15: 2.3,    # threshold variability
            16: 26.1,   # bistability
            18: 20,     # accomodation
            19: 70,     # inhibition-induced spiking
            20: 70,     # inhibition-induced bursting
            21: 26.1,   # bistability 2
        }
        return amplitudes.get(cellType, 10.0)

    def get_dt(self, cellType: int) -> float:
        """Get simulation timestep for each cell type."""
        # From generate_izhikevich_stim.py
        dts = {
            1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1,
            7: 0.1, 8: 0.1, 9: 0.1, 11: 0.5, 12: 0.5,
            13: 0.1, 14: 0.1, 15: 1.0, 16: 0.05, 18: 0.1,
            19: 0.1, 20: 0.1, 21: 0.05
        }
        return dts.get(cellType, 0.1)

    def generate_ou_noise(
        self,
        n_samples: int,
        dt: float,
        tau: float = 10.0,
        sigma: float = 1.0
    ) -> np.ndarray:
        """
        Generate Ornstein-Uhlenbeck (colored) noise.

        Args:
            n_samples: Number of samples to generate
            dt: Time step in ms
            tau: Time constant of the OU process (ms)
            sigma: Standard deviation of the noise

        Returns:
            noise: OU noise array of shape (n_samples,)
        """
        noise = np.zeros(n_samples)
        # OU process: dx = -x/tau * dt + sigma * sqrt(2*dt/tau) * dW
        decay = np.exp(-dt / tau)
        noise_scale = sigma * np.sqrt(1 - decay**2)

        for i in range(1, n_samples):
            noise[i] = decay * noise[i-1] + noise_scale * np.random.randn()

        return noise

    def generate_single_trial(
        self,
        cellType: int,
        amplitude: Optional[float] = None,
        silence_pre_ms: float = 50.0,
        step_duration_ms: float = 150.0,
        jitter_timing: bool = True,
        jitter_amplitude: bool = True,
        add_ou_noise: bool = False,
        ou_tau: float = 10.0,
        ou_sigma_fraction: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate a single trial with step stimulus.

        Args:
            cellType: Izhikevich neuron type
            amplitude: Stimulus amplitude (None for default)
            silence_pre_ms: Pre-stimulus silence duration
            step_duration_ms: Step stimulus duration
            jitter_timing: Add timing jitter
            jitter_amplitude: Add amplitude jitter
            add_ou_noise: Add Ornstein-Uhlenbeck noise to stimulus
            ou_tau: OU noise time constant (ms)
            ou_sigma_fraction: OU noise std as fraction of amplitude

        Returns:
            I: Stimulus current
            spikes: Spike times in ms
            dt: Simulation timestep
        """
        dt = self.get_dt(cellType)

        if amplitude is None:
            amplitude = self.get_default_amplitude(cellType)

        # Apply timing jitter
        if jitter_timing:
            silence_pre = silence_pre_ms + np.random.uniform(-10, 10)
            step_dur = step_duration_ms * np.random.uniform(0.95, 1.05)
        else:
            silence_pre = silence_pre_ms
            step_dur = step_duration_ms

        silence_post = self.trial_duration_ms - silence_pre - step_dur

        # Apply amplitude jitter
        if jitter_amplitude:
            amp_scale = np.random.uniform(0.95, 1.05)
            amp_noise = np.random.normal(0, 0.05 * abs(amplitude))
            amp = amplitude * amp_scale + amp_noise
        else:
            amp = amplitude

        # Generate stimulus
        n_samples = int(self.trial_duration_ms / dt)
        I = np.zeros(n_samples)

        start_idx = int(silence_pre / dt)
        end_idx = int((silence_pre + step_dur) / dt)
        I[start_idx:end_idx] = amp

        # Add OU noise if requested
        if add_ou_noise:
            ou_sigma = ou_sigma_fraction * abs(amplitude)
            ou_noise = self.generate_ou_noise(n_samples, dt, tau=ou_tau, sigma=ou_sigma)
            I = I + ou_noise

        # Simulate neuron
        v, u, spikes_binary, cid = simulate_izhikevich(
            cellType, I, dt, jitter=0, plotFlag=0, saveFlag=0, fid=''
        )

        # Convert to spike times
        spike_indices = np.where(spikes_binary)[0]
        spike_times = spike_indices * dt

        return I, spike_times, dt

    def bin_trial(
        self,
        I: np.ndarray,
        spike_times: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin stimulus and spike times to RNN resolution.

        Args:
            I: Stimulus at simulation resolution
            spike_times: Spike times in ms
            dt: Simulation timestep

        Returns:
            I_binned: Binned stimulus
            y_binned: Spike counts per bin
        """
        samples_per_bin = int(self.bin_size_ms / dt)
        n_bins = int(self.trial_duration_ms / self.bin_size_ms)

        # Trim and reshape
        n_samples = n_bins * samples_per_bin
        I_trimmed = I[:n_samples]

        # Average stimulus over bins
        I_binned = I_trimmed.reshape(n_bins, samples_per_bin).mean(axis=1)

        # Count spikes per bin
        bin_edges = np.arange(0, self.trial_duration_ms + self.bin_size_ms, self.bin_size_ms)
        y_binned, _ = np.histogram(spike_times, bins=bin_edges)
        y_binned = y_binned[:n_bins].astype(np.float64)

        return I_binned, y_binned

    def smooth_spikes(
        self,
        y: np.ndarray,
        sigma_ms: float = 3.0
    ) -> np.ndarray:
        """
        Apply Gaussian smoothing to spike counts.

        Args:
            y: Spike counts array, shape (n_trials, seq_len) or (seq_len,)
            sigma_ms: Gaussian kernel standard deviation in ms

        Returns:
            y_smoothed: Smoothed spike counts (same shape as input)
        """
        # Convert sigma from ms to bins
        sigma_bins = sigma_ms / self.bin_size_ms

        if y.ndim == 1:
            return gaussian_filter1d(y, sigma=sigma_bins, mode='reflect')
        else:
            # Apply along time axis (axis=1)
            return gaussian_filter1d(y, sigma=sigma_bins, axis=1, mode='reflect')

    def add_derivative_feature(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Add derivative (rate of change) of the current as a new feature channel.

        Computes ΔI = I_t - I_{t-1} for each input channel and concatenates.
        The first timestep derivative is set to 0.

        Args:
            X: Input array, shape (n_trials, seq_len, n_features) or (1, seq_len, n_features)

        Returns:
            X_with_derivative: Augmented input with derivative channels
                              Shape: (n_trials, seq_len, n_features * 2)
        """
        # Compute discrete derivative: dI/dt ≈ I_t - I_{t-1}
        derivative = np.zeros_like(X)
        derivative[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
        # First timestep derivative is 0 (no previous value)

        # Concatenate original and derivative
        X_with_derivative = np.concatenate([X, derivative], axis=2)

        return X_with_derivative

    def add_history_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mode: HistoryMode = 'none'
    ) -> np.ndarray:
        """
        Augment input X with history features.

        Args:
            X: Input array, shape (n_trials, seq_len, n_features)
            y: Target spike counts, shape (n_trials, seq_len)
            mode: History mode - 'none', 'spike', 'input', or 'full'
                - 'none': Return X unchanged [I_t]
                - 'spike': Add lagged spike history [I_t, S_{t-1}]
                - 'input': Add lagged input history [I_t, I_{t-1}]
                - 'full': Add both [I_t, I_{t-1}, S_{t-1}]

        Returns:
            X_augmented: Augmented input array
        """
        if mode == 'none':
            return X

        n_trials, seq_len, n_features = X.shape

        # Create lagged versions (shift right, pad with zeros)
        if mode == 'spike':
            # [I_t, S_{t-1}]
            y_lagged = np.zeros_like(y)
            y_lagged[:, 1:] = y[:, :-1]
            X_augmented = np.concatenate([X, y_lagged[:, :, np.newaxis]], axis=2)

        elif mode == 'input':
            # [I_t, I_{t-1}]
            I_lagged = np.zeros((n_trials, seq_len, n_features))
            I_lagged[:, 1:, :] = X[:, :-1, :]
            X_augmented = np.concatenate([X, I_lagged], axis=2)

        elif mode == 'full':
            # [I_t, I_{t-1}, S_{t-1}]
            I_lagged = np.zeros((n_trials, seq_len, n_features))
            I_lagged[:, 1:, :] = X[:, :-1, :]
            y_lagged = np.zeros_like(y)
            y_lagged[:, 1:] = y[:, :-1]
            X_augmented = np.concatenate([X, I_lagged, y_lagged[:, :, np.newaxis]], axis=2)
        else:
            raise ValueError(f"Unknown history mode: {mode}")

        return X_augmented

    def generate_training_data(
        self,
        cellType: int,
        n_trials: int = 500,
        verbose: bool = True,
        add_ou_noise: bool = False,
        ou_tau: float = 10.0,
        ou_sigma_fraction: float = 0.1,
        add_derivative: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multiple trials for RNN training.

        Args:
            cellType: Izhikevich neuron type
            n_trials: Number of trials to generate
            verbose: Print progress
            add_ou_noise: Add OU noise to stimuli
            ou_tau: OU noise time constant (ms)
            ou_sigma_fraction: OU noise std as fraction of amplitude
            add_derivative: If True, add derivative (dI/dt) as additional feature channel

        Returns:
            X: Input sequences (n_trials, seq_len, n_features)
               n_features = 1 if add_derivative=False, else 2
            y: Target spike counts (n_trials, seq_len)
        """
        if verbose:
            print(f"  Generating {n_trials} training trials...")

        X_trials = []
        y_trials = []

        for i in range(n_trials):
            I, spike_times, dt = self.generate_single_trial(
                cellType,
                add_ou_noise=add_ou_noise,
                ou_tau=ou_tau,
                ou_sigma_fraction=ou_sigma_fraction
            )
            I_binned, y_binned = self.bin_trial(I, spike_times, dt)

            X_trials.append(I_binned)
            y_trials.append(y_binned)

        # Stack into arrays
        X = np.array(X_trials)[:, :, np.newaxis]  # (n_trials, seq_len, 1)
        y = np.array(y_trials)  # (n_trials, seq_len)

        # Add derivative feature if requested
        if add_derivative:
            X = self.add_derivative_feature(X)
            if verbose:
                print(f"  Added derivative feature: X shape now {X.shape}")

        if verbose:
            print(f"  X shape: {X.shape}, y shape: {y.shape}")
            print(f"  Total spikes: {y.sum():.0f}, Mean per trial: {y.sum() / n_trials:.1f}")

        return X, y

    def generate_training_data_extended(
        self,
        cellType: int,
        n_trials: int = 500,
        verbose: bool = True,
        add_ou_noise: bool = False,
        ou_tau: float = 10.0,
        ou_sigma_fraction: float = 0.1,
        smooth_sigma_ms: float = 3.0,
        history_mode: HistoryMode = 'none'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training data with extended features.

        Args:
            cellType: Izhikevich neuron type
            n_trials: Number of trials
            verbose: Print progress
            add_ou_noise: Add OU noise
            ou_tau: OU noise time constant
            ou_sigma_fraction: OU noise fraction
            smooth_sigma_ms: Gaussian smoothing sigma for targets
            history_mode: History feature mode

        Returns:
            X: Input sequences (possibly augmented with history)
            y: Raw spike counts (for standard training)
            y_smoothed: Smoothed spike counts (for smoothed training)
        """
        # Generate base data
        X, y = self.generate_training_data(
            cellType, n_trials, verbose,
            add_ou_noise, ou_tau, ou_sigma_fraction
        )

        # Smooth targets
        y_smoothed = self.smooth_spikes(y, sigma_ms=smooth_sigma_ms)

        # Add history features
        X = self.add_history_features(X, y, mode=history_mode)

        if verbose and history_mode != 'none':
            print(f"  Augmented X shape: {X.shape} (history_mode='{history_mode}')")

        return X, y, y_smoothed

    def generate_test_sequence(
        self,
        cellType: int,
        T_ms: float = 1000.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Generate a test sequence matching GLM evaluation style.

        Uses the same step pattern as GLM training data.

        Args:
            cellType: Izhikevich neuron type
            T_ms: Total duration in ms

        Returns:
            I: Raw stimulus at simulation resolution
            spikes: Binary spike array
            I_binned: Binned stimulus for RNN
            y_binned: Binned spike counts
            dt: Simulation timestep
        """
        # Generate stimulus using standard GLM pattern
        I, dt = generate_izhikevich_stim(cellType, T_ms)

        if I is None:
            raise ValueError(f"Cannot generate stimulus for cellType {cellType}")

        # Simulate
        v, u, spikes, cid = simulate_izhikevich(
            cellType, I, dt, jitter=0, plotFlag=0, saveFlag=0, fid=''
        )

        # Bin for RNN
        samples_per_bin = int(self.bin_size_ms / dt)
        n_bins = len(I) // samples_per_bin
        n_samples = n_bins * samples_per_bin

        I_trimmed = I[:n_samples]
        I_binned = I_trimmed.reshape(n_bins, samples_per_bin).mean(axis=1)

        spike_times = np.where(spikes[:n_samples])[0] * dt
        bin_edges = np.arange(0, n_bins * self.bin_size_ms + self.bin_size_ms, self.bin_size_ms)
        y_binned, _ = np.histogram(spike_times, bins=bin_edges)
        y_binned = y_binned[:n_bins].astype(np.float64)

        return I, spikes, I_binned, y_binned, dt

    def prepare_rnn_input(
        self,
        I_binned: np.ndarray,
        y_binned: Optional[np.ndarray] = None,
        history_mode: HistoryMode = 'none',
        add_derivative: bool = False
    ) -> np.ndarray:
        """
        Prepare binned stimulus as RNN input (single sequence).

        Args:
            I_binned: Binned stimulus (seq_len,)
            y_binned: Binned spike counts for history features (seq_len,)
            history_mode: History feature mode
            add_derivative: If True, add derivative (dI/dt) as additional feature channel

        Returns:
            X: RNN input (1, seq_len, n_features)
        """
        X = I_binned.reshape(1, -1, 1)

        # Add derivative feature if requested (before history features)
        if add_derivative:
            X = self.add_derivative_feature(X)

        if history_mode != 'none' and y_binned is not None:
            # Create dummy y for history feature extraction
            y = y_binned.reshape(1, -1)
            X = self.add_history_features(X, y, mode=history_mode)

        return X
