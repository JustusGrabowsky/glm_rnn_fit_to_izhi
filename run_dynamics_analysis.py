"""Runner script for dynamical-system analysis of trained RNN models.

Supports arbitrary cell types and saves results to result_plots/<neuron_type>/.

Pipeline
--------
1. Generate Izhikevich neuron data for specified cell type(s).
2. Build RNN training set via RNNDataGenerator.
3. Train a GRU regressor.
4. Search for fixed points at rest (I = 0) and under stimulation.
5. Report readout rates and stability (max |eigenvalue|).
6. Visualise the PCA-projected phase space and save to result_plots.

Usage:
    python run_dynamics_analysis.py           # Run all cell types
    python run_dynamics_analysis.py 4         # Run single cell type
    python run_dynamics_analysis.py 1 2 3 4   # Run multiple specific types
"""

import os
import sys
import argparse
import numpy as np

# Ensure this directory is on the path so local imports work when the
# script is invoked from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generate_izhikevich_stim import generate_izhikevich_stim
from simulate_izhikevich import simulate_izhikevich
from rnn_data_generator import RNNDataGenerator
from rnn_models import TorchRNNRegressor
from rnn_dynamics import RNNSystemAnalyzer
from izhikevich_configs import cids, index_to_name


# Cell type to directory name mapping (from run_all.py)
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


def analyze_cell_type(cell_type, n_trials=1000, hidden_dim=64, n_epochs=5,
                      batch_size=32, bin_size_ms=1.0, trial_duration_ms=250.0,
                      seed=42, fp_restarts=50):
    """Run dynamics analysis for a single cell type.

    Parameters
    ----------
    cell_type : int
        Izhikevich cell type (1-21, except 10, 17).
    n_trials : int
        Number of RNN training trials.
    hidden_dim : int
        GRU hidden state dimension.
    n_epochs : int
        Training epochs.
    batch_size : int
        Training batch size.
    bin_size_ms : float
        RNN input bin size in ms.
    trial_duration_ms : float
        Training trial duration in ms.
    seed : int
        Random seed.
    fp_restarts : int
        Number of fixed-point search restarts.

    Returns
    -------
    dict
        Analysis results including fixed points and trajectory.
    """
    if cell_type not in NEURON_TYPES:
        print(f"Warning: Cell type {cell_type} not in NEURON_TYPES mapping")
        neuron_dir = f"type_{cell_type}"
    else:
        neuron_dir = NEURON_TYPES[cell_type]

    neuron_name = index_to_name.get(cell_type, f"Type {cell_type}")
    cid = cids[cell_type - 1] if cell_type <= len(cids) else "?"

    print("\n" + "=" * 70)
    print(f"Dynamical System Analysis — Cell Type {cell_type}: {neuron_name} ({cid})")
    print("=" * 70)

    # ---- 1. Izhikevich simulation ----
    print("\n[1] Generating Izhikevich stimulus & simulating neuron ...")
    I_full, dt = generate_izhikevich_stim(cell_type, T=10000)
    if I_full is None:
        print(f"    ERROR: Cell type {cell_type} not available (likely subthreshold)")
        return None

    v, u, spikes, cid = simulate_izhikevich(
        cell_type, I_full, dt, plotFlag=0, saveFlag=0,
    )
    print(f"    Stimulus samples : {len(I_full)}, dt = {dt} ms")
    print(f"    Total spikes     : {int(spikes.sum())}")

    # ---- 2. RNN training data ----
    print("\n[2] Building RNN training set ...")
    gen = RNNDataGenerator(
        bin_size_ms=bin_size_ms,
        trial_duration_ms=trial_duration_ms,
        seed=seed,
    )
    X_train, y_train = gen.generate_training_data(
        cell_type, n_trials=n_trials, verbose=False,
    )
    print(f"    X_train : {X_train.shape}")
    print(f"    y_train : {y_train.shape}")

    # ---- 3. Train GRU ----
    print("\n[3] Training GRU model ...")
    model = TorchRNNRegressor(
        hidden_dim=hidden_dim,
        num_layers=1,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=False,
    )
    model.fit(X_train, y_train)
    print(f"    Final training loss : {model.train_losses[-1]:.6f}")

    # ---- 4. Initialise analyser ----
    print("\n[4] Initialising RNNSystemAnalyzer ...")
    analyzer = RNNSystemAnalyzer(model)

    # ---- 5a. Fixed points at rest (I = 0) ----
    print("\n[5] Fixed-point search at I = 0 (resting) ...")
    fps_rest = analyzer.find_fixed_points(
        raw_input_val=0.0, n_restarts=fp_restarts, tol=1e-6,
    )
    for i, fp in enumerate(fps_rest):
        rate = analyzer.get_model_readout(fp)
        eigs = analyzer.compute_stability(fp, raw_input_val=0.0)
        max_eig = np.max(np.abs(eigs))
        tag = "stable" if max_eig < 1.0 else "UNSTABLE"
        print(f"    FP {i}: rate = {rate:.4f},  max|eig| = {max_eig:.4f}  "
              f"({tag})")

    # ---- 5b. Fixed points under stimulation ----
    # Use the mean positive stimulus from the training data as a
    # representative "on" level.
    pos_vals = X_train[X_train > 0]
    stim_level = float(pos_vals.mean()) if pos_vals.size > 0 else 0.6

    print(f"\n[6] Fixed-point search at I = {stim_level:.3f} "
          f"(stimulated) ...")
    fps_stim = analyzer.find_fixed_points(
        raw_input_val=stim_level, n_restarts=fp_restarts, tol=1e-6,
    )
    for i, fp in enumerate(fps_stim):
        rate = analyzer.get_model_readout(fp)
        eigs = analyzer.compute_stability(fp, raw_input_val=stim_level)
        max_eig = np.max(np.abs(eigs))
        tag = "stable" if max_eig < 1.0 else "UNSTABLE"
        print(f"    FP {i}: rate = {rate:.4f},  max|eig| = {max_eig:.4f}  "
              f"({tag})")

    # ---- 6. Extract a test trajectory ----
    print("\n[7] Extracting hidden trajectory from first training trial ...")
    test_input = X_train[0, :, 0]       # (seq_len,)
    trajectory = analyzer.get_trajectory(test_input)
    print(f"    Trajectory shape : {trajectory.shape}")

    # ---- 7. Phase-space visualisation ----
    print("\n[8] Plotting phase space ...")
    all_fps = fps_rest + fps_stim

    # Create result directory if needed
    result_dir = os.path.join(_HERE, "result_plots", neuron_dir)
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, "dynamics_analysis.png")

    analyzer.plot_phase_space(
        trajectory=trajectory,
        fixed_points=all_fps if all_fps else [],
        raw_input_val=0.0,
        title=f"Phase Space — Type {cell_type}: {neuron_name} ({cid})",
        save_path=save_path,
    )

    print("\n" + "=" * 70)
    print(f"Analysis complete for cell type {cell_type}")
    print(f"Results saved to: {result_dir}/")
    print("=" * 70)

    return {
        'cell_type': cell_type,
        'neuron_name': neuron_name,
        'cid': cid,
        'fps_rest': fps_rest,
        'fps_stim': fps_stim,
        'trajectory': trajectory,
        'stim_level': stim_level,
    }


def get_cell_types(args_cell_types):
    """Parse cell types from command-line arguments.

    Parameters
    ----------
    args_cell_types : list of str or None
        Command-line arguments.

    Returns
    -------
    list of int
        Cell type IDs to analyse.
    """
    if not args_cell_types:
        # Default: all available types
        return sorted(NEURON_TYPES.keys())

    cell_types = []
    for arg in args_cell_types:
        if arg.lower() == 'all':
            return sorted(NEURON_TYPES.keys())
        try:
            ct = int(arg)
            if ct in NEURON_TYPES or ct in [10, 17]:  # 10, 17 are unavailable
                cell_types.append(ct)
            else:
                print(f"Warning: Cell type {ct} is not a standard Izhikevich type")
                cell_types.append(ct)
        except ValueError:
            print(f"Warning: Could not parse '{arg}' as cell type")

    return cell_types if cell_types else sorted(NEURON_TYPES.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Dynamical system analysis for RNN models trained on "
                    "Izhikevich data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dynamics_analysis.py           # Analyse all cell types
  python run_dynamics_analysis.py 4         # Analyse cell type 4 only
  python run_dynamics_analysis.py 1 2 3 4   # Analyse specific types
  python run_dynamics_analysis.py all       # Explicit all cell types
        """)

    parser.add_argument(
        'cell_types',
        nargs='*',
        help='Cell type IDs to analyse (default: all available types)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=1000,
        help='Number of RNN training trials (default: 1000)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help='GRU hidden state dimension (default: 64)'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=5,
        help='Training epochs (default: 5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    parser.add_argument(
        '--bin-size-ms',
        type=float,
        default=1.0,
        help='RNN input bin size in ms (default: 1.0)'
    )
    parser.add_argument(
        '--trial-duration-ms',
        type=float,
        default=250.0,
        help='Training trial duration in ms (default: 250.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--fp-restarts',
        type=int,
        default=50,
        help='Number of fixed-point search restarts (default: 50)'
    )

    args = parser.parse_args()

    cell_types = get_cell_types(args.cell_types)

    print(f"\n{'#' * 70}")
    print(f"Dynamical System Analysis for RNN Models")
    print(f"Cell types to analyse: {cell_types}")
    print(f"{'#' * 70}")

    results = {}
    for ct in cell_types:
        result = analyze_cell_type(
            cell_type=ct,
            n_trials=args.n_trials,
            hidden_dim=args.hidden_dim,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            bin_size_ms=args.bin_size_ms,
            trial_duration_ms=args.trial_duration_ms,
            seed=args.seed,
            fp_restarts=args.fp_restarts,
        )
        if result is not None:
            results[ct] = result

    print(f"\n{'#' * 70}")
    print(f"All analyses complete!")
    print(f"Completed {len(results)} cell type(s)")
    print(f"Results saved to: result_plots/<neuron_type>/")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
