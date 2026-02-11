"""Dynamical system analysis module for trained RNN models.

Provides fixed-point search, Jacobian stability analysis, and PCA-projected
phase-space visualisation of RNN hidden-state dynamics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class RNNSystemAnalyzer:
    """Analyse the autonomous dynamics of a trained RNN regressor.

    Supports both GRU-based (TorchRNNRegressor) and vanilla-RNN-based
    (TorchVanillaRNNRegressor) models.  Bidirectional and attention
    architectures are flagged with a warning because their dynamics are
    not purely Markovian in the hidden state.

    Parameters
    ----------
    rnn_regressor : TorchRNNRegressor or TorchVanillaRNNRegressor
        A *fitted* regressor instance (must have ``_is_fitted == True``).
    """

    def __init__(self, rnn_regressor):
        if not getattr(rnn_regressor, '_is_fitted', False):
            raise ValueError("Regressor must be fitted before analysis.")

        self.regressor = rnn_regressor
        self.model = rnn_regressor.model
        self.device = rnn_regressor.device
        self.hidden_dim = rnn_regressor.hidden_dim
        self.num_layers = rnn_regressor.num_layers

        # Warnings for non-Markovian architectures
        if getattr(rnn_regressor, 'bidirectional', False):
            print("Warning: Bidirectional model detected. Hidden-state "
                  "dynamics are non-Markovian; fixed-point analysis may "
                  "not be meaningful.")
        if getattr(rnn_regressor, 'use_attention', False):
            print("Warning: Attention model detected. Hidden-state "
                  "dynamics are non-Markovian; fixed-point analysis may "
                  "not be meaningful.")

        # Detect cell type
        if hasattr(self.model, 'gru'):
            self.cell = self.model.gru
            self.cell_type = 'gru'
        elif hasattr(self.model, 'rnn'):
            self.cell = self.model.rnn
            self.cell_type = 'rnn'
        else:
            raise ValueError("Model architecture not recognised: expected a "
                             "'gru' or 'rnn' attribute on the network.")

        self.num_directions = (
            2 if getattr(self.cell, 'bidirectional', False) else 1
        )

        # Normalisation statistics (numpy arrays, shape (1, 1, n_features))
        self.X_mean = rnn_regressor.X_mean
        self.X_std = rnn_regressor.X_std
        self._input_size = int(self.X_mean.shape[-1])

        # Freeze model parameters — we only need forward passes and
        # gradients w.r.t. hidden states, never w.r.t. weights.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_scaled_input(self, raw_val):
        """Z-score normalise a raw scalar input.

        Returns a tensor of shape ``(1, 1, input_size)`` on ``self.device``.
        """
        mean = self.X_mean.flatten()
        std = self.X_std.flatten()

        if np.isscalar(raw_val):
            raw_arr = np.full(self._input_size, raw_val, dtype=np.float64)
        else:
            raw_arr = np.asarray(raw_val, dtype=np.float64).flatten()[
                :self._input_size
            ]

        scaled = (raw_arr - mean) / (std + 1e-8)
        x = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        return x.reshape(1, 1, self._input_size)

    def _step_torch(self, h, x_tensor):
        """One-step RNN map: h_{t+1} = f(h_t, x).

        Parameters
        ----------
        h : Tensor, shape ``(num_layers*num_directions, 1, hidden_dim)``
        x_tensor : Tensor, shape ``(1, 1, input_size)``

        Returns
        -------
        Tensor  — new hidden state, same shape as *h*.
        """
        _, h_new = self.cell(x_tensor, h)
        return h_new

    # ------------------------------------------------------------------
    # Fixed-point search
    # ------------------------------------------------------------------

    def find_fixed_points(self, raw_input_val=0.0, n_restarts=10,
                          tol=1e-6, lr=1.0, max_iter=200):
        """Find fixed points of the RNN dynamics at a constant input.

        Uses L-BFGS to minimise ``||f(h, x) - h||^2`` from random seeds.

        Parameters
        ----------
        raw_input_val : float
            Raw (un-normalised) input level.
        n_restarts : int
            Number of random initialisations.
        tol : float
            Loss threshold below which a point is accepted as a fixed point.
        lr : float
            L-BFGS learning rate.
        max_iter : int
            Maximum function evaluations per L-BFGS ``.step()`` call.

        Returns
        -------
        list of np.ndarray
            Unique fixed points, each of shape
            ``(num_layers * num_directions, hidden_dim)``.
        """
        x_tensor = self._get_scaled_input(raw_input_val)
        h_shape = (self.num_layers * self.num_directions, 1, self.hidden_dim)

        found_fps = []
        found_losses = []

        for i in range(n_restarts):
            h_init = torch.randn(*h_shape, device=self.device) * 0.5
            h_opt = h_init.clone().detach().requires_grad_(True)

            optimizer = torch.optim.LBFGS(
                [h_opt], lr=lr, max_iter=max_iter,
                tolerance_grad=1e-9, tolerance_change=1e-12,
                line_search_fn='strong_wolfe',
            )

            try:
                def closure():
                    optimizer.zero_grad()
                    h_new = self._step_torch(h_opt, x_tensor)
                    loss = ((h_new - h_opt) ** 2).sum()
                    loss.backward()
                    return loss

                optimizer.step(closure)

                with torch.no_grad():
                    h_new = self._step_torch(h_opt, x_tensor)
                    final_loss = ((h_new - h_opt) ** 2).sum().item()

                if final_loss < tol:
                    fp = h_opt.detach().cpu().numpy().reshape(
                        self.num_layers * self.num_directions, self.hidden_dim
                    )
                    found_fps.append(fp)
                    found_losses.append(final_loss)

            except Exception as exc:
                print(f"  Restart {i}: optimisation failed ({exc})")
                continue

        # De-duplicate
        unique_fps: list[np.ndarray] = []
        for fp in found_fps:
            if all(np.linalg.norm(fp - u) > 1e-3 for u in unique_fps):
                unique_fps.append(fp)

        print(f"Found {len(unique_fps)} unique fixed point(s) "
              f"(from {len(found_fps)} converged / {n_restarts} restarts)")
        return unique_fps

    # ------------------------------------------------------------------
    # Stability analysis
    # ------------------------------------------------------------------

    def compute_stability(self, h_fp, raw_input_val):
        """Compute eigenvalues of the recurrence Jacobian at a fixed point.

        Parameters
        ----------
        h_fp : np.ndarray
            Fixed point, shape ``(num_layers*num_directions, hidden_dim)``.
        raw_input_val : float
            Raw input level at which the fixed point was found.

        Returns
        -------
        np.ndarray
            Complex eigenvalues of the Jacobian ``df/dh`` evaluated at
            ``h_fp``.
        """
        x_tensor = self._get_scaled_input(raw_input_val)
        total_dim = self.num_layers * self.num_directions * self.hidden_dim
        h_flat = torch.tensor(
            h_fp.flatten(), dtype=torch.float32, device=self.device
        )

        def mapping(hf):
            h = hf.reshape(
                self.num_layers * self.num_directions, 1, self.hidden_dim
            )
            h_new = self._step_torch(h, x_tensor)
            return h_new.reshape(-1)

        J = torch.autograd.functional.jacobian(mapping, h_flat)
        J_np = J.detach().cpu().numpy()
        return np.linalg.eigvals(J_np)

    # ------------------------------------------------------------------
    # Readout
    # ------------------------------------------------------------------

    def get_model_readout(self, h_fp):
        """Firing-rate prediction at a given hidden state.

        Passes the last-layer hidden state through the model's
        ``fc`` → ``softplus`` readout.

        Parameters
        ----------
        h_fp : np.ndarray
            Shape ``(num_layers*num_directions, hidden_dim)``.

        Returns
        -------
        float
        """
        h_tensor = torch.tensor(
            h_fp, dtype=torch.float32, device=self.device
        )

        if self.num_directions == 2:
            h_last = torch.cat([h_tensor[-2], h_tensor[-1]], dim=-1)
        else:
            h_last = h_tensor[-1]

        with torch.no_grad():
            out = self.model.fc(h_last.unsqueeze(0))
            out = self.model.softplus(out)
        return out.item()

    # ------------------------------------------------------------------
    # Trajectory extraction
    # ------------------------------------------------------------------

    def get_trajectory(self, raw_input_sequence):
        """Run the model step-by-step and collect last-layer hidden states.

        Parameters
        ----------
        raw_input_sequence : np.ndarray
            Shape ``(seq_len,)`` or ``(seq_len, n_features)``; raw
            (un-normalised) values.

        Returns
        -------
        np.ndarray, shape ``(seq_len, hidden_dim)``
        """
        if raw_input_sequence.ndim == 1:
            raw_input_sequence = raw_input_sequence.reshape(-1, 1)
        seq_len = raw_input_sequence.shape[0]

        mean = self.X_mean.reshape(1, -1)
        std = self.X_std.reshape(1, -1)
        scaled = (raw_input_sequence - mean) / (std + 1e-8)

        x_tensor = torch.tensor(
            scaled, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, seq_len, n_features)

        h = torch.zeros(
            self.num_layers * self.num_directions, 1, self.hidden_dim,
            device=self.device,
        )

        hidden_states = []
        with torch.no_grad():
            for t in range(seq_len):
                x_t = x_tensor[:, t : t + 1, :]  # (1, 1, n_features)
                _, h = self.cell(x_t, h)
                hidden_states.append(h[-1, 0, :].cpu().numpy())

        return np.array(hidden_states)

    # ------------------------------------------------------------------
    # Phase-space visualisation
    # ------------------------------------------------------------------

    def plot_phase_space(self, trajectory, fixed_points, raw_input_val,
                         title="Phase Space", save_path=None, grid_n=20):
        """PCA-projected phase portrait with flow field and fixed points.

        Parameters
        ----------
        trajectory : np.ndarray
            Shape ``(seq_len, hidden_dim)`` — e.g. from :meth:`get_trajectory`.
        fixed_points : list of np.ndarray
            Each shape ``(num_layers*num_directions, hidden_dim)``.
        raw_input_val : float
            Input level used to compute the flow field.
        title : str
        save_path : str or None
            If given, saves the figure.
        grid_n : int
            Resolution of the flow-field grid.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # --- PCA on the trajectory ---
        pca = PCA(n_components=2)
        traj_pc = pca.fit_transform(trajectory)

        # --- Fixed-point projections and stability ---
        fp_last = [fp[-1] for fp in fixed_points]  # last-layer states
        fp_stable = []
        for fp in fixed_points:
            eigs = self.compute_stability(fp, raw_input_val)
            fp_stable.append(np.max(np.abs(eigs)) < 1.0)

        fp_pc = (
            pca.transform(np.array(fp_last)) if fp_last
            else np.empty((0, 2))
        )

        # --- Flow field via batched RNN step ---
        margin = 0.15
        xr = traj_pc[:, 0].ptp() or 1.0
        yr = traj_pc[:, 1].ptp() or 1.0
        x_lo, x_hi = traj_pc[:, 0].min() - margin * xr, traj_pc[:, 0].max() + margin * xr
        y_lo, y_hi = traj_pc[:, 1].min() - margin * yr, traj_pc[:, 1].max() + margin * yr

        gx = np.linspace(x_lo, x_hi, grid_n)
        gy = np.linspace(y_lo, y_hi, grid_n)
        GX, GY = np.meshgrid(gx, gy)

        pc_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
        h_hd = pca.inverse_transform(pc_pts)  # (N, hidden_dim)

        N = grid_n * grid_n
        h_batch = torch.zeros(
            self.num_layers * self.num_directions, N, self.hidden_dim,
            device=self.device,
        )
        h_batch[-1] = torch.tensor(h_hd, dtype=torch.float32,
                                   device=self.device)

        x_tensor = self._get_scaled_input(raw_input_val)
        x_batch = x_tensor.expand(N, -1, -1)  # (N, 1, input_size)

        with torch.no_grad():
            _, h_new_batch = self.cell(x_batch, h_batch)

        h_new_last = h_new_batch[-1].cpu().numpy()  # (N, hidden_dim)
        h_new_pc = pca.transform(h_new_last)

        flow = h_new_pc - pc_pts
        U = flow[:, 0].reshape(grid_n, grid_n)
        V = flow[:, 1].reshape(grid_n, grid_n)
        speed = np.sqrt(U ** 2 + V ** 2)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 7))

        strm = ax.streamplot(
            gx, gy, U, V,
            color=speed, cmap='coolwarm',
            density=1.2, linewidth=0.8, arrowsize=1.0,
        )
        fig.colorbar(strm.lines, ax=ax, label='|flow|')

        ax.plot(traj_pc[:, 0], traj_pc[:, 1],
                'k-', lw=1.5, alpha=0.7, label='Trajectory')
        ax.plot(traj_pc[0, 0], traj_pc[0, 1],
                'go', ms=10, zorder=5, label='Start')
        ax.plot(traj_pc[-1, 0], traj_pc[-1, 1],
                'rs', ms=10, zorder=5, label='End')

        for idx, (fpc, stable) in enumerate(zip(fp_pc, fp_stable)):
            colour = 'blue' if stable else 'red'
            marker = 'o' if stable else 'x'
            sz = 150 if stable else 120
            lbl = ('Stable FP' if stable else 'Unstable FP') if idx == 0 else None
            ax.scatter(fpc[0], fpc[1], c=colour, marker=marker, s=sz,
                       edgecolors='black', linewidths=1.5, zorder=10,
                       label=lbl)

        ax.set_xlabel(
            f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)')
        ax.set_ylabel(
            f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Phase-space plot saved to {save_path}")
        plt.close(fig)
        return fig
