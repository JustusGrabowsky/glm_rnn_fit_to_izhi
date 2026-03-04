"""
plrnn_models.py - PLRNN and Dendritic PLRNN models for neural encoding

Two PLRNN architectures adapted for input-driven spike prediction:
- Standard PLRNN: z_{t+1} = A*z_t + W*relu(z_t) + h + C*I_t
- Dendritic PLRNN (dendPLRNN): z_{t+1} = A*z_t + W*basis_expansion(z_t) + h + C*I_t

Training uses BPTT with Teacher Forcing (TF) where observations are
forced into the latent state every tau steps.

Adapted from Durstewitz (2017) PLRNN with external input support.
"""

import math
import numpy as np
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# Latent Step Modules
# =============================================================================

class Latent_Step(nn.Module):
    """Base class for PLRNN latent steps."""

    def __init__(self, dz: int, clip_range: Optional[float] = None):
        super().__init__()
        self.dz = dz
        self.clip_range = clip_range

    def init_AW(self) -> nn.Parameter:
        """
        Initialize AW matrix following Talathi & Vartak 2016.
        Spectral radius normalized to 1.
        """
        matrix_random = torch.randn(self.dz, self.dz)
        matrix_positive_normal = (1 / self.dz) * matrix_random.T @ matrix_random
        matrix = torch.eye(self.dz) + matrix_positive_normal
        max_ev = torch.max(torch.abs(torch.linalg.eigvals(matrix)))
        matrix_spectral_norm_one = matrix / max_ev
        return nn.Parameter(matrix_spectral_norm_one, requires_grad=True)

    def init_uniform(self, shape: Tuple[int, ...]) -> nn.Parameter:
        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)

    def clip_z_to_range(self, z: torch.Tensor) -> torch.Tensor:
        if self.clip_range is not None:
            torch.clip_(z, -self.clip_range, self.clip_range)
        return z


class PLRNN_Step(Latent_Step):
    """
    Standard PLRNN latent step with external input.

    z_{t+1} = A * z_t + relu(z_t) @ W^T + h + C * I_t
    """

    def __init__(self, dz: int, dim_x: int, clip_range: Optional[float] = None):
        super().__init__(dz, clip_range)
        self.AW = self.init_AW()
        self.h = self.init_uniform((self.dz,))
        self.C = nn.Linear(dim_x, dz, bias=False)

    def forward(
        self, z: torch.Tensor, A: torch.Tensor, W: torch.Tensor,
        h: torch.Tensor, inj_input: torch.Tensor
    ) -> torch.Tensor:
        z_activated = torch.relu(z)
        z = A * z + z_activated @ W.t() + h + self.C(inj_input)
        return self.clip_z_to_range(z)


class PLRNN_Basis_Step(Latent_Step):
    """
    Dendritic PLRNN latent step with basis expansion and external input.

    z_{t+1} = A * z_t + basis_expansion(z_t) @ W^T + h + C * I_t

    where basis_expansion(z) = sum_j alpha_j * relu(z + theta_j)
    """

    def __init__(
        self, dz: int, dim_x: int, db: int,
        clip_range: Optional[float] = None
    ):
        super().__init__(dz, clip_range)
        self.AW = self.init_AW()
        self.h = self.init_uniform((self.dz,))
        self.db = db
        self.C = nn.Linear(dim_x, dz, bias=False)

        self.thetas = nn.Parameter(torch.randn(self.dz, self.db))
        self.alphas = self.init_uniform((self.db,))

    def forward(
        self, z: torch.Tensor, A: torch.Tensor, W: torch.Tensor,
        h: torch.Tensor, alphas: torch.Tensor, thetas: torch.Tensor,
        inj_input: torch.Tensor
    ) -> torch.Tensor:
        # z: (batch, dz) -> unsqueeze to (batch, dz, 1) for broadcasting with thetas (dz, db)
        z_expanded = z.unsqueeze(-1)
        # thetas are broadcasted: (batch, dz, 1) + (dz, db) -> (batch, dz, db)
        be = torch.sum(alphas * torch.relu(z_expanded + thetas), dim=-1)  # (batch, dz)
        z = A * z + be @ W.t() + h + self.C(inj_input)
        return self.clip_z_to_range(z)


# =============================================================================
# Main PLRNN Module
# =============================================================================

class PLRNNModule(nn.Module):
    """
    Input-driven PLRNN for spike prediction.

    Supports standard PLRNN (n_bases=0) and dendritic PLRNN (n_bases>0).
    Training uses BPTT with teacher forcing every tau steps.

    Args:
        dim_x: Input dimension (typically 1 for stimulus)
        dim_z: Latent state dimension (hidden size)
        dim_y: Output dimension (typically 1 for spike rate)
        n_bases: Number of basis functions (0 = standard PLRNN, >0 = dendPLRNN)
        clip_range: Latent state clipping value
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dim_y: int,
        n_bases: int = 0,
        clip_range: Optional[float] = None
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.n_bases = n_bases
        self.use_bases = n_bases > 0

        if self.use_bases:
            self.latent_step = PLRNN_Basis_Step(
                dz=dim_z, dim_x=dim_x, db=n_bases, clip_range=clip_range
            )
        else:
            self.latent_step = PLRNN_Step(
                dz=dim_z, dim_x=dim_x, clip_range=clip_range
            )

        # Readout layer: latent -> output
        self.readout = nn.Linear(dim_z, dim_y)
        self.softplus = nn.Softplus()

    def get_latent_parameters(self):
        """Extract A, W, h from the AW matrix."""
        AW = self.latent_step.AW
        A = torch.diag(AW)
        W = AW - torch.diag(A)
        h = self.latent_step.h
        return A, W, h

    def get_parameters(self):
        """Get all parameters needed for the latent step forward call."""
        params = self.get_latent_parameters()
        if self.use_bases:
            params += (self.latent_step.alphas, self.latent_step.thetas)
        return params

    def forward(
        self,
        input_sequence: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        tau: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.

        Args:
            input_sequence: External stimulus (batch, T, dim_x)
            observations: Spike observations for TF (batch, T) or None for free run
            tau: Teacher forcing interval. Force every tau steps. None = no forcing.

        Returns:
            Predicted rates (batch, T, dim_y)
        """
        batch, T, _ = input_sequence.shape
        device = input_sequence.device

        # Initialize latent state
        z = torch.randn(batch, self.dim_z, device=device) * 0.01

        # If observations provided and tau given, force initial state
        if observations is not None and tau is not None:
            z[:, :self.dim_y] = observations[:, 0].unsqueeze(-1) if observations.ndim == 2 else observations[:, 0]

        # Gather parameters
        params = self.get_parameters()

        outputs = []
        for t in range(T):
            # Teacher forcing: force first dim_y dims of z to match observations
            if observations is not None and tau is not None and t > 0 and t % tau == 0:
                if observations.ndim == 2:
                    z = z.clone()
                    z[:, :self.dim_y] = observations[:, t].unsqueeze(-1) if self.dim_y == 1 else observations[:, t]
                else:
                    z = z.clone()
                    z[:, :self.dim_y] = observations[:, t]

            # Current input
            inp_t = input_sequence[:, t, :]  # (batch, dim_x)

            # Latent step
            if self.use_bases:
                A, W, h, alphas, thetas = params
                z = self.latent_step(z, A, W, h, alphas, thetas, inp_t)
            else:
                A, W, h = params
                z = self.latent_step(z, A, W, h, inp_t)

            # Readout
            y_t = self.softplus(self.readout(z))  # (batch, dim_y)
            outputs.append(y_t)

        # Stack: (batch, T, dim_y)
        return torch.stack(outputs, dim=1)


# =============================================================================
# API Wrapper: TorchPLRNNRegressor
# =============================================================================

class TorchPLRNNRegressor:
    """
    PLRNN wrapper with sklearn-like API matching TorchRNNRegressor.

    Supports standard PLRNN (n_bases=0) and dendritic PLRNN (n_bases>0).
    Uses BPTT with teacher forcing and Poisson NLL loss.

    Args:
        hidden_dim: Latent state dimension
        n_bases: Number of basis functions (0=standard PLRNN, >0=dendPLRNN)
        tau: Teacher forcing interval (force every tau steps)
        n_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for RAdam optimizer
        clip_range: Latent state clipping value
        gradient_clip: Max gradient norm for clipping
        device: Compute device
        verbose: Print training progress
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_bases: int = 0,
        tau: int = 10,
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        clip_range: float = 10.0,
        gradient_clip: float = 10.0,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.n_bases = n_bases
        self.tau = tau
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.gradient_clip = gradient_clip
        self.verbose = verbose

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[PLRNNModule] = None
        self.train_losses: List[float] = []
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TorchPLRNNRegressor':
        """
        Fit the PLRNN model using BPTT with teacher forcing.

        Args:
            X: Input sequences (N, T, dim_x) - stimulus
            y: Target spike counts (N, T) - spikes

        Returns:
            self
        """
        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        dim_x = X.shape[2]
        dim_y = 1  # Single output (spike rate)

        # Z-score normalization of inputs
        self.X_mean = np.mean(X, axis=(0, 1), keepdims=True)
        self.X_std = np.std(X, axis=(0, 1), keepdims=True)
        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)

        # Initialize model
        self.model = PLRNNModule(
            dim_x=dim_x,
            dim_z=self.hidden_dim,
            dim_y=dim_y,
            n_bases=self.n_bases,
            clip_range=self.clip_range
        ).to(self.device)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # RAdam optimizer as in the paper
        optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.learning_rate)

        # Multi-step LR schedule as in original BPTT trainer
        e = self.n_epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [int(0.1 * e), int(0.8 * e), int(0.9 * e)], gamma=0.1
        )

        self.train_losses = []
        self.model.train()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad(set_to_none=True)

                # Forward with teacher forcing
                pred = self.model(
                    input_sequence=X_batch,
                    observations=y_batch,
                    tau=self.tau
                )  # (batch, T, 1)
                pred = pred.squeeze(-1)  # (batch, T)

                # Poisson NLL loss: lambda - y * log(lambda)
                eps = 1e-8
                loss = torch.mean(pred - y_batch * torch.log(pred + eps))

                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.gradient_clip
                )

                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                print(f"    Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        self._is_fitted = True
        return self

    def predict_rate(
        self, X: np.ndarray, return_sequence: bool = False
    ) -> np.ndarray:
        """
        Predict firing rates (free run, no teacher forcing).

        Args:
            X: Input sequences (N, T, dim_x)
            return_sequence: If True, return full sequence predictions

        Returns:
            Predicted firing rates
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)

        self.model.eval()
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            # Free run: no observations, no teacher forcing
            pred = self.model(input_sequence=X_tensor, observations=None, tau=None)
            pred = pred.squeeze(-1)  # (batch, T)

        result = pred.cpu().numpy()

        if return_sequence:
            return result
        else:
            # Return last timestep prediction (consistent with other regressors)
            return result
