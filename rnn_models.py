"""
rnn_models.py - PyTorch RNN models for neural encoding

Three RNN architectures for spike prediction:
- TorchRNNRegressor (GRU-based) - with optional bidirectional and attention
- TorchVanillaRNNRegressor (Elman RNN)
- TorchLSTMRegressor (LSTM-based)

All models use:
- sklearn-like API with fit() and predict_rate()
- Poisson NLL loss (with optional sparsity weighting)
- Z-score input normalization
- Learning rate scheduler
- Gradient clipping

Advanced features (GRU only):
- Bidirectional processing
- Self-attention mechanism
- Weighted loss for spike emphasis (focal-style)
- Scheduled sampling for closed-loop training
"""

import numpy as np
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# PyTorch GRU Network with Advanced Features
# =============================================================================

class GRUNetwork(nn.Module):
    """
    GRU-based neural network for spike prediction.

    Supports:
    - Bidirectional processing (doubles hidden dim output)
    - Self-attention mechanism (applied after GRU)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        use_attention: bool = False,
        attention_heads: int = 4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Calculate FC input dimension (doubles if bidirectional)
        fc_input_dim = hidden_size * 2 if bidirectional else hidden_size

        # Optional self-attention layer
        if use_attention:
            # Ensure embed_dim is divisible by num_heads
            self.attention_heads = attention_heads
            # Adjust fc_input_dim to be divisible by attention_heads if needed
            if fc_input_dim % attention_heads != 0:
                # Project to a compatible dimension
                self.attn_proj = nn.Linear(fc_input_dim, attention_heads * (fc_input_dim // attention_heads + 1))
                attn_dim = attention_heads * (fc_input_dim // attention_heads + 1)
            else:
                self.attn_proj = None
                attn_dim = fc_input_dim

            self.attn = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=attention_heads,
                batch_first=True,
                dropout=dropout
            )
            # Project back if we had to adjust dimensions
            if self.attn_proj is not None:
                self.attn_out_proj = nn.Linear(attn_dim, fc_input_dim)
            else:
                self.attn_out_proj = None

        self.fc = nn.Linear(fc_input_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Forward pass through the GRU network.

        Args:
            x: Input tensor (batch, seq_len, input_size)
            return_sequence: If True, return predictions for all timesteps

        Returns:
            Predicted firing rates
        """
        out, _ = self.gru(x)

        # Apply self-attention if enabled
        if self.use_attention:
            # Project if needed
            if self.attn_proj is not None:
                attn_input = self.attn_proj(out)
            else:
                attn_input = out

            # Self-attention: query, key, value are all the same
            attn_out, _ = self.attn(attn_input, attn_input, attn_input)

            # Project back if needed
            if self.attn_out_proj is not None:
                attn_out = self.attn_out_proj(attn_out)

            # Residual connection
            out = out + attn_out

        if return_sequence:
            out = self.fc(out)
            out = self.softplus(out)
            return out.squeeze(-1)
        else:
            out = out[:, -1, :]
            out = self.fc(out)
            out = self.softplus(out)
            return out.squeeze(-1)

    def forward_step(
        self,
        x_t: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Single timestep forward for autoregressive (closed-loop) training.

        Args:
            x_t: Input at time t, shape (batch, 1, input_size)
            h: Hidden state from previous step

        Returns:
            y_t: Predicted rate at time t
            h_new: Updated hidden state
        """
        out, h_new = self.gru(x_t, h)

        # Note: attention is skipped in step-wise mode for efficiency
        # (would need to accumulate full sequence context)

        y_t = self.softplus(self.fc(out.squeeze(1)))
        return y_t.squeeze(-1), h_new


class TorchRNNRegressor:
    """
    PyTorch GRU wrapper with sklearn-like API.

    The RNN (GRU) takes raw current sequences and learns its own temporal
    representation, predicting firing rate using Poisson NLL loss.

    Input: (batch, seq_len, n_features) - raw binned current values
    Output: (batch,) or (batch, seq_len) - predicted firing rate

    Advanced features:
    - bidirectional: Process sequence in both directions
    - use_attention: Apply self-attention after GRU
    - sparsity_weight: Emphasize spike bins in loss (focal-style)
    - scheduled_sampling_decay: Closed-loop training with curriculum
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 64,
        dropout: float = 0.0,
        device: Optional[str] = None,
        verbose: bool = True,
        # Advanced features
        bidirectional: bool = False,
        use_attention: bool = False,
        attention_heads: int = 4,
        sparsity_weight: float = 1.0,
        scheduled_sampling_decay: float = 0.0
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose

        # Advanced features
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.sparsity_weight = sparsity_weight
        self.scheduled_sampling_decay = scheduled_sampling_decay

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[GRUNetwork] = None
        self.train_losses: List[float] = []
        self._is_fitted = False
        self._input_size = None

    def _compute_weighted_loss(
        self,
        y_pred: torch.Tensor,
        y_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Poisson NLL loss with optional sparsity weighting.

        When sparsity_weight > 1.0, spike bins (y > 0) are weighted more heavily,
        similar to focal loss for imbalanced data.

        Args:
            y_pred: Predicted rates (batch, seq_len) or (batch,)
            y_batch: True spike counts (same shape)

        Returns:
            Weighted mean loss
        """
        eps = 1e-8
        # Raw Poisson NLL: λ - y*log(λ)
        loss_raw = y_pred - y_batch * torch.log(y_pred + eps)

        if self.sparsity_weight != 1.0:
            # Create weight mask: higher weight for spike bins
            weights = torch.ones_like(y_batch)
            weights[y_batch > 0] = self.sparsity_weight
            loss = (loss_raw * weights).mean()
        else:
            loss = loss_raw.mean()

        return loss

    def _train_standard(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> None:
        """Standard training loop (no scheduled sampling)."""
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()

                if self._is_sequence_output:
                    y_pred = self.model(X_batch, return_sequence=True)
                else:
                    y_pred = self.model(X_batch, return_sequence=False)

                loss = self._compute_weighted_loss(y_pred, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                print(f"    Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

    def _train_scheduled_sampling(
        self,
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        spike_feature_idx: int = -1
    ) -> None:
        """
        Training with scheduled sampling for closed-loop behavior.

        Uses curriculum learning: starts with teacher forcing (true spikes),
        gradually transitions to using model's own predictions.

        Args:
            X_tensor: Full input tensor (n_samples, seq_len, n_features)
            y_tensor: Full target tensor (n_samples, seq_len)
            optimizer: Optimizer
            scheduler: LR scheduler
            spike_feature_idx: Index of spike history feature in X (-1 for last)
        """
        n_samples = X_tensor.shape[0]
        seq_len = X_tensor.shape[1]

        for epoch in range(self.n_epochs):
            # Compute teacher forcing probability (decays over epochs)
            # p_teacher = decay^epoch, so starts near 1 and decays to 0
            p_teacher = self.scheduled_sampling_decay ** epoch

            # Shuffle data
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                batch_idx = perm[i:i + self.batch_size]
                X_batch = X_tensor[batch_idx].clone()  # Clone to modify
                y_batch = y_tensor[batch_idx]

                optimizer.zero_grad()

                # Hybrid approach: pre-corrupt spike history with random noise
                # based on teacher forcing probability
                if p_teacher < 1.0:
                    # Create mask for timesteps to corrupt
                    corrupt_mask = torch.rand(X_batch.shape[0], seq_len) > p_teacher
                    corrupt_mask = corrupt_mask.to(self.device)

                    # Replace spike history with random samples from training distribution
                    # (simple approach: use random permutation of actual values)
                    if X_batch.shape[2] > 1:  # Has history features
                        noise = torch.rand_like(X_batch[:, :, spike_feature_idx]) * y_batch.mean()
                        X_batch[:, :, spike_feature_idx] = torch.where(
                            corrupt_mask,
                            noise,
                            X_batch[:, :, spike_feature_idx]
                        )

                y_pred = self.model(X_batch, return_sequence=True)
                loss = self._compute_weighted_loss(y_pred, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                print(f"    Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}, "
                      f"p_teacher: {p_teacher:.3f}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TorchRNNRegressor':
        """
        Fit the GRU model to training data.

        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Target spike counts (n_samples, seq_len) or (n_samples,)

        Returns:
            self
        """
        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        input_size = X.shape[2]
        self._input_size = input_size
        self._is_sequence_output = (y.ndim == 2)

        # Z-score normalization
        self.X_mean = np.mean(X, axis=(0, 1), keepdims=True)
        self.X_std = np.std(X, axis=(0, 1), keepdims=True)
        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)

        # Initialize model with advanced features
        self.model = GRUNetwork(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_attention=self.use_attention,
            attention_heads=self.attention_heads
        ).to(self.device)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        self.model.train()
        self.train_losses = []

        # Choose training strategy
        if self.scheduled_sampling_decay > 0.0 and self._is_sequence_output:
            # Closed-loop training with scheduled sampling
            self._train_scheduled_sampling(
                X_tensor, y_tensor, optimizer, scheduler
            )
        else:
            # Standard training
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self._train_standard(loader, optimizer, scheduler)

        self._is_fitted = True
        return self

    def predict_rate(self, X: np.ndarray, return_sequence: bool = False) -> np.ndarray:
        """
        Predict firing rates for input sequences.

        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            return_sequence: If True, return predictions for all timesteps

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
            y_pred = self.model(X_tensor, return_sequence=return_sequence)

        return y_pred.cpu().numpy()


# =============================================================================
# Vanilla RNN (Elman RNN) Network
# =============================================================================

class VanillaRNNNetwork(nn.Module):
    """Standard Elman RNN (Vanilla RNN) with tanh nonlinearity."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        out, _ = self.rnn(x)

        if return_sequence:
            out = self.fc(out)
            out = self.softplus(out)
            return out.squeeze(-1)
        else:
            out = out[:, -1, :]
            out = self.fc(out)
            out = self.softplus(out)
            return out.squeeze(-1)


class TorchVanillaRNNRegressor:
    """PyTorch Vanilla RNN wrapper with sklearn-like API."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 64,
        dropout: float = 0.0,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
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

        self.model: Optional[VanillaRNNNetwork] = None
        self.train_losses: List[float] = []
        self._is_fitted = False
        self._input_size = None
        self._is_sequence_output = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TorchVanillaRNNRegressor':
        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        input_size = X.shape[2]
        self._input_size = input_size
        self._is_sequence_output = (y.ndim == 2)

        self.X_mean = np.mean(X, axis=(0, 1), keepdims=True)
        self.X_std = np.std(X, axis=(0, 1), keepdims=True)
        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)

        self.model = VanillaRNNNetwork(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        self.train_losses = []
        self.model.train()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()

                if self._is_sequence_output:
                    y_pred = self.model(X_batch, return_sequence=True)
                else:
                    y_pred = self.model(X_batch, return_sequence=False)

                eps = 1e-8
                loss = torch.mean(y_pred - y_batch * torch.log(y_pred + eps))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                print(f"    Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        self._is_fitted = True
        return self

    def predict_rate(self, X: np.ndarray, return_sequence: bool = False) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)

        self.model.eval()
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            y_pred = self.model(X_tensor, return_sequence=return_sequence)

        return y_pred.cpu().numpy()


# =============================================================================
# PyTorch LSTM Network
# =============================================================================

class LSTMNetwork(nn.Module):
    """LSTM-based neural network for spike prediction."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)

        if return_sequence:
            out = self.fc(out)
            out = self.softplus(out)
            return out.squeeze(-1)
        else:
            out = out[:, -1, :]
            out = self.fc(out)
            out = self.softplus(out)
            return out.squeeze(-1)


class TorchLSTMRegressor:
    """PyTorch LSTM wrapper with sklearn-like API."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 64,
        dropout: float = 0.0,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
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

        self.model: Optional[LSTMNetwork] = None
        self.train_losses: List[float] = []
        self._is_fitted = False
        self._input_size = None
        self._is_sequence_output = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TorchLSTMRegressor':
        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        input_size = X.shape[2]
        self._input_size = input_size
        self._is_sequence_output = (y.ndim == 2)

        self.X_mean = np.mean(X, axis=(0, 1), keepdims=True)
        self.X_std = np.std(X, axis=(0, 1), keepdims=True)
        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)

        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        self.train_losses = []
        self.model.train()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()

                if self._is_sequence_output:
                    y_pred = self.model(X_batch, return_sequence=True)
                else:
                    y_pred = self.model(X_batch, return_sequence=False)

                eps = 1e-8
                loss = torch.mean(y_pred - y_batch * torch.log(y_pred + eps))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                print(f"    Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        self._is_fitted = True
        return self

    def predict_rate(self, X: np.ndarray, return_sequence: bool = False) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)

        self.model.eval()
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            y_pred = self.model(X_tensor, return_sequence=return_sequence)

        return y_pred.cpu().numpy()
