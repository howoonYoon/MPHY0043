from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class TimeOnlyMLP(nn.Module):
    """
    Time-only MLP baseline for Task A.

    This model predicts remaining surgical time targets using only
    point-wise time features (no temporal context).

    Architecture
    ------------
    Linear(in_dim -> hidden) + ReLU
    Linear(hidden -> hidden) + ReLU
    Linear(hidden -> out_dim)

    Parameters
    ----------
    in_dim:
        Dimensionality of input time features (e.g., 1 for "u", 2 for "u,u^2").
    out_dim:
        Output dimensionality (default 15 for Task A targets).
    hidden:
        Hidden layer width.
    """

    def __init__(self, in_dim: int = 1, out_dim: int = 15, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (B, in_dim).

        Returns
        -------
        y_pred:
            Predicted targets in log-space, shape (B, out_dim).
        """
        return self.net(x)


class FeatLSTMTime(nn.Module):
    """
    Feature + time LSTM model with temporal attention for Task A.

    This model consumes:
      - a sequence of visual features (e.g., ResNet-18 features),
      - a sequence of time features,
    and predicts remaining surgical time targets at the current timestep.

    Design
    ------
    1) Feature encoder (LayerNorm + Linear + ReLU)
    2) Concatenate encoded features with time features
    3) LSTM over the sequence
    4) Temporal attention over LSTM outputs
    5) MLP head to predict Task A targets

    Parameters
    ----------
    feat_dim:
        Dimensionality of visual feature vectors (e.g., 512).
    time_dim:
        Dimensionality of time feature vectors.
    hidden:
        LSTM hidden size.
    layers:
        Number of LSTM layers.
    out_dim:
        Output dimensionality (default 15 for Task A targets).
    dropout:
        Dropout probability used in feature encoder and prediction head.
    """

    def __init__(
        self,
        feat_dim: int,
        time_dim: int,
        hidden: int = 256,
        layers: int = 1,
        out_dim: int = 15,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Encode visual features before fusion
        self.feat_enc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.in_dim = feat_dim + time_dim

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.0 if layers == 1 else dropout,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

        # Normalization and attention
        self.in_norm = nn.LayerNorm(self.in_dim)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, feat_seq: torch.Tensor, time_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        feat_seq:
            Visual feature sequence of shape (B, L, feat_dim).
        time_seq:
            Time feature sequence of shape (B, L, time_dim).

        Returns
        -------
        y_pred:
            Predicted targets in log-space, shape (B, out_dim).

        Notes
        -----
        - Attention weights are normalized across the temporal dimension (L).
        - The final sequence representation is a weighted sum of LSTM outputs.
        """
        # Encode visual features
        feat_seq = self.feat_enc(feat_seq)

        # Concatenate with time features
        x = torch.cat([feat_seq, time_seq], dim=-1)
        x = self.in_norm(x)

        # LSTM over time
        out, _ = self.lstm(x)  # (B, L, H)

        # Temporal attention
        attn_w = torch.softmax(self.attn(out), dim=1)  # (B, L, 1)
        h = (out * attn_w).sum(dim=1)                  # (B, H)

        return self.head(h)                            # (B, out_dim)


class TimeOnlyLSTM(nn.Module):
    """
    Time-only LSTM model for Task A.

    This model predicts remaining surgical time targets using
    only temporal sequences of time features (no visual input).

    Architecture
    ------------
    - LSTM over time-feature sequences
    - MLP head on the final timestep hidden state

    Parameters
    ----------
    in_dim:
        Dimensionality of input time features.
    hidden:
        LSTM hidden size.
    num_layers:
        Number of LSTM layers.
    out_dim:
        Output dimensionality (default 15 for Task A targets).
    dropout:
        Dropout probability between LSTM layers (ignored if num_layers=1).
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 128,
        num_layers: int = 1,
        out_dim: int = 15,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_seq:
            Time-feature sequence of shape (B, L, in_dim).

        Returns
        -------
        y_pred:
            Predicted targets in log-space, shape (B, out_dim).
        """
        out, _ = self.lstm(x_seq)   # (B, L, H)
        h_last = out[:, -1, :]      # last timestep hidden state (B, H)
        return self.head(h_last)    # (B, out_dim)
