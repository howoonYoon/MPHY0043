from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class TimeOnlyMLP(nn.Module):
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
        return self.net(x)


class FeatLSTMTime(nn.Module):
    def __init__(self, feat_dim: int, time_dim: int, hidden: int = 256, layers: int = 1, out_dim: int = 15, dropout: float = 0.2):
        super().__init__()
        self.feat_enc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.in_dim = feat_dim + time_dim
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=0.0 if layers == 1 else dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.in_norm = nn.LayerNorm(self.in_dim)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, feat_seq: torch.Tensor, time_seq: torch.Tensor) -> torch.Tensor:
        # feat_seq: (B,L,512), time_seq: (B,L,Ft)
        feat_seq = self.feat_enc(feat_seq)
        x = torch.cat([feat_seq, time_seq], dim=-1)
        x = self.in_norm(x)
        out, _ = self.lstm(x)       # (B,L,H)
        attn_w = torch.softmax(self.attn(out), dim=1)  # (B,L,1)
        h = (out * attn_w).sum(dim=1)                  # (B,H)
        return self.head(h)                            # (B,15)


class TimeOnlyLSTM(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 128, num_layers: int = 1, out_dim: int = 15, dropout: float = 0.0):
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
        # x_seq: (B, L, F)
        out, _ = self.lstm(x_seq)        # out: (B, L, H)
        h_last = out[:, -1, :]           # (B, H)
        return self.head(h_last)         # (B, out_dim)
