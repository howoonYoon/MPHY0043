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
        self.in_dim = feat_dim + time_dim
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=0.0 if layers == 1 else dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.in_norm = nn.LayerNorm(self.in_dim)

    def forward(self, feat_seq: torch.Tensor, time_seq: torch.Tensor) -> torch.Tensor:
        # feat_seq: (B,L,512), time_seq: (B,L,Ft)
        x = torch.cat([feat_seq, time_seq], dim=-1)
        x = self.in_norm(x)
        out, _ = self.lstm(x)       # (B,L,H)
        h = out.mean(dim=1)         # last step
        return self.head(h)         # (B,15)


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


class SmallResNet18(nn.Module):
    """ResNet18 encoder -> global pooled feature (512)."""
    def __init__(self):
        super().__init__()
        # torchvision 없이 간단히 쓰려면 직접 구현해야 해서,
        # 여기서는 torchvision 사용을 전제로 하는 게 현실적임.
        # (mphy0043-pt에 torchvision이 없으면 설치 필요)
        import torchvision.models as models
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # remove fc, keep avgpool
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)
        f = self.backbone(x)          # (B,512,1,1)
        f = f.flatten(1)              # (B,512)
        return f


class CNNLSTMTime(nn.Module):
    def __init__(
        self,
        time_dim: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        out_dim: int = 15,
        freeze_cnn: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cnn = SmallResNet18()
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        feat_in = self.cnn.out_dim + time_dim

        self.lstm = nn.LSTM(
            input_size=feat_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, out_dim),
        )

    def forward(self, img_seq: torch.Tensor, time_seq: torch.Tensor) -> torch.Tensor:
        # img_seq: (B,L,3,H,W)
        B, L, C, H, W = img_seq.shape
        x = img_seq.view(B * L, C, H, W)
        f = self.cnn(x)                    # (B*L,512)
        f = f.view(B, L, -1)               # (B,L,512)

        z = torch.cat([f, time_seq], dim=-1)  # (B,L,512+time_dim)
        out, _ = self.lstm(z)                 # (B,L,H)
        h_last = out[:, -1, :]
        return self.head(h_last)              # (B,15)



class ResNetLSTMTime(nn.Module):
    def __init__(
        self,
        time_dim: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        out_dim: int = 15,
        freeze_cnn: bool = True,     # 기존 유지
        unfreeze_layer4: bool = True, # ✅ 추가
        dropout: float = 0.2,
    ):
        super().__init__()

        base = resnet18(weights=ResNet18_Weights.DEFAULT)

        # ✅ backbone: fc 제거하고 512-d feature 뽑기
        self.cnn = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool
        )
        self.feat_dim = 512

        # ✅ freeze 정책
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

            # ✅ layer4만 unfreeze
            if unfreeze_layer4:
                for p in base.layer4.parameters():
                    p.requires_grad = True
                # (layer4가 self.cnn 안에 들어있으니 base.layer4로 풀어도 적용됨)

        # LSTM (feat + time)
        self.lstm = nn.LSTM(
            input_size=self.feat_dim + time_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else dropout,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, out_dim),
        )

    def forward(self, img_seq: torch.Tensor, time_seq: torch.Tensor) -> torch.Tensor:
        # img_seq: (B,L,3,H,W), time_seq: (B,L,Ft)
        B, L, C, H, W = img_seq.shape
        x = img_seq.view(B * L, C, H, W)

        f = self.cnn(x)                 # (B*L,512,1,1)
        f = f.flatten(1)                # (B*L,512)
        f = f.view(B, L, self.feat_dim) # (B,L,512)

        z = torch.cat([f, time_seq], dim=-1)  # (B,L,512+Ft)
        out, _ = self.lstm(z)
        h = out[:, -1, :]
        return self.head(h)
