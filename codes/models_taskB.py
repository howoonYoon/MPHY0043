from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18ToolDet(nn.Module):
    def __init__(self, num_tools: int = 7, pretrained: bool = True):
        super().__init__()
        if pretrained:
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            m = resnet18(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_tools)  # logits
        self.net = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,7) logits

class ResNet18ToolDetWithTimePred(nn.Module):
    def __init__(
        self,
        time_feat_dim: int,
        num_tools: int = 7,
        pretrained: bool = True,
        dropout: float = 0.2,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        if pretrained:
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            m = resnet18(weights=None)

        self.backbone = nn.Sequential(*list(m.children())[:-1])  # output (B,512,1,1)
        self.img_dim = m.fc.in_features  # 512

        self.time_emb = nn.Sequential(
            nn.LayerNorm(time_feat_dim),
            nn.Linear(time_feat_dim, time_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(self.img_dim + time_emb_dim, self.img_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.img_dim, num_tools),
        )

    def forward(self, x: torch.Tensor, tfeat: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x).flatten(1)  # (B,512)
        t = self.time_emb(tfeat)         # (B,time_emb_dim)
        z = torch.cat([f, t], dim=1)
        return self.head(z)
