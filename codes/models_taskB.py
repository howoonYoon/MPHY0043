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

