# models.py

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def _create_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Load backbone and remove classifier head.
    Return feature extractor and feature dimension.
    """
    if backbone_name == 'resnet18':
        print("Loading pretrained ResNet-18 backbone.")
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == 'resnet34':
        print("Loading pretrained ResNet-34 backbone.")
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == 'resnet50':
        print("Loading pretrained ResNet-50 backbone.")
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == 'efficientnet_b0':
        print("Loading pretrained EfficientNet-B0 backbone.")
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Backbone '{backbone_name}' is not supported.")

    return backbone, num_ftrs


class ThresholdNet(nn.Module):
    """
    Predicts h(x) in probability domain.
    CRITICAL FIX:
      - clamp pre-sigmoid z to avoid explosion / saturation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, z_clip: float = 5.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()
        self.z_clip = float(z_clip)

        # (optional) make weights small to start near 0.5
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z = self.fc2(self.act(self.fc1(features))).squeeze(1)  # pre-sigmoid
        z = torch.clamp(z, -self.z_clip, self.z_clip)          # prevent blow-up
        h = torch.sigmoid(z)                                  # in (0,1)
        return h


class CATModel(nn.Module):
    def __init__(self, backbone_module: nn.Module, num_ftrs: int, num_classes: int):
        super(CATModel, self).__init__()
        self.backbone = backbone_module
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.threshold_net = ThresholdNet(input_dim=num_ftrs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        h_x = self.threshold_net(features)
        return logits, h_x


def get_model(
    model_type: str,
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True
) -> nn.Module:
    """
    Unified model factory.
    """
    backbone_module, num_ftrs = _create_backbone(backbone_name, pretrained)

    if model_type == 'standard':
        print(f"Creating a 'standard' model with '{backbone_name}' backbone.")
        model = nn.Sequential(
            backbone_module,
            nn.Flatten(),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_type == 'cat':
        print(f"Creating a 'CAT' model with '{backbone_name}' backbone.")
        model = CATModel(backbone_module, num_ftrs, num_classes)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported. Choose 'standard' or 'cat'.")

    print(f"Model created. Final layer configured for {num_classes} classes.")
    return model