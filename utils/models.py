# models.py (Refactored Version)

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models

# --- 内部辅助函数，用于创建和处理Backbone ---
def _create_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    一个内部辅助函数，用于加载指定的backbone并移除其分类头。
    返回 feature extractor 和 feature dimension。
    """
    if backbone_name == 'resnet18':
        print("Loading pretrained ResNet-18 backbone.")
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity() # 移除分类头
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
        # EfficientNet的特征提取部分在 'features'，分类器在 'classifier'
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Backbone '{backbone_name}' is not supported.")
        
    return backbone, num_ftrs

# --- UACT 模型的定义 ---
class ThresholdNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(ThresholdNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)

class UACTModel(nn.Module):
    def __init__(self, backbone_module: nn.Module, num_ftrs: int, num_classes: int):
        super(UACTModel, self).__init__()
        self.backbone = backbone_module
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.threshold_net = ThresholdNet(input_dim=num_ftrs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        logits = self.classifier(features)
        h_x = self.threshold_net(features.detach())
        
        return logits, h_x

# --- 最终的、统一的模型创建函数 ---
def get_model(
    model_type: str, 
    backbone_name: str, 
    num_classes: int, 
    pretrained: bool = True
) -> nn.Module:
    """
    统一的模型创建函数。

    Args:
        model_type (str): 模型类型, 'standard' 或 'uact'.
        backbone_name (str): Backbone名称, e.g., 'resnet34'.
        num_classes (int): 分类数量.
        pretrained (bool): 是否使用预训练权重.

    Returns:
        nn.Module: 构建好的PyTorch模型.
    """
    # 1. 创建基础的backbone特征提取器
    backbone_module, num_ftrs = _create_backbone(backbone_name, pretrained)
    
    # 2. 根据模型类型构建最终模型
    if model_type == 'standard':
        print(f"Creating a 'standard' model with '{backbone_name}' backbone.")
        # 对于标准模型，我们只需要在backbone后加上分类头
        model = nn.Sequential(
            backbone_module,
            nn.Flatten(),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_type == 'uact':
        print(f"Creating a 'UACT' model with '{backbone_name}' backbone.")
        model = UACTModel(backbone_module, num_ftrs, num_classes)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported. Choose 'standard' or 'uact'.")
        
    print(f"Model created. Final layer configured for {num_classes} classes.")
    return model