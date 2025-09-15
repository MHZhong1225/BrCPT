# models.py (PyTorch Version)

from typing import Dict, Tuple # 确保引入 Tuple
import torch
import torch.nn as nn
from torchvision import models

class ThresholdNet(nn.Module):
    """
    一个简单的MLP，用于根据图像特征预测一个个性化的置信阈值 h(x)。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(ThresholdNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # 使用Sigmoid确保阈值在(0, 1)范围内，因为我们的整合分数是概率
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1) # 输出形状 [batch_size]

class UACTModel(nn.Module):
    """
    统一自适应置信训练模型 (U-ACT Model)。
    包含一个主干网络 (用于特征提取和分类) 和一个阈值网络。
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(UACTModel, self).__init__()
        
        # 1. 加载主干网络 (ResNet-50)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        num_ftrs = self.backbone.fc.in_features
        
        # 移除原始的fc层，我们将手动执行avgpool和分类
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(num_ftrs, num_classes)
        
        # 2. 创建阈值网络
        self.threshold_net = ThresholdNet(input_dim=num_ftrs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 通过主干网络提取特征
        features = self.backbone(x) # [batch_size, num_ftrs, 1, 1]
        features = torch.flatten(features, 1) # [batch_size, num_ftrs]
        
        # a) 分类头输出logits
        logits = self.classifier(features)
        
        # b) 阈值网络输出个性化阈值 h(x)
        # 我们使用 .detach() 来阻止梯度从阈值流向特征提取器
        # 因为阈值网络的训练应该只基于当前特征的好坏，而不应该反过来影响特征提取
        h_x = self.threshold_net(features.detach())
        
        return logits, h_x


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Creates a ResNet-50 model for the BRACS classification task.

    Args:
        num_classes (int): The number of output classes for the model.
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        nn.Module: The constructed ResNet-50 model.
    """
    # Load the specified ResNet-50 model from torchvision
    if pretrained:
        print("Loading pretrained ResNet-50 model.")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        print("Creating a new ResNet-50 model from scratch.")
        model = models.resnet50(weights=None)

    # The final fully connected layer in a standard ResNet is named 'fc'.
    # We need to get the number of input features to this layer.
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer with a new one that has the
    # correct number of output units (num_classes for our BRACS dataset).
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    print(f"Model created. Final layer configured for {num_classes} classes.")
    
    return model

if __name__ == '__main__':
    # This is a simple test to verify that our model creation function works correctly.
    
    # --- Configuration for the test ---
    NUM_CLASSES_BRACS = 7  # We have 7 classes in our BRACS dataset
    USE_PRETRAINED = True
    BATCH_SIZE = 4
    IMAGE_SIZE = 224



    # --- Create the model ---
    test_model = create_model(num_classes=NUM_CLASSES_BRACS, pretrained=USE_PRETRAINED)
    
    # --- Create a dummy input tensor ---
    # The shape should be (batch_size, channels, height, width)
    dummy_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    # --- Perform a forward pass ---
    print("\n--- Model Test ---")
    print(f"Input tensor shape: {dummy_input.shape}")
    output = test_model(dummy_input)
    print(f"Output tensor shape: {output.shape}") # Should be [BATCH_SIZE, NUM_CLASSES_BRACS]

    # --- Verify output shape ---
    assert output.shape == (BATCH_SIZE, NUM_CLASSES_BRACS), "The output shape is incorrect!"
    
    print("\nModel test passed successfully!")



    print("\n--- UACTModel Test ---")
    uact_model = UACTModel(num_classes=7, pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)
    logits_out, h_out = uact_model(dummy_input)
    print(f"Logits shape: {logits_out.shape}") # 应为 [4, 7]
    print(f"Thresholds shape: {h_out.shape}") # 应为 [4]
    assert logits_out.shape == (4, 7)
    assert h_out.shape == (4,)
    print("UACTModel test passed!")
    