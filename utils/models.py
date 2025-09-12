# models.py (PyTorch Version)

from typing import Dict
import torch
import torch.nn as nn
from torchvision import models

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