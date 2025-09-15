# utils/explainability.py

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models

class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    This class allows generating a heatmap that highlights the regions
    in an image that are most important for a given prediction.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        self._register_hooks()

    def _capture_activations(self, module, input, output):
        """Forward hook to capture the activations."""
        self.activations = output.detach()

    def _capture_gradients(self, module, grad_in, grad_out):
        """Backward hook to capture the gradients."""
        self.gradients = grad_out[0].detach()

    def _register_hooks(self):
        """Register forward and backward hooks to the target layer."""
        self.target_layer.register_forward_hook(self._capture_activations)
        self.target_layer.register_backward_hook(self._capture_gradients)

    def generate_heatmap(self, output: torch.Tensor, target_class_index: int) -> torch.Tensor:
        """
        Generates the Grad-CAM heatmap.

        Args:
            output (torch.Tensor): The raw output (logits) from the model.
            target_class_index (int): The index of the class for which to generate the map.

        Returns:
            torch.Tensor: A 2D tensor representing the heatmap.
        """
        # Step 1: Zero out gradients from any previous runs
        self.model.zero_grad()

        # Step 2: Get the score for the target class
        class_score = output[:, target_class_index]

        # Step 3: Perform backward pass to compute gradients
        class_score.backward(retain_graph=True)

        # Ensure gradients and activations were captured
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations. Check hook registration.")
        
        # Step 4: Pool the gradients across spatial dimensions to get channel weights
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3])

        # Step 5: Weight the activation channels with the computed gradients
        # Get activations for the specific image in the batch (assuming batch size 1 for simplicity)
        activations = self.activations[0] 
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[0, i]

        # Step 6: Average the channels to create the heatmap
        heatmap = torch.mean(activations, dim=0)
        
        # Step 7: Apply ReLU to keep only positive influences
        heatmap = F.relu(heatmap)
        
        # Step 8: Normalize the heatmap
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays a heatmap on an original image.

    Args:
        image (np.ndarray): The original image (in BGR format, 0-255).
        heatmap (np.ndarray): The heatmap (0-1).
        alpha (float): The blending factor for the heatmap.
        colormap: The OpenCV colormap to use.

    Returns:
        np.ndarray: The image with the heatmap overlay.
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to a color map and apply it
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    
    # Blend the heatmap with the original image
    overlaid_image = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlaid_image