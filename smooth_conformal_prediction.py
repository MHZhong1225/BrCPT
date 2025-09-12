# smooth_conformal_prediction.py (PyTorch Version)

from typing import Callable
import torch
from torchsort import soft_rank

# Define type hints for clarity
_SmoothQuantileFn = Callable[[torch.Tensor, float], torch.Tensor]

def soft_quantile(
    tensor: torch.Tensor,
    q: float,
    regularization_strength: float = 0.1
) -> torch.Tensor:
    """
    A self-contained implementation of a differentiable quantile using torchsort's soft_rank.

    This function calculates the soft rank of the input tensor, converts it to a
    probability distribution, and then uses that to compute a weighted average
    of the sorted tensor, which is a differentiable approximation of the quantile.

    Args:
        tensor (torch.Tensor): The 1D input tensor of scores. It should have requires_grad=True.
        q (float): The quantile to compute (e.g., 0.1 for the 10th percentile).
        regularization_strength (float): The strength for the soft rank.

    Returns:
        torch.Tensor: A scalar tensor representing the differentiable quantile.
    """
    # Sort the tensor to get the ordered values
    sorted_tensor = torch.sort(tensor)[0]
    
    # Use torchsort to get the differentiable rank of each element
    # We pass the tensor in a batch of 1
    soft_ranks = soft_rank(
        tensor.unsqueeze(0), 
        regularization_strength=regularization_strength
    ).squeeze(0)

    # The target rank for the q-th quantile
    n = tensor.shape[0]
    target_rank = q * (n - 1)

    # Create weights based on how close each element's soft rank is to the target rank
    # We use a Gaussian-like weighting (a squared exponential kernel)
    weights = torch.exp(-((soft_ranks - target_rank)**2) / (2 * 0.5**2))
    weights = weights / weights.sum()

    # The soft quantile is the weighted average of the original (unsorted) tensor
    soft_q = torch.sum(weights * tensor)
    
    return soft_q

def smooth_conformal_quantile(
    scores: torch.Tensor,
    alpha: float,
    regularization_strength: float = 0.1
) -> torch.Tensor:
    """
    Computes a differentiable quantile of the scores using torchsort.

    This function is the core of the differentiable calibration step.

    Args:
        scores (torch.Tensor): A 1D tensor of conformity scores from the calibration set.
        alpha (float): The desired miscoverage level.
        regularization_strength (float): The strength of the regularization for the soft quantile.
                                         A smaller value is closer to the true quantile but might be
                                         less stable for differentiation.

    Returns:
        torch.Tensor: A scalar tensor representing the smoothed (differentiable) threshold tau.
    """
    # Adjust the quantile level for finite sample correction
    n = scores.shape[0]
    level = (n + 1) * alpha / n 
    level = min(level, 1.0) # Ensure level does not exceed 1
    
    # We want to find the threshold that rejects alpha proportion of the lowest scores.
    # This corresponds to the alpha-quantile of the scores.
    # torchsort's soft_rank is what we need.
    # It returns a differentiable approximation of the quantile.
    return soft_quantile(scores, q=level, regularization_strength=regularization_strength)

def smooth_predict_threshold(
    scores: torch.Tensor,
    tau: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Creates smooth (differentiable) prediction sets using a sigmoid function.

    Instead of a hard threshold (score >= tau), this function outputs a
    probability of inclusion for each class, making the operation differentiable.

    Args:
        scores (torch.Tensor): Model output scores for the prediction set. Shape: [n_pred, n_classes].
        tau (torch.Tensor): The smoothed threshold from `smooth_conformal_quantile`.
        temperature (float): Controls the steepness of the sigmoid function. A smaller
                             temperature makes the function closer to a hard step function.

    Returns:
        torch.Tensor: A tensor of probabilities representing the soft prediction sets.
                      Shape: [n_pred, n_classes].
    """
    return torch.sigmoid((scores - tau) / temperature)

if __name__ == '__main__':
    # --- This is a test script to verify the smooth functions ---
    
    # --- Test Configuration ---
    CALIBRATION_SET_SIZE = 50
    ALPHA = 0.1
    REG_STRENGTH = 0.01

    # --- 1. Test smooth_conformal_quantile ---
    print("\n--- Testing smooth_conformal_quantile ---")
    
    # Create a dummy tensor of scores that requires gradients
    dummy_scores = torch.randn(CALIBRATION_SET_SIZE, requires_grad=True)
    
    # Compute the smooth quantile
    tau = smooth_conformal_quantile(dummy_scores, alpha=ALPHA, regularization_strength=REG_STRENGTH)
    
    print(f"Computed smooth quantile (tau): {tau.item()}")
    
    # Verify that we can backpropagate through it
    tau.backward()
    assert dummy_scores.grad is not None, "Gradients are not flowing through smooth_conformal_quantile!"
    print("Backward pass successful for smooth_conformal_quantile.")

    # --- 2. Test smooth_predict_threshold ---
    print("\n--- Testing smooth_predict_threshold ---")

    PREDICTION_SET_SIZE = 10
    NUM_CLASSES = 7
    TEMPERATURE = 0.1

    # Create dummy scores and a dummy tau that require gradients
    dummy_pred_scores = torch.randn(PREDICTION_SET_SIZE, NUM_CLASSES, requires_grad=True)
    dummy_tau = torch.tensor(0.5, requires_grad=True)

    # Get smooth prediction sets
    smooth_sets = smooth_predict_threshold(dummy_pred_scores, dummy_tau, temperature=TEMPERATURE)
    
    print(f"Shape of smooth prediction sets: {smooth_sets.shape}")
    assert smooth_sets.shape == (PREDICTION_SET_SIZE, NUM_CLASSES), "Shape of smooth sets is incorrect."
    
    # Verify that gradients flow
    smooth_sets.sum().backward()
    assert dummy_pred_scores.grad is not None, "Gradients are not flowing through scores in smooth_predict_threshold!"
    assert dummy_tau.grad is not None, "Gradients are not flowing through tau in smooth_predict_threshold!"
    print("Backward pass successful for smooth_predict_threshold.")

    print("\nAll smooth function tests passed!")
    