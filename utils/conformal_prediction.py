# conformal_prediction.py (PyTorch Version)

from typing import Optional, Callable, Any

import torch

# Define type hints for clarity
_QuantileFn = Callable[[Any, float], float]
_CalibrateFn = Callable[[torch.Tensor, torch.Tensor, Any], Any]
_PredictFn = Callable[[torch.Tensor, Any, Any], torch.Tensor]

def conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    """
    Computes the corrected quantile for conformal prediction.
    
    This is a key step in split conformal prediction to ensure the coverage
    guarantee holds for future test examples. It computes a slightly adjusted
    quantile level.

    Args:
        scores (torch.Tensor): A 1D tensor of conformity scores from the calibration set.
        alpha (float): The desired miscoverage level (e.g., 0.05 for 95% coverage).

    Returns:
        float: The calibrated threshold tau.
    """
    # Adjust the quantile level for finite sample correction
    n = len(scores)
    level = torch.ceil(torch.tensor((n + 1) * (1 - alpha))) / n
    level = min(level, 1.0) # Ensure level does not exceed 1

    # Use item() to get the value as a Python float
    return torch.quantile(scores, level.item(), interpolation='higher')

def calibrate_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.1) -> float:
    """
    Calibrates a threshold for the simple threshold-based conformal method.

    Args:
        scores (torch.Tensor): The model's output scores (e.g., softmax probabilities)
                               for the calibration set. Shape: [n_cal, n_classes].
        labels (torch.Tensor): The ground truth labels for the calibration set. Shape: [n_cal].
        alpha (float): The desired miscoverage level.

    Returns:
        float: The calibrated threshold tau.
    """
    # Get the scores corresponding to the true labels
    n = scores.shape[0]
    conformity_scores = scores[torch.arange(n), labels]
    
    # We want to find a threshold such that (1-alpha) of the true scores are above it.
    # This is equivalent to the alpha-quantile of the *negative* scores.
    # Or, more directly, the alpha-quantile of the scores themselves. A low score
    # indicates a likely misclassification.
    
    # The quantile level needs to be corrected for the guarantee to hold.
    # P(Y in C(X)) >= 1 - alpha. We set the threshold `tau` to be the
    # q-th quantile of the conformity scores, where q = alpha.
    # To get high coverage, we need to accept scores that are low, so we take the alpha-quantile.
    n = len(conformity_scores)
    q = torch.ceil(torch.tensor((n + 1) * alpha)) / n
    q = min(q, 1.0)

    # A lower score is "less conforming". We want to find the threshold that
    # rejects alpha proportion of the lowest scores.
    return torch.quantile(conformity_scores, q.item(), interpolation='lower')


def predict_threshold(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Creates prediction sets using the simple thresholding method.

    Args:
        scores (torch.Tensor): Model output scores for the test set. Shape: [n_test, n_classes].
        tau (float): The calibrated threshold from `calibrate_threshold`.

    Returns:
        torch.Tensor: A boolean tensor representing the prediction sets. Shape: [n_test, n_classes].
    """
    return scores >= tau


def calibrate_raps(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.1,
    k_reg: Optional[int] = None,
    lambda_reg: Optional[float] = None) -> float:
    """
    Calibrates a threshold for Regularized Adaptive Prediction Sets (RAPS).

    Args:
        probabilities (torch.Tensor): Softmax probabilities for the calibration set.
        labels (torch.Tensor): Ground truth labels for the calibration set.
        alpha (float): Desired miscoverage level.
        k_reg (Optional[int]): Target set size for regularization.
        lambda_reg (Optional[float]): Regularization strength.

    Returns:
        float: The calibrated threshold tau for RAPS.
    """
    reg = k_reg is not None and lambda_reg is not None
    n = probabilities.shape[0]

    # Sort probabilities in descending order for each example
    sorted_probs, pi = torch.sort(probabilities, dim=1, descending=True)
    
    # Get the rank of the true label for each example
    # This requires creating an inverse permutation of pi
    reverse_pi = torch.argsort(pi, dim=1)
    true_label_rank = reverse_pi[torch.arange(n), labels]

    # Calculate cumulative sums of sorted probabilities
    cum_probs = torch.cumsum(sorted_probs, dim=1)
    
    # Get the cumulative probability for the true label
    conformity_scores = cum_probs[torch.arange(n), true_label_rank]

    if reg:
        # Add regularization term based on set size
        penalty = lambda_reg * torch.maximum(torch.tensor(0), true_label_rank + 1 - k_reg)
        conformity_scores += penalty.to(conformity_scores.device)
    
    # The quantile level for RAPS is 1 - alpha
    return conformal_quantile(conformity_scores, 1 - alpha)


def predict_raps(
    probabilities: torch.Tensor,
    tau: float,
    k_reg: Optional[int] = None,
    lambda_reg: Optional[float] = None) -> torch.Tensor:
    """
    Creates prediction sets using the RAPS method.

    Args:
        probabilities (torch.Tensor): Softmax probabilities for the test set.
        tau (float): The calibrated threshold from `calibrate_raps`.
        k_reg (Optional[int]): Target set size for regularization.
        lambda_reg (Optional[float]): Regularization strength.

    Returns:
        torch.Tensor: A boolean tensor representing the prediction sets.
    """
    reg = k_reg is not None and lambda_reg is not None
    n, num_classes = probabilities.shape

    # Sort probabilities in descending order
    sorted_probs, pi = torch.sort(probabilities, dim=1, descending=True)
    
    # Calculate cumulative sums
    cum_probs = torch.cumsum(sorted_probs, dim=1)

    if reg:
        # Add regularization term to cumulative probabilities
        ranks = torch.arange(1, num_classes + 1, device=probabilities.device).expand_as(cum_probs)
        penalty = lambda_reg * torch.maximum(torch.tensor(0), ranks - k_reg)
        cum_probs += penalty

    # Find the number of classes to include for each example
    # These are the sets before applying the inverse permutation
    sorted_sets = cum_probs <= tau
    
    # Use the inverse permutation to map the sets back to the original class order
    prediction_sets = torch.gather(sorted_sets, 1, torch.argsort(pi, dim=1))
    
    return prediction_sets