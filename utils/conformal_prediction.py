# conformal_prediction.py (PyTorch Version)

import math
from typing import Optional, Callable, Any
import numpy as np
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
    scores = scores.view(-1)  # ensure 1D
    n = scores.numel()
    # k = ceil((n+1)*(1-alpha)), then clamp to [1, n]
    k = int(torch.ceil(torch.tensor((n + 1) * (1 - alpha), device=scores.device)).item())
    k = max(1, min(k, n))
    v = torch.kthvalue(scores, k).values.item()
    return float(v)


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
    reg = (k_reg is not None) and (lambda_reg is not None)
    n = probabilities.size(0)

    sorted_probs, pi = torch.sort(probabilities, dim=1, descending=True)
    reverse_pi = torch.argsort(pi, dim=1)
    true_rank = reverse_pi[torch.arange(n, device=labels.device), labels]

    cum_probs = torch.cumsum(sorted_probs, dim=1)
    conformity_scores = cum_probs[torch.arange(n, device=labels.device), true_rank]

    if reg:
        #  RAPS：加 lambda * max(0, rank+1 - k_reg)
        penalty = (true_rank.to(conformity_scores.dtype) + 1.0) - float(k_reg)
        penalty = torch.clamp(penalty, min=0.0) * float(lambda_reg)
        # conformity_scores = conformity_scores + penalty
        conformity_scores += penalty.to(conformity_scores.device)
    # 这里要传 alpha（不是 1-alpha）
    return conformal_quantile(conformity_scores, alpha)

def calibrate_aps(probs: torch.Tensor, labels: torch.Tensor, alpha: float=0.1) -> float:
    with torch.no_grad():
        sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
        cumsums = torch.cumsum(sorted_probs, dim=1)
        N = probs.size(0)
        # 找到 true label 在排序后的列位置
        pos = (sorted_idx == labels.unsqueeze(1)).nonzero(as_tuple=False)
        pos_j = torch.zeros(N, dtype=torch.long, device=probs.device)
        pos_j[pos[:, 0]] = pos[:, 1]
        C_true = cumsums[torch.arange(N, device=probs.device), pos_j]
        # qhat at 1 - alpha
        n = C_true.numel()
        k = int(torch.ceil(torch.tensor((n + 1) * (1 - alpha), device=C_true.device)).item())
        k = max(1, min(k, n))
        qhat = torch.kthvalue(C_true, k).values.item()
        return float(qhat)

def predict_aps(probs: torch.Tensor, qhat: float, top1_guarantee: bool=True) -> torch.Tensor:
    with torch.no_grad():
        sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
        cumsums = torch.cumsum(sorted_probs, dim=1)
        keep_sorted = cumsums <= qhat
        keep = torch.zeros_like(keep_sorted, dtype=torch.bool)
        keep.scatter_(1, sorted_idx, keep_sorted)
        if top1_guarantee:
            top1 = torch.argmax(probs, dim=1)
            empty = ~keep.any(dim=1)
            keep[empty, top1[empty]] = True


        return keep

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
    reg = (k_reg is not None) and (lambda_reg is not None)
    n, C = probabilities.shape

    sorted_probs, pi = torch.sort(probabilities, dim=1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=1)

    if reg:
        ranks = torch.arange(1, C + 1, device=probabilities.device, dtype=probabilities.dtype).expand_as(cum_probs)
        penalty = torch.clamp(ranks - float(k_reg), min=0.0) * float(lambda_reg)
        adj_cum = cum_probs + penalty
    else:
        adj_cum = cum_probs

    keep_sorted = adj_cum <= tau

    prediction_sets = torch.zeros_like(keep_sorted, dtype=torch.bool)
    prediction_sets.scatter_(1, pi, keep_sorted)

    return prediction_sets



