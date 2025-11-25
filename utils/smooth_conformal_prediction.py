# utils/smooth_conformal_prediction.py

import torch
import torchsort

def soft_quantile(tensor: torch.Tensor, q: float, regularization_strength: float = 1e-2):
    """
    Differentiable approximation of the q-quantile of a 1D tensor using soft-ranks.
    torchsort.soft_rank expects a 2D tensor [B, N], so we unsqueeze to [1, N].
    """
    # flatten to 1D first
    if tensor.dim() != 1:
        tensor = tensor.reshape(-1)

    n = tensor.numel()
    if n == 0:
        raise ValueError("soft_quantile got empty tensor")

    # torchsort requires [B, N]
    values_2d = tensor.unsqueeze(0)  # [1, n]
    soft_ranks = torchsort.soft_rank(values_2d, regularization_strength=regularization_strength)
    soft_ranks = soft_ranks.squeeze(0)  # back to [n]

    target_rank = q * (n - 1)

    weights = torch.exp(- (soft_ranks - target_rank) ** 2 / (2.0 * (regularization_strength ** 2)))
    weights = weights / (weights.sum() + 1e-12)

    return torch.sum(weights * tensor)


def smooth_conformal_quantile(conformity_scores: torch.Tensor, alpha: float, regularization_strength: float = 1e-2):
    """
    For conformity scores c (higher is better), return a differentiable threshold tau such that
    P(c >= tau) ≈ 1 - alpha. tau is the LOWER alpha-quantile of conformity.
    """
    if conformity_scores.dim() != 1:
        conformity_scores = conformity_scores.reshape(-1)
    n = conformity_scores.numel()
    level = (n + 1.0) * alpha / max(n, 1.0)
    level = min(level, 1.0)
    return soft_quantile(conformity_scores, q=level, regularization_strength=regularization_strength)


def smooth_conformal_quantile_nonconformity(nonconformity_scores: torch.Tensor, alpha: float, regularization_strength: float = 1e-2):
    """
    For nonconformity scores s (higher is worse), return differentiable qhat such that
    P(s <= qhat) ≈ 1 - alpha, i.e. qhat is the UPPER (1-alpha)-quantile.
    """
    if nonconformity_scores.dim() != 1:
        nonconformity_scores = nonconformity_scores.reshape(-1)
    n = nonconformity_scores.numel()
    level = (n + 1.0) * (1.0 - alpha) / max(n, 1.0)
    level = min(level, 1.0)
    return soft_quantile(nonconformity_scores, q=level, regularization_strength=regularization_strength)


def smooth_predict_threshold(scores: torch.Tensor, tau: torch.Tensor, temperature: float = 0.1):
    """
    Differentiable prediction set indicator:
       soft_set = sigmoid((scores - tau)/T)
    scores: [B, K]
    tau: scalar or [B,1] or [B,K] broadcastable to scores
    """
    return torch.sigmoid((scores - tau) / temperature)