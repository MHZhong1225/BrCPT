"""
PyTorch implementation of Smooth Conformal Prediction (threshold-based and APS)
-------------------------------------------------------------------------------
This module provides a lightweight, differentiable alternative to the JAX-based
reference that uses variational sorting networks. It relies on `torchsort`
for soft ranks and uses pairwise smooth comparisons to avoid explicit
permutation matrices. The API mirrors (at a high level) the functionality of
`cp.predict_threshold`, `cp.calibrate_threshold`, and APS-style set prediction
with smooth relaxations.

Key functions
-------------
- soft_quantile(tensor, q, regularization_strength): differentiable quantile
- smooth_calibrate_threshold(probabilities, labels, alpha): returns tau
- smooth_predict_threshold(probabilities, tau, temperature): soft sets
- smooth_calibrate_aps(probabilities, labels, alpha, ...): returns tau for APS
- smooth_predict_aps(probabilities, tau, ...): soft APS sets

*With `_with_checks` variants for input validation*

Notes
-----
- Threshold-based calibration chooses tau as the `alpha`-quantile of TRUE-CLASS
  probabilities (so that ~1-alpha fraction exceed tau).
- APS calibration chooses tau as the (1 - alpha)-quantile of the per-example
  \"entry threshold\" — the cumulative probability mass of classes with higher
  probability than the true class (using a smooth comparator). Intuitively, the
  true class enters the set once the running sum (from highest to lowest) passes
  that value.

"""
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import sigmoid
try:
    # torchsort >= 0.1.9 provides soft_rank; soft_sort is optional here
    from torchsort import soft_rank  # type: ignore
except Exception as e:
    raise ImportError("This module requires `torchsort`. Install via `pip install torchsort`.\n" + str(e))

# -------------------------------
# Utilities & Validation
# -------------------------------

def _ensure_2d_probs(probabilities: Tensor) -> None:
    if probabilities.ndim != 2:
        raise ValueError(f"`probabilities` must be a 2D tensor [N, C], got shape {tuple(probabilities.shape)}")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("`probabilities` must have non-zero batch and class dimension")
    if not torch.is_floating_point(probabilities):
        raise ValueError("`probabilities` must be a floating tensor")
    if torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("`probabilities` entries must be in [0, 1]")


def _ensure_labels(labels: Tensor, num_classes: int) -> None:
    if labels.ndim != 1:
        raise ValueError("`labels` must be a 1D tensor of class indices [N]")
    if labels.shape[0] == 0:
        raise ValueError("`labels` must be non-empty")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("`labels` must be an integer tensor of class indices")
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("`labels` must be in [0, C-1]")


def _ensure_alpha(alpha: float) -> None:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("`alpha` must be in (0, 1)")


def _ensure_tau(tau: Tensor) -> None:
    if tau.ndim > 1:
        raise ValueError("`tau` must be a scalar or 1D tensor (per-example)")


def _as_batch_tau(tau: Tensor, batch_size: int) -> Tensor:
    if tau.ndim == 0:
        return tau.view(1, 1).expand(batch_size, 1)
    elif tau.ndim == 1:
        if tau.shape[0] not in (1, batch_size):
            raise ValueError("Per-example `tau` must have shape [1] or [N]")
        return tau.view(-1, 1).expand(batch_size, 1)
    else:
        raise ValueError("`tau` must be a scalar or 1D tensor")


# -------------------------------
# Differentiable Quantile
# -------------------------------
_SmoothQuantileFn = Callable[[Tensor, float, float], Tensor]


def soft_quantile(tensor: Tensor, q: float, regularization_strength: float = 0.1) -> Tensor:
    """Differentiable quantile using soft ranks as weights.

    Args:
        tensor: 1D tensor [N]. Requires gradients if used in training.
        q: quantile level in [0, 1]. E.g., 0.1 for 10th percentile.
        regularization_strength: softness for soft ranks (smaller = sharper).

    Returns:
        Scalar tensor: differentiable quantile approximation.
    """
    if tensor.ndim != 1:
        raise ValueError("`tensor` must be 1D for quantile computation")
    if not (0.0 <= q <= 1.0):
        raise ValueError("`q` must be in [0, 1]")

    # Compute differentiable ranks (1..N approximately)
    sr = soft_rank(tensor.unsqueeze(0), regularization_strength=regularization_strength).squeeze(0)
    n = tensor.shape[0]
    target_rank = q * (n - 1)

    # Gaussian weights around target rank
    # bandwidth 0.5 works well empirically; can be exposed if needed
    weights = torch.exp(-((sr - target_rank) ** 2) / (2 * 0.5 ** 2))
    weights = weights / (weights.sum() + 1e-12)
    return (weights * tensor).sum()


def smooth_conformal_quantile_with_checks(scores: Tensor, q: float, regularization_strength: float = 0.1) -> Tensor:
    if scores.ndim != 1:
        raise ValueError("`scores` must be a 1D tensor")
    if scores.shape[0] == 0:
        raise ValueError("`scores` must be non-empty")
    if not torch.is_floating_point(scores):
        raise ValueError("`scores` must be a floating tensor")
    if not (0.0 <= q <= 1.0):
        raise ValueError("`q` must be in [0, 1]")
    if regularization_strength <= 0:
        raise ValueError("`regularization_strength` must be > 0")
    return soft_quantile(scores, q=q, regularization_strength=regularization_strength)


def smooth_conformal_quantile(scores: Tensor, q: float, regularization_strength: float = 0.1) -> Tensor:
    return soft_quantile(scores, q=q, regularization_strength=regularization_strength)


# -------------------------------
# Threshold-based Conformal (Calibration + Prediction)
# -------------------------------

def _true_class_probs(probabilities: Tensor, labels: Tensor) -> Tensor:
    """Gather p_true for each example."""
    _ensure_2d_probs(probabilities)
    _ensure_labels(labels, probabilities.shape[1])
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Batch size mismatch between probabilities and labels")
    return probabilities.gather(dim=1, index=labels.view(-1, 1)).squeeze(1)


def smooth_calibrate_threshold_with_checks(
    probabilities: Tensor,
    labels: Tensor,
    alpha: float,
    regularization_strength: float = 0.1,
) -> Tensor:
    _ensure_2d_probs(probabilities)
    _ensure_labels(labels, probabilities.shape[1])
    _ensure_alpha(alpha)
    p_true = _true_class_probs(probabilities, labels)
    # tau = alpha-quantile of p_true (so that ~1-alpha exceed tau)
    return smooth_conformal_quantile_with_checks(p_true, q=float(alpha), regularization_strength=regularization_strength)


def smooth_calibrate_threshold(
    probabilities: Tensor,
    labels: Tensor,
    alpha: float,
    regularization_strength: float = 0.1,
) -> Tensor:
    p_true = _true_class_probs(probabilities, labels)
    return smooth_conformal_quantile(p_true, q=float(alpha), regularization_strength=regularization_strength)


def smooth_predict_threshold_with_checks(
    probabilities: Tensor,
    tau: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    _ensure_2d_probs(probabilities)
    if temperature <= 0:
        raise ValueError("`temperature` must be > 0")
    _ensure_tau(tau)
    tau_b = _as_batch_tau(tau, probabilities.shape[0])
    # Soft inclusion probabilities
    return torch.sigmoid((probabilities - tau_b) / temperature)


def smooth_predict_threshold(
    probabilities: Tensor,
    tau: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    tau_b = _as_batch_tau(tau, probabilities.shape[0])
    return torch.sigmoid((probabilities - tau_b) / temperature)


# -------------------------------
# APS (Adaptive Prediction Sets): Calibration + Prediction
# -------------------------------
# We avoid explicit soft permutations by using pairwise smooth comparisons.
# For a given example and class i with probability p_i, define
#     cum_before_i = sum_{j} sigma((p_j - p_i)/beta) * p_j
# which approximates the cumulative mass of classes strictly larger than p_i.
# A class is included when cum_before_i < tau.


def _pairwise_cum_before(probabilities: Tensor, beta: float = 0.1) -> Tensor:
    """Compute smooth cumulative mass before each class using pairwise comps.

    Args:
        probabilities: [N, C]
        beta: temperature for pairwise comparator; smaller -> sharper ordering.

    Returns:
        Tensor [N, C] with cumulative mass excluding the class itself (expected
        sum of probabilities of classes that rank above it).
    """
    _ensure_2d_probs(probabilities)
    if beta <= 0:
        raise ValueError("`beta` must be > 0")
    # p_ij comparator: prob that j is ranked above i
    p_i = probabilities.unsqueeze(2)      # [N, C, 1]
    p_j = probabilities.unsqueeze(1)      # [N, 1, C]
    above = torch.sigmoid((p_j - p_i) / beta)  # [N, C, C]
    cum_before = (above * p_j).sum(dim=-1)     # [N, C]
    return cum_before


def smooth_predict_aps_with_checks(
    probabilities: Tensor,
    tau: Tensor,
    beta: float = 0.1,
    dispersion: float = 0.1,
) -> Tensor:
    _ensure_2d_probs(probabilities)
    if dispersion <= 0:
        raise ValueError("`dispersion` must be > 0")
    if beta <= 0:
        raise ValueError("`beta` must be > 0")
    _ensure_tau(tau)
    tau_b = _as_batch_tau(tau, probabilities.shape[0])
    cum_before = _pairwise_cum_before(probabilities, beta=beta)
    # inclusion when cum_before < tau  -> sigmoid((tau - cum_before)/dispersion)
    return torch.sigmoid((tau_b - cum_before) / dispersion)


def smooth_predict_aps(
    probabilities: Tensor,
    tau: Tensor,
    beta: float = 0.1,
    dispersion: float = 0.1,
) -> Tensor:
    tau_b = _as_batch_tau(tau, probabilities.shape[0])
    cum_before = _pairwise_cum_before(probabilities, beta=beta)
    return torch.sigmoid((tau_b - cum_before) / dispersion)


def _entry_thresholds_for_true_labels(probabilities: Tensor, labels: Tensor, beta: float = 0.1) -> Tensor:
    """Per-example smooth entry thresholds for the true class.

    The minimal tau such that the true class would be included by APS is the
    cumulative mass of higher-probability classes. We compute a smooth version
    via pairwise comparisons.
    """
    _ensure_2d_probs(probabilities)
    _ensure_labels(labels, probabilities.shape[1])
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Batch size mismatch between probabilities and labels")
    cum_before = _pairwise_cum_before(probabilities, beta=beta)  # [N, C]
    entry = cum_before.gather(dim=1, index=labels.view(-1, 1)).squeeze(1)  # [N]
    return entry


def smooth_calibrate_aps_with_checks(
    probabilities: Tensor,
    labels: Tensor,
    alpha: float,
    beta: float = 0.1,
    regularization_strength: float = 0.1,
) -> Tensor:
    _ensure_2d_probs(probabilities)
    _ensure_labels(labels, probabilities.shape[1])
    _ensure_alpha(alpha)
    if beta <= 0:
        raise ValueError("`beta` must be > 0")
    entry = _entry_thresholds_for_true_labels(probabilities, labels, beta=beta)
    # Choose tau as (1 - alpha)-quantile of entry thresholds so that
    # P(entry <= tau) ≈ 1 - alpha (i.e., true class included with that prob).
    q = 1.0 - float(alpha)
    return smooth_conformal_quantile_with_checks(entry, q=q, regularization_strength=regularization_strength)


def smooth_calibrate_aps(
    probabilities: Tensor,
    labels: Tensor,
    alpha: float,
    beta: float = 0.1,
    regularization_strength: float = 0.1,
) -> Tensor:
    entry = _entry_thresholds_for_true_labels(probabilities, labels, beta=beta)
    q = 1.0 - float(alpha)
    return smooth_conformal_quantile(entry, q=q, regularization_strength=regularization_strength)


# -------------------------------
# Minimal self-tests (gradient flow + sanity checks)
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    N, C = 64, 10
    ALPHA = 0.1
    TEMP = 0.05
    BETA = 0.1
    REG = 0.01

    probs = torch.rand(N, C, requires_grad=True)
    probs = probs / probs.sum(dim=1, keepdim=True)
    probs.retain_grad()

    labels = torch.randint(low=0, high=C, size=(N,))

    # --- Threshold calibration ---
    tau_thr = smooth_calibrate_threshold(probs, labels, alpha=ALPHA, regularization_strength=REG)
    print("[Threshold] tau:", float(tau_thr))

    sets_thr = smooth_predict_threshold(probs, tau_thr.detach(), temperature=TEMP)
    print("[Threshold] soft set shape:", tuple(sets_thr.shape))

    loss_thr = sets_thr.gather(1, labels.view(-1, 1)).mean()
    loss_thr.backward(retain_graph=True)
    assert probs.grad is not None
    print("[Threshold] Backward OK")

    # --- APS calibration ---
    tau_aps = smooth_calibrate_aps(probs, labels, alpha=ALPHA, beta=BETA, regularization_strength=REG)
    print("[APS] tau:", float(tau_aps))

    sets_aps = smooth_predict_aps(probs, tau_aps.detach(), beta=BETA, dispersion=TEMP)
    print("[APS] soft set shape:", tuple(sets_aps.shape))

    loss_aps = sets_aps.gather(1, labels.view(-1, 1)).mean()
    probs.grad.zero_()
    loss_aps.backward()
    assert probs.grad is not None
    print("[APS] Backward OK")

    print("All tests passed.")
