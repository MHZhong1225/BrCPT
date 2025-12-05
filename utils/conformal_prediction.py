# conformal_prediction.py
import os
import random

from typing import Optional, Callable, Any
import torch
import numpy as np
# Define type hints for clarity
_QuantileFn = Callable[[Any, float], float]
_CalibrateFn = Callable[[torch.Tensor, torch.Tensor, Any], Any]
_PredictFn = Callable[[torch.Tensor, Any, Any], torch.Tensor]

SEED = 999
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

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

# def calibrate_aps(probs: torch.Tensor, labels: torch.Tensor, alpha: float=0.1) -> float:
#     with torch.no_grad():
#         sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
#         cumsums = torch.cumsum(sorted_probs, dim=1)
#         N = probs.size(0)
#         pos = (sorted_idx == labels.unsqueeze(1)).nonzero(as_tuple=False)
#         pos_j = torch.zeros(N, dtype=torch.long, device=probs.device)
#         pos_j[pos[:, 0]] = pos[:, 1]
#         C_true = cumsums[torch.arange(N, device=probs.device), pos_j]
#         # qhat at 1 - alpha
#         n = C_true.numel()
#         k = int(torch.ceil(torch.tensor((n + 1) * (1 - alpha), device=C_true.device)).item())
#         k = max(1, min(k, n))
#         qhat = torch.kthvalue(C_true, k).values.item()
#         return float(qhat)

@torch.no_grad()
def predict_aps(probs: torch.Tensor,
                qhat: float,
                top1_guarantee: bool = True) -> torch.Tensor:
    """
    Randomized APS prediction (与上面的 calibrate_aps 一致):

    对每个样本:
      1) 按概率排序，得到 sorted_probs, sorted_idx, cumsums;
      2) 所有满足 C_full <= qhat 的类全部加入 (确定加入区间);
      3) 若存在唯一 L 使得 C_prev_L < qhat < C_full_L:
           以概率 (qhat - C_prev_L) / p_L 随机决定是否加入该边界类;
      4) 其它类不加入;
      5) 可选 top-1 guarantee：如果集合为空，则加入 top-1 类别。
    """
    N, K = probs.shape
    # 1) 排序 & 累积
    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)  # [N,K]
    cumsums = torch.cumsum(sorted_probs, dim=1)                            # [N,K]
    C_prev = cumsums - sorted_probs                                       # [N,K]

    # 2) 确定性包含：C_full <= qhat 的全部保留
    keep_sorted = (cumsums <= qhat)   # [N,K]

    # 3) 边界类：C_prev < qhat < C_full
    #    注意：理论上每行最多一个 True（可能没有）
    boundary_mask = (C_prev < qhat) & (cumsums > qhat)   # [N,K]
    if boundary_mask.any():
        rows, cols = boundary_mask.nonzero(as_tuple=True)  # 边界位置索引
        # (qhat - C_prev) / p_L
        numer = (qhat - C_prev[rows, cols]).clamp(min=0.0)
        denom = sorted_probs[rows, cols].clamp(min=1e-12)
        prob_include = (numer / denom).clamp(0.0, 1.0)

        U = torch.rand_like(prob_include)
        take = (U <= prob_include)

        keep_sorted[rows[take], cols[take]] = True

    # 4) 回到原始类别顺序
    keep = torch.zeros_like(keep_sorted, dtype=torch.bool)
    keep.scatter_(1, sorted_idx, keep_sorted)

    # 5) top-1 保证（可选）
    if top1_guarantee:
        empty = ~keep.any(dim=1)
        if empty.any():
            top1 = probs.argmax(dim=1)
            keep[empty, top1[empty]] = True

    return keep
from utils import conformal_prediction as cp 
@torch.no_grad()
def calibrate_aps(probs: torch.Tensor,
                  labels: torch.Tensor,
                  alpha: float = 0.1) -> float:
    """
    Randomized APS calibration.

    对每个校准样本 i:
      1) 将类别按概率从大到小排序:
           sorted_probs[i], sorted_idx[i]
      2) 计算累积和 cumsums[i]
      3) 找到真实类别在排序中的位置 j_i:
           sorted_idx[i,j_i] == labels[i]
      4) 令:
         C_prev_i = sum_{k < j_i} p_{i,(k)}   (前缀和)
         p_true_i = p_{i,(j_i)}
         U_i ~ Unif[0,1]
         score_i = C_prev_i + U_i * p_true_i
    最后对 {score_i} 取 (1 - alpha) 的 conformal 分位数，得到 qhat.
    """
    # 1) 排序
    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)  # [N,K]
    cumsums = torch.cumsum(sorted_probs, dim=1)                           # [N,K]

    N = probs.size(0)
    # 2) 真实类别 mask
    true_mask = (sorted_idx == labels.unsqueeze(1))  # [N,K]，每行一个 True

    # 3) 前缀和 C_prev = cumsums - sorted_probs
    C_prev_all = cumsums - sorted_probs             # [N,K]
    C_prev = C_prev_all[true_mask]                  # [N]
    p_true = sorted_probs[true_mask]                # [N]

    # 4) 随机化分数: C_prev + U * p_true
    U = torch.rand(N, device=probs.device)
    scores = C_prev + U * p_true                    # [N]

    # 5) conformal 分位数 (用你项目里统一的 quantile 函数)
    qhat = cp.conformal_quantile(scores, alpha=alpha)
    return float(qhat)

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



