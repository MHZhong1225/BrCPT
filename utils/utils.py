# utils.py (PyTorch Version)

import torch
import torch.nn as nn
import numpy as np

import copy
from sklearn.metrics import f1_score
from utils import conformal_prediction as cp

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=5e-4, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.best_model_state_dict = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model_state_dict = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss

def compute_size_loss(
    confidence_sets: torch.Tensor,
    target_size: int = 1,
    weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes the hinge size loss for the confidence sets.

    This loss penalizes confidence sets that are larger than the target size.
    It's the core component for encouraging smaller, more efficient prediction sets.

    Args:
        confidence_sets (torch.Tensor): The soft prediction sets (probabilities of inclusion) 
                                        from the model. Shape: [batch_size, num_classes].
        target_size (int): The desired target size for the confidence sets. 
                           '1' (for 'valid' loss) encourages sets of size 1 or less.
                           '0' (for 'normal' loss) encourages empty sets (used in combination
                           with a coverage loss).
        weights (torch.Tensor, optional): Per-example weights to apply to the loss.
                                           Shape: [batch_size]. Defaults to None.

    Returns:
        torch.Tensor: A scalar tensor representing the mean size loss for the batch.
    """
    # Sum the probabilities to get the expected size of each set
    sizes = torch.sum(confidence_sets, dim=1)
    
    # Calculate the hinge loss: max(0, size - target_size)
    loss_per_example = torch.relu(sizes - target_size)
    
    # Apply per-example weights if provided
    if weights is not None:
        loss_per_example = weights * loss_per_example
        
    return torch.mean(loss_per_example)


def compute_coverage_loss(
    confidence_sets: torch.Tensor,
    labels: torch.Tensor,
    loss_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Computes a general classification loss on the confidence sets.

    This loss ensures that the true label is included in the set, and optionally
    penalizes the inclusion of other specific labels based on the loss_matrix.

    Args:
        confidence_sets (torch.Tensor): The soft prediction sets. Shape: [batch_size, num_classes].
        labels (torch.Tensor): The ground truth labels. Shape: [batch_size].
        loss_matrix (torch.Tensor): A matrix defining the penalties. 
                                    Shape: [num_classes, num_classes].

    Returns:
        torch.Tensor: A scalar tensor representing the mean coverage loss for the batch.
    """
    num_classes = confidence_sets.shape[1]
    
    # Create one-hot encoded labels
    one_hot_labels = nn.functional.one_hot(labels, num_classes=num_classes)
    
    # Get the row from the loss matrix corresponding to each true label in the batch
    batch_loss_matrix = loss_matrix[labels]
    
    # Penalty for not including the true class (the `on-diagonal` part of the loss)
    # loss = (1 - confidence_set_prob_for_true_label) * penalty_for_missing_true_label
    loss1 = (1 - confidence_sets) * one_hot_labels * batch_loss_matrix
    
    # Penalty for including incorrect classes (the `off-diagonal` part of the loss)
    # loss = confidence_set_prob_for_wrong_label * penalty_for_including_wrong_label
    loss2 = confidence_sets * (1 - one_hot_labels) * batch_loss_matrix
    
    # The total loss for each example is the sum of these penalties
    loss_per_example = torch.sum(loss1 + loss2, dim=1)
    
    return torch.mean(loss_per_example)




def get_predictions(model, dataloader, device, return_h=False):
    model.eval()
    all_probs, all_labels, all_h = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                logits, h_x = outputs[0], outputs[1]
            else:
                logits, h_x = outputs, None
            
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            if return_h and h_x is not None:
                if h_x.dim() == 1: h_x = h_x.unsqueeze(1)
                all_h.append(h_x.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    h_all = torch.cat(all_h) if return_h else None
    return probs, labels, h_all

def final_evaluation(model, dls, device, config):
    print("\n[Final Eval] Starting Conformal Evaluation...")
    alpha = float(config['conformal']['alpha'])
    
    # 1. data
    cal_loader = dls['val']
    test_loader = dls['test']

    cal_probs, cal_labels, cal_h = get_predictions(model, cal_loader, device, return_h=True)
    test_probs, test_labels, test_h = get_predictions(model, test_loader, device, return_h=True)

    # 3. Calibration Step
    cal_h_val = cal_h.squeeze(1)
    cal_p_true = cal_probs[torch.arange(len(cal_labels)), cal_labels]
    s_cal = torch.relu(cal_h_val - cal_p_true)

    # 4. Q-hat (Hard Quantile)
    qhat = cp.conformal_quantile(s_cal, alpha=alpha)
    print(f"[Final Eval] Calibrated qhat (alpha={alpha}): {qhat:.4f}")

    # 5. Prediction Step
    test_h_val = test_h.squeeze(1)
    test_thresholds = test_h_val - qhat
    
    #  {y | p(y) >= h(x) - qhat}
    prediction_sets = (test_probs >= test_thresholds.unsqueeze(1))

    empty_mask = ~prediction_sets.any(dim=1)
    if empty_mask.any():
        top1_preds = torch.argmax(test_probs, dim=1)
        prediction_sets[empty_mask, top1_preds[empty_mask]] = True

    # 6. matix
    covered = prediction_sets[torch.arange(len(test_labels)), test_labels].float()
    coverage = covered.mean().item()
    avg_set_size = torch.sum(prediction_sets, dim=1).float().mean().item()
    
    # Accuracy / F1
    preds = torch.argmax(test_probs, dim=1)
    acc = (preds == test_labels).sum().item() / len(test_labels)
    f1_macro = f1_score(test_labels.numpy(), preds.numpy(), average='macro')

    print(f"[Final Eval] Result -> Acc: {acc:.4f}, Cov: {coverage:.4f}, Size: {avg_set_size:.4f}")
    
    return {
        "Test/Accuracy": acc,
        "Test/Coverage": coverage,
        "Test/Avg_Set_Size": avg_set_size,
        "Test/F1_Macro": f1_macro,
        "Final/Q_hat": qhat
    }
