# train_utils.py (PyTorch Version)

import torch
import torch.nn as nn
import numpy as np

import copy

class EarlyStopping:
    """在验证集损失不再改善时提前停止训练。"""
    def __init__(self, patience=7, verbose=False, delta=5e-4, path='checkpoint.pth'):
        """
        Args:
            patience (int): 在上次验证集损失改善后，等待多少个epoch。
            verbose (bool): 如果为True，则为每次验证集损失改善打印一条信息。
            delta (float):  被认为是改善的最小变化量。
            path (str):     保存最佳模型的路径。
        """
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
        """当验证集损失下降时，保存模型。"""
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


if __name__ == '__main__':
    # --- This is a test script to verify the loss functions ---
    
    BATCH_SIZE = 4
    NUM_CLASSES = 7

    # --- 1. Test compute_size_loss ---
    print("\n--- Testing compute_size_loss ---")
    
    # Create dummy confidence sets that require gradients
    dummy_sets = torch.rand(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
    
    # Calculate loss
    size_loss = compute_size_loss(dummy_sets, target_size=1)
    
    print(f"Calculated size loss: {size_loss.item()}")
    assert size_loss >= 0, "Size loss must be non-negative."
    
    # Verify backward pass
    size_loss.backward()
    assert dummy_sets.grad is not None, "Gradients are not flowing through compute_size_loss!"
    print("Backward pass successful for compute_size_loss.")

    # --- 2. Test compute_coverage_loss ---
    print("\n--- Testing compute_coverage_loss ---")

    # Create new dummy data
    dummy_sets_2 = torch.rand(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
    dummy_labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    
    # Use a simple identity matrix for the loss matrix (only penalizes missing the true label)
    dummy_loss_matrix = torch.eye(NUM_CLASSES)
    
    # Calculate loss
    coverage_loss = compute_coverage_loss(dummy_sets_2, dummy_labels, dummy_loss_matrix)
    
    print(f"Calculated coverage loss: {coverage_loss.item()}")
    
    # Verify backward pass
    coverage_loss.backward()
    assert dummy_sets_2.grad is not None, "Gradients are not flowing through compute_coverage_loss!"
    print("Backward pass successful for compute_coverage_loss.")

    print("\nAll loss function tests passed!")
