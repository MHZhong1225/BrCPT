# train_conformal.py (PyTorch Version)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Any, Tuple
import numpy as np

# Import from our refactored PyTorch files
from utils import models
from utils import data
import smooth_conformal_prediction as scp
from utils import train_utils as cputils
from train_normal import evaluate # We can reuse the standard evaluation function

def train_one_epoch_conformal(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    config: Dict[str, Any],
    device: torch.device
) -> float:
    """
    Performs one full training pass using the Conformal Training objective.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): The DataLoader for the training data.
        optimizer (optim.Optimizer): The optimizer for updating model weights.
        config (Dict[str, Any]): A dictionary containing conformal training hyperparameters.
        device (torch.device): The device (CPU or GPU) to perform training on.

    Returns:
        float: The average total loss for the epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Conformal Training")
    
    # Extract conformal config parameters for easy access
    conf_config = config['conformal']
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # --- Core Conformal Training Logic ---
        
        # 1. Forward pass to get scores
        outputs = model(inputs)
        # We use log_softmax as scores, which is often more numerically stable
        scores = torch.log_softmax(outputs, dim=1)

        # 2. Split the batch into calibration and prediction sets
        batch_size = inputs.shape[0]
        cal_split_size = int(batch_size * conf_config['fraction'])
        
        # Ensure we have data for both splits
        if cal_split_size == 0 or cal_split_size == batch_size:
            # This can happen if the last batch is too small. Skip for simplicity.
            continue

        cal_scores, pred_scores = scores[:cal_split_size], scores[cal_split_size:]
        cal_labels, pred_labels = labels[:cal_split_size], labels[cal_split_size:]

        # 3. Differentiable Calibration on the calibration set
        # Get conformity scores for the true classes
        # NOTE: The original paper may use a different definition of conformity score.
        # This implementation uses the negative log-probability of the true class.
        cal_conformity_scores = -cal_scores[torch.arange(cal_split_size), cal_labels]
        
        # Compute the smooth (differentiable) quantile to get the threshold tau
        tau = scp.smooth_conformal_quantile(
            cal_conformity_scores, 
            alpha=conf_config['alpha'],
            # temperature=conf_config.get('regularization_strength', 0.01) # Use temperature for our implementation
        )
        
        # 4. Differentiable Prediction on the prediction set
        # Get the soft (differentiable) confidence sets. 
        # Here, scores are log-probs, so a higher score is better.
        # We need to adjust the logic based on the score definition.
        # Let's assume the score is conformity score (e.g., -log_softmax)
        pred_conformity_scores = -pred_scores
        
        smooth_sets = scp.smooth_predict_threshold(
            pred_conformity_scores, # Should be conformity scores, not raw model scores
            tau, 
            temperature=conf_config['temperature']
        )
        
        # 5. Compute the conformal loss on the prediction set
        # a) Coverage Loss
        loss_matrix = torch.eye(outputs.shape[1], device=device) # Simple identity matrix for now
        coverage_loss = cputils.compute_coverage_loss(smooth_sets, pred_labels, loss_matrix)
        
        # b) Size Loss
        size_loss = cputils.compute_size_loss(smooth_sets, target_size=1) # 'valid' loss
        
        # c) Combine losses
        # The log-transform is used for stability, as in the original paper
        conformal_loss = torch.log(coverage_loss + conf_config['size_weight'] * size_loss + 1e-8)
        
        # Add optional standard cross-entropy and weight decay
        ce_loss = nn.CrossEntropyLoss()(outputs, labels) * conf_config['cross_entropy_weight']
        
        total_loss = conformal_loss + ce_loss
        
        # 6. Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item() * inputs.size(0)
        progress_bar.set_postfix(loss=total_loss.item())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def run_conformal_training(config: Dict[str, Any]):
    """
    The main function to run the Conformal Training pipeline.
    """
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu")
    print(f"Using device: {device}")
    
    # Extract nested training config
    train_config = config['training']

    # --- 2. Data ---
    data_info = data.get_dataloaders(
        dataset_path=config['dataset_path'],
        batch_size=train_config['batch_size'],
        mean=config['mean'],
        std=config['std']
    )
    dataloaders = data_info['dataloaders']
    num_classes = len(data_info['class_names'])

    # --- 3. Model ---
    model = models.create_model(num_classes=num_classes, pretrained=config['model']['pretrained'])
    model.to(device)

    # --- 4. Optimizer ---
    optimizer = optim.SGD(
        model.parameters(), 
        lr=train_config['learning_rate'], 
        momentum=train_config['momentum'],
        weight_decay=train_config['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # --- 5. Training Loop ---
    print("\n--- Starting Conformal Training ---")
    for epoch in range(train_config['epochs']):
        print(f"\nEpoch {epoch+1}/{train_config['epochs']}")
        
        train_loss = train_one_epoch_conformal(model, dataloaders['train'], optimizer, config, device)
        print(f"Epoch {epoch+1} Conformal Train Loss: {train_loss:.4f}")
        
        # We can reuse the standard evaluation function as it only needs the model's final outputs
        val_loss, val_acc = evaluate(model, dataloaders['val'], nn.CrossEntropyLoss(), device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        scheduler.step()

    print("\n--- Conformal Training Finished ---")
    
    # --- 6. Final Evaluation on Test Set ---
    test_loss, test_acc = evaluate(model, dataloaders['test'], nn.CrossEntropyLoss(), device)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")

    # --- 7. Save the trained model ---
    save_path = os.path.join(config['output_dir'], 'conformal_model.pth')
    os.makedirs(config['output_dir'], exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    # --- This is a test script to run a full conformal training cycle ---
    
    # This mock_config now matches the nested structure of the main config.py
    mock_config = {
        'dataset_path': './BRACS_Rol/latest_version', # IMPORTANT: Update this path
        'output_dir': './experiments/conformal_test',
        'device': 'cuda',
        'mean': (0.5, 0.5, 0.5), # Placeholder
        'std': (0.5, 0.5, 0.5),  # Placeholder
        'model': {'pretrained': True},
        'training': {
            'batch_size': 16,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'epochs': 5,
        },
        'conformal': { # Conformal training specific hyperparameters
            'alpha': 0.1,
            'fraction': 0.5, # 50% for calibration, 50% for prediction
            'temperature': 0.1,
            'regularization_strength': 0.01,
            'size_weight': 0.5,
            'cross_entropy_weight': 0.1 # Can add a small amount of CE loss for stability
        }
    }

    if os.path.exists(mock_config['dataset_path']):
        run_conformal_training(mock_config)
    else:
        print(f"Test path '{mock_config['dataset_path']}' does not exist. Please update the path to run the test.")
