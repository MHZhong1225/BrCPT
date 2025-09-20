# train_conformal.py (PyTorch Version)
import os, random, numpy as np, torch

SEED = 2024
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)


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
from utils import smooth_conformal_prediction as scp
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
        # scores = torch.log_softmax(outputs, dim=1)
        scores = torch.softmax(outputs, dim=1)
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
        # cal_conformity_scores = -cal_scores[torch.arange(cal_split_size), cal_labels]

        cal_conformity_scores = cal_scores[torch.arange(cal_split_size), cal_labels]
        
        # Compute the smooth (differentiable) quantile to get the threshold tau
        tau = scp.smooth_conformal_quantile(
            cal_conformity_scores, 
            alpha=conf_config['alpha'],
            # temperature=conf_config.get('regularization_strength', 0.01) # Use temperature for our implementation
            regularization_strength=conf_config['regularization_strength']
        )
        
        # 4. Differentiable Prediction on the prediction set
        # Get the soft (differentiable) confidence sets. 
        # Here, scores are log-probs, so a higher score is better.
        # We need to adjust the logic based on the score definition.
        # Let's assume the score is conformity score (e.g., -log_softmax)
        pred_conformity_scores = -pred_scores
        
        smooth_sets = scp.smooth_predict_threshold(
            # pred_conformity_scores, # Should be conformity scores, not raw model scores
            pred_scores,
            tau, 
            temperature=conf_config['temperature']
        )
        
        # 5. Compute the conformal loss on the prediction set
        # a) Coverage Loss
        loss_matrix = torch.eye(outputs.shape[1], device=device) # Simple identity matrix for now
        coverage_loss = cputils.compute_coverage_loss(smooth_sets, pred_labels, loss_matrix)
        
        # b) Size Loss
        size_loss = cputils.compute_size_loss(smooth_sets, target_size=1) # 'valid' loss
        
        # # c) Combine losses
        # # The log-transform is used for stability, as in the original paper
        # conformal_loss = torch.log(coverage_loss + conf_config['size_weight'] * size_loss + 1e-8)
        
        # # Add optional standard cross-entropy and weight decay
        # ce_loss = nn.CrossEntropyLoss()(outputs, labels) * conf_config['cross_entropy_weight']
        # total_loss = conformal_loss + ce_loss
        
        
        # --- STABILITY FIX: Use a Stable Linear Combination of Losses ---
        # Remove the unstable log transform.
        # Add a standard Cross-Entropy loss as a stabilizer.
        ce_loss = nn.CrossEntropyLoss()(outputs, labels)
        
        total_loss = (
            coverage_loss + 
            conf_config['size_weight'] * size_loss +
            conf_config['cross_entropy_weight'] * ce_loss
        )

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
    device = config['device']
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
    model = models.get_model(
        model_type='standard',
        backbone_name=config['model']['name'], # 从config中读取backbone名称
        num_classes=num_classes,
        pretrained=config['model']['pretrained'])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("--- Model Parameters ---")
    print(f"Total Parameters:     {total_params / (1024*1024):.2f}M")
    print(f"Trainable Parameters: {trainable_params_count/ (1024*1024):.2f}M")
    print("------------------------")

    # --- 4. Optimizer ---
    optimizer = optim.SGD(
        model.parameters(), 
        lr=train_config['learning_rate'], 
        momentum=train_config['momentum'],
        weight_decay=train_config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model_p = f"model_best.pth"
    save_dir = os.path.join(config['output_dir'], config['model']['name'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_p)
    early_stopping = cputils.EarlyStopping(patience=15, verbose=True, path=save_path)

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

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break # 跳出训练循环
    print("\n--- Conformal Training Finished ---")

    print(f"Loading best model from epoch with val_loss: {early_stopping.val_loss_min:.4f}")
    state_dict = early_stopping.best_model_state_dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2) 直接用原模型评测（evaluate 里应有 no_grad）
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, dataloaders['test'], criterion, device)
    print(f"\nFinal Test Loss (from best model): {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")

    
    torch.save(model.state_dict(), early_stopping.path)
    print(f"Best model saved to {early_stopping.path}")

    

if __name__ == '__main__':
    # --- This is a test script to run a full conformal training cycle ---
    
    # This mock_config now matches the nested structure of the main config.py
    mock_config = {
        'dataset_path': './BRACS_Rol/latest_version', # IMPORTANT: Update this path
        'output_dir': './experiments/conformal_test',
        'device': 'cuda',
        'mean': (0.712733, 0.545225, 0.685850), # Placeholder
        'std': (0.170330, 0.209620, 0.151623),  # Placeholder

        'model': {'pretrained': True},
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'epochs': 5,
        },
        'conformal': { # Conformal training specific hyperparameters
            'alpha': 0.1,
            'fraction': 0.5, # 50% for calibration, 50% for prediction
            'temperature': 0.1,
            'regularization_strength': 0.01,
            'size_weight': 0.1,
            'cross_entropy_weight': 0.1 # Can add a small amount of CE loss for stability
        }
    }

    if os.path.exists(mock_config['dataset_path']):
        run_conformal_training(mock_config)
    else:
        print(f"Test path '{mock_config['dataset_path']}' does not exist. Please update the path to run the test.")
        