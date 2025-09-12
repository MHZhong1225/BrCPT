# train_normal.py (PyTorch Version)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Any, Tuple

# Import from our refactored files
from utils import models
from utils import data

def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: torch.device
) -> float:
    """
    Performs one full training pass over the training dataset.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): The DataLoader for the training data.
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (optim.Optimizer): The optimizer to update the model's weights.
        device (torch.device): The device (CPU or GPU) to perform training on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    # Use tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for inputs, labels in progress_bar:
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item() * inputs.size(0)
        
        # Update progress bar description
        progress_bar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates the model on a given dataset (validation or test).

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): The DataLoader for the evaluation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to perform evaluation on.

    Returns:
        Tuple[float, float]: A tuple containing the average loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient calculation for efficiency
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            # Update running loss
            running_loss += loss.item() * inputs.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def run_normal_training(config: Dict[str, Any]):
    """
    The main function to run a standard training and evaluation pipeline.
    """
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_config = config['training']

    # --- 2. Data ---
    # Load your calculated mean and std, or use placeholders for now
    data_info = data.get_dataloaders(
        dataset_path=config['dataset_path'],
        batch_size=train_config['batch_size'],
        mean=config['mean'],
        std=config['std']
    )
    dataloaders = data_info['dataloaders']
    num_classes = len(data_info['class_names'])

    # --- 3. Model ---
    model = models.create_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    # --- 4. Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=train_config['learning_rate'], 
        momentum=train_config['momentum'],
        weight_decay=train_config['weight_decay']
    )
    # Learning rate scheduler (optional but recommended)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # --- 5. Training Loop ---
    print("\n--- Starting Standard Training ---")
    for epoch in range(train_config['epochs']):
        print(f"\nEpoch {epoch+1}/{train_config['epochs']}")
        
        train_loss = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
        
        val_loss, val_acc = evaluate(model, dataloaders['val'], criterion, device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        scheduler.step() # Update learning rate

    print("\n--- Training Finished ---")
    
    # --- 6. Final Evaluation on Test Set ---
    test_loss, test_acc = evaluate(model, dataloaders['test'], criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")

    # --- 7. Save the trained model (optional) ---
    save_path = os.path.join(config['output_dir'], 'baseline_model.pth')
    os.makedirs(config['output_dir'], exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    # --- This is a test script to run a full training cycle ---
    
    # Create a mock config dictionary for the test
    mock_config = {
        'dataset_path': './BRACS_Rol/latest_version', # IMPORTANT: Update this path
        'batch_size': 16, # Use a smaller batch size for testing
        'learning_rate': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': 5, # Run for only a few epochs for the test
        'output_dir': './experiments/baseline_test',
        'mean': (0.5, 0.5, 0.5), # Placeholder
        'std': (0.5, 0.5, 0.5)   # Placeholder
    }

    if os.path.exists(mock_config['dataset_path']):
        run_normal_training(mock_config)
    else:
        print(f"Test path '{mock_config['dataset_path']}' does not exist. Please update the path to run the test.")