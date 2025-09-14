# eval.py (Correct PyTorch Version)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import argparse
from typing import Dict, Any, Tuple

# Import from our refactored PyTorch files
from utils import models
from utils import data
from utils import conformal_prediction as cp
from config import get_config

def get_predictions(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the model's softmax predictions and corresponding labels for a given dataset.
    """
    model.eval() # Set the model to evaluation mode
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Getting model predictions"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_probs), torch.cat(all_labels)

def run_evaluation(config: Dict[str, Any], args: argparse.Namespace):
    """
    Runs the full evaluation pipeline for a trained model.
    """
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    data_info = data.get_dataloaders(
        dataset_path=config['dataset_path'],
        batch_size=config['training']['batch_size'],
        mean=config['mean'],
        std=config['std']
    )
    cal_loader = data_info['dataloaders']['val']
    test_loader = data_info['dataloaders']['test']

    # --- 3. Load Trained Model ---
    model = models.create_model(num_classes=config['num_classes'], pretrained=False)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device,weights_only=True))
    model.to(device)
    print(f"Model loaded from {args.model_path}")

    # --- 4. Get Predictions on Calibration and Test Sets ---
    print("\nGetting predictions on calibration set (val split)...")
    cal_probs, cal_labels = get_predictions(model, cal_loader, device)
    
    print("\nGetting predictions on test set...")
    test_probs, test_labels = get_predictions(model, test_loader, device)
    
    # --- 5. Perform Conformal Prediction ---
    print(f"\nPerforming conformal prediction with alpha = {args.alpha}...")
    
    tau = cp.calibrate_threshold(cal_probs, cal_labels, alpha=args.alpha)
    print(f"Calibrated threshold (tau): {tau:.4f}")

    prediction_sets = cp.predict_threshold(test_probs, tau)
    
    # --- 6. Calculate and Report Metrics ---
    coverage = torch.mean(
        prediction_sets[torch.arange(len(test_labels)), test_labels].float()
    ).item()
    
    avg_set_size = torch.mean(torch.sum(prediction_sets, dim=1).float()).item()
    
    print("\n--- Conformal Prediction Evaluation Results ---")
    print(f"Target Coverage: {1 - args.alpha:.2%}")
    print(f"Empirical Coverage on Test Set: {coverage:.2%}")
    print(f"Average Prediction Set Size on Test Set: {avg_set_size:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model with Conformal Prediction.")
    parser.add_argument('--model_path', type=str, default='./experiments/bracs/conformal_20250914-122634/',
                        help='Path to the trained model checkpoint (.pth file).')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='The desired miscoverage level alpha (e.g., 0.1 for 90% target coverage).')
    
    args = parser.parse_args()
    args.model_path = args.model_path+'conformal_model.pth'
    config = get_config()
    
    run_evaluation(config, args)