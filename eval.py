# eval.py (Fixed: Keyword Arguments & Probability Domain)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from typing import Dict, Any, Tuple

from utils import models
from utils import data
from utils import conformal_prediction as cp
from config import get_config

def get_predictions(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device,
    return_h: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    获取模型预测结果 (Probabilities)。
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_h = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Getting predictions"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # CATModel returns (logits, h_x)
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                logits, h_x = outputs[0], outputs[1]
            else:
                logits, h_x = outputs, None

            # Softmax for probabilities
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

            if return_h:
                if h_x is None:
                    raise ValueError("Expected model to return h_x in CAT mode.")
                if h_x.dim() == 1:
                    h_x = h_x.unsqueeze(1)
                all_h.append(h_x.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    h_all = torch.cat(all_h) if return_h else None
        
    return probs, labels, h_all

def run_evaluation(config: Dict[str, Any], args: argparse.Namespace):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data (FIXED: Use Keyword Arguments) ---
    # 必须使用关键字参数，否则 batch_size 会被传给 image_size
    data_info = data.get_dataloaders(
        dataset_path=config['dataset_path'],
        batch_size=config['training']['batch_size'],
        mean=config['mean'],
        std=config['std']
    )
    cal_loader = data_info['dataloaders']['val']
    test_loader = data_info['dataloaders']['test']

    # --- 3. Load Model ---
    if args.mode == 'cat':
        model = models.get_model(model_type='cat', backbone_name=config['model']['name'], num_classes=config['num_classes'], pretrained=config['model']['pretrained'])
    else:
        model = models.get_model(model_type='standard', backbone_name=config['model']['name'], num_classes=config['num_classes'], pretrained=config['model']['pretrained'])
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    print(f"Model loaded from {args.model_path}")

    # --- 4. Get Predictions ---
    print("\nGetting predictions...")
    is_cat = (args.mode == 'cat')
    
    cal_probs, cal_labels, cal_h = get_predictions(model, cal_loader, device, return_h=is_cat)
    test_probs, test_labels, test_h = get_predictions(model, test_loader, device, return_h=is_cat)

    if is_cat:
        print(f"CAT Debug - h(x) range: [{test_h.min():.4f}, {test_h.max():.4f}], mean: {test_h.mean():.4f}")

    # --- 5. Perform Conformal Prediction ---
    print(f"\nPerforming conformal prediction with alpha = {args.alpha}...")

    if args.cp_method == 'thr':
        if is_cat:
            # === CAT Mode (Probability Domain) ===
            # Consistent with fixed cat.py
            
            # 1. Prepare Data
            cal_h_val = cal_h.squeeze(1)
            test_h_val = test_h.squeeze(1)
            
            # 2. Calibration Score: s = h - p_true
            # (No ReLU, No Log)
            cal_p_true = cal_probs[torch.arange(len(cal_labels)), cal_labels]
            s_cal = torch.relu(cal_h_val - cal_p_true) 

            # 3. Calibrate Qhat
            qhat = cp.conformal_quantile(s_cal, alpha=args.alpha)
            print(f"CAT-THR (residual-prob) — calibrated qhat: {qhat:.4f}")

            # 4. Apply Threshold: tau = h - qhat
            test_thresholds = test_h_val - qhat
            
            # 5. Prediction Set: p >= tau
            prediction_sets = (test_probs >= test_thresholds.unsqueeze(1))
            
        else:
            # === Normal Mode (Standard CP) ===
            # Score s = 1 - p_true
            p_true = cal_probs[torch.arange(len(cal_labels)), cal_labels]
            scores = 1.0 - p_true 
            qhat = cp.conformal_quantile(scores, alpha=args.alpha)
            
            tau_global = 1.0 - qhat
            print(f"THR — calibrated global tau: {tau_global:.4f}")
            
            prediction_sets = cp.predict_threshold(test_probs, tau_global)

        # Top-1 Guarantee
        empty_mask = ~prediction_sets.any(dim=1)
        if empty_mask.any():
            top1_preds = torch.argmax(test_probs, dim=1)
            prediction_sets[empty_mask, top1_preds[empty_mask]] = True

    elif args.cp_method == 'aps':
        qhat = cp.calibrate_aps(cal_probs, cal_labels, alpha=args.alpha)
        print(f"APS — calibrated qhat: {qhat:.4f}")
        prediction_sets = cp.predict_aps(test_probs, qhat, top1_guarantee=True)

    elif args.cp_method == 'raps':
        k_reg = getattr(args, "k_reg", None)
        lambda_reg = getattr(args, "lambda_reg", None)
        tau = cp.calibrate_raps(cal_probs, cal_labels, alpha=args.alpha, k_reg=k_reg, lambda_reg=lambda_reg)
        print(f"RAPS — calibrated tau: {tau:.4f}")
        prediction_sets = cp.predict_raps(test_probs, tau, k_reg=k_reg, lambda_reg=lambda_reg)
    
    else:
        raise ValueError(f"Unknown cp-method: {args.cp_method}")

    # --- 6. Metrics ---
    covered = prediction_sets[torch.arange(len(test_labels)), test_labels].float()
    coverage = covered.mean().item()
    avg_set_size = torch.sum(prediction_sets, dim=1).float().mean().item()
    
    print("\n--- Conformal Prediction Evaluation Results ---")
    print(f"Target Coverage: {1 - args.alpha:.2%}")
    print(f"Empirical Coverage: {coverage:.2%}")
    print(f"Average Prediction Set Size: {avg_set_size:.3f}")

    # --- 7. Classification Metrics ---
    with torch.no_grad():
        preds = torch.argmax(test_probs, dim=1)
        correct = (preds == test_labels).sum().item()
        print(f"Accuracy: {correct / len(test_labels):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='breakhis')
    parser.add_argument('--xs', type=str, default='40X')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--mode', type=str, default='cat', choices=['conformal', 'cat', 'normal'])
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--cp-method', type=str, default='thr')

    args = parser.parse_args()
    
    if args.dataset == 'breakhis':
        args.model_path = f'./experiments/{args.dataset}/{args.mode}/{args.model}/{args.xs}/model_best.pth'
    else:
        args.model_path = f'./experiments/{args.dataset}/{args.mode}/{args.model}/model_best.pth'
        
    config = get_config(dataset=args.dataset)
    if args.num_classes: config['num_classes'] = args.num_classes
    if args.dataset == 'breakhis' and args.xs != 'all':
        config['dataset_path'] = f'./datasets/{args.dataset}/{args.xs}'
    else:
        config['dataset_path'] = f'./datasets/{args.dataset}'
    if args.model: config['model']['name'] = args.model

    print(f"Evaluating: {args.model_path}")
    run_evaluation(config, args)