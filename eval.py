# eval.py (Correct PyTorch Version)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
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
            # CATModel returns (logits, h_x)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
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
    if args.mode == 'cat':
        model = models.get_model(model_type='cat',backbone_name=config['model']['name'],num_classes=config['num_classes'],pretrained=config['model']['pretrained'])
    else:
        model = models.get_model(model_type='standard',backbone_name=config['model']['name'],num_classes=config['num_classes'],pretrained=config['model']['pretrained'])
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")
    
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    print(f"Model loaded from {args.model_path}")

    # --- 4. Get Predictions on Calibration and Test Sets ---
    print("\nGetting predictions on calibration set (val split)...")
    cal_probs, cal_labels = get_predictions(model, cal_loader, device)
    
    print("\nGetting predictions on test set...")
    test_probs, test_labels = get_predictions(model, test_loader, device)
    
    # --- 5. Perform Conformal Prediction ---
    print(f"\nPerforming conformal prediction with alpha = {args.alpha}...")
    

    if args.cp_method == 'thr':
        p_true = cal_probs[torch.arange(len(cal_labels)), cal_labels]
        scores = 1.0 - p_true
        qhat = cp.conformal_quantile(scores, alpha=args.alpha)  # 分位
        tau = 1.0 - qhat
        print(f"THR — calibrated tau: {tau:.4f}")
        prediction_sets = cp.predict_threshold(test_probs, tau)  # probs >= tau
        top1_guarantee=True
        if top1_guarantee:
            empty_sets_mask = ~prediction_sets.any(dim=1)
            if empty_sets_mask.any():
                top1_preds = torch.argmax(test_probs, dim=1)
                prediction_sets[empty_sets_mask, top1_preds[empty_sets_mask]] = True

    elif args.cp_method == 'aps':
        qhat = cp.calibrate_aps(cal_probs, cal_labels, alpha=args.alpha)
        print(f"APS — calibrated qhat: {qhat:.4f}")
        prediction_sets = cp.predict_aps(test_probs, qhat, top1_guarantee=True)

    elif args.cp_method == 'raps':
        k_reg = getattr(args, "k_reg", None)
        lambda_reg = getattr(args, "lambda_reg", None)

        tau = cp.calibrate_raps(cal_probs, cal_labels, alpha=args.alpha,
                                k_reg=k_reg, lambda_reg=lambda_reg)
        print(f"RAPS — calibrated tau: {tau:.4f}")

        prediction_sets = cp.predict_raps(test_probs, tau,
                                        k_reg=k_reg, lambda_reg=lambda_reg)
    else:
        raise ValueError(f"Unknown cp-method: {args.cp_method}")


    empty_sets = (~prediction_sets.any(dim=1)).float().mean().item()
    print("Empty prediction sets ratio:", empty_sets)
    # --- 6. Calculate and Report Metrics ---
    coverage = torch.mean(
        prediction_sets[torch.arange(len(test_labels)), test_labels].float()
    ).item()
    
    avg_set_size = torch.mean(torch.sum(prediction_sets, dim=1).float()).item()
    
    print("\n--- Conformal Prediction Evaluation Results ---")
    print(f"Target Coverage: {1 - args.alpha:.2%}")
    print(f"Empirical Coverage on Test Set: {coverage:.2%}")
    print(f"Average Prediction Set Size on Test Set: {avg_set_size:.3f}")

    # --- 7. Standard Classification Metrics on Test Set ---
    print("\n--- Classification Metrics (Test Set) ---")
    with torch.no_grad():
        preds = torch.argmax(test_probs, dim=1)
        correct = (preds == test_labels).sum().item()
        total = test_labels.numel()
        acc = correct / max(total, 1)
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")

        num_classes = config['num_classes']
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for t, p in zip(test_labels, preds):
            cm[t.long(), p.long()] += 1

        # Compute per-class precision/recall/F1
        tp = cm.diag().to(torch.float32)
        fp = (cm.sum(dim=0) - tp).to(torch.float32)
        fn = (cm.sum(dim=1) - tp).to(torch.float32)

        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)

        class_names = data_info.get('class_names', [str(i) for i in range(num_classes)])
        print("\nPer-class metrics:")
        for i in range(num_classes):
            print(f"  {class_names[i]}: P={precision[i].item():.4f} R={recall[i].item():.4f} F1={f1[i].item():.4f} (support={int(cm[i].sum().item())})")

        macro_p = precision.mean().item()
        macro_r = recall.mean().item()
        macro_f1 = f1.mean().item()
        print(f"\nMacro-avg: P={macro_p:.4f} R={macro_r:.4f} F1={macro_f1:.4f}")

        # print("\nConfusion Matrix (rows=true, cols=pred):")
        # # Print compact confusion matrix
        # for i in range(num_classes):
        #     row = ' '.join(f"{int(x)}" for x in cm[i].tolist())
        #     print(f"  {class_names[i]:>10}: {row}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model with Conformal Prediction.")
    parser.add_argument('--dataset', type=str, default='breakhis', choices=['bracs', 'bach', 'breakhis'])
    parser.add_argument('--xs', type=str, default='40X', choices=['40X', '100X', '200X', '400X', 'all'])
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='The desired miscoverage level alpha (e.g., 0.1 for 90% target coverage).')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50','efficientnet_b0'])
    parser.add_argument('--mode', type=str, default='cat', choices=['conformal', 'cat', 'normal'],
                        help='Model architecture to instantiate for loading the checkpoint.')
    parser.add_argument('--num_classes', type=int, default=None,
                    help="Number of classes in the dataset. Overrides config default.")
    parser.add_argument('--cp-method', type=str, default='thr', choices=['thr','aps','raps'])

    args = parser.parse_args()
    if args.dataset == 'breakhis':
        args.model_path = f'./experiments/{args.dataset}/{args.mode}/{args.model}/{args.xs}/model_best.pth'
    else:
        args.model_path = f'./experiments/{args.dataset}/{args.mode}/{args.model}/model_best.pth'
    print(args.model_path)

    config = get_config(dataset=args.dataset)
    if args.num_classes: 
        config['num_classes'] = args.num_classes
    if args.dataset == 'breakhis' and args.xs != 'all':
        config['dataset_path'] = f'./datasets/{args.dataset}/{args.xs}'
    else:
        config['dataset_path'] = f'./datasets/{args.dataset}'
    if args.model: config['model']['name'] = args.model

    print(config['dataset_path'])
    run_evaluation(config, args)
