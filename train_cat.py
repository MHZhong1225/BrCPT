# train_cat.py (Corrected Version)
import os, random, numpy as np, torch
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

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Any
from utils import models

from utils import data
from utils import smooth_conformal_prediction as scp
from utils import train_utils as cputils
from train_normal import evaluate


def train_one_epoch_cat(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer_backbone: optim.Optimizer,
    optimizer_h: optim.Optimizer,
    config: Dict[str, Any],
    device: torch.device
) -> float:
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc="CAT Training")

    conf_config = config['conformal']

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)


        batch_size = inputs.shape[0]
        cal_split_size = int(batch_size * conf_config['fraction'])
        if cal_split_size == 0 or cal_split_size == batch_size:
            continue

        cal_labels, pred_labels = labels[:cal_split_size], labels[cal_split_size:]

        all_logits, all_h_x = model(inputs)

        cal_logits, pred_logits = all_logits[:cal_split_size], all_logits[cal_split_size:]
        cal_h_x, pred_h_x = all_h_x[:cal_split_size], all_h_x[cal_split_size:]

        cal_scores = torch.softmax(cal_logits, dim=1)
        pred_scores = torch.softmax(pred_logits, dim=1)

        optimizer_backbone.zero_grad()

        smooth_sets = scp.smooth_predict_threshold(
            pred_scores,
            pred_h_x.detach(),
            temperature=conf_config['temperature']
        )

        loss_matrix = torch.eye(pred_logits.shape[1], device=device)
        coverage_loss = cputils.compute_coverage_loss(smooth_sets, pred_labels, loss_matrix)
        size_loss = cputils.compute_size_loss(smooth_sets, target_size=1)
        ce_loss_main = nn.CrossEntropyLoss()(pred_logits, pred_labels)

        loss_backbone = (
            coverage_loss +
            conf_config['size_weight'] * size_loss +
            conf_config['cross_entropy_weight'] * ce_loss_main
        )

        loss_backbone.backward()
        optimizer_backbone.step()

        optimizer_h.zero_grad()

        cal_conformity_scores = cal_scores[torch.arange(cal_split_size), cal_labels]
        tau_cal = scp.smooth_conformal_quantile(
            cal_conformity_scores,
            alpha=conf_config['alpha'],
            regularization_strength=conf_config['regularization_strength']
        )

        loss_h = nn.MSELoss()(cal_h_x, tau_cal.detach().expand_as(cal_h_x))

        loss_h.backward()
        optimizer_h.step()

        total_loss = loss_backbone.item() + loss_h.item()
        running_loss += total_loss * inputs.size(0)
        progress_bar.set_postfix(loss_main=f"{loss_backbone.item():.3f}", loss_h=f"{loss_h.item():.3f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def run_cat_training(config: Dict[str, Any]):
    device = config['device']
    print(f"Using device: {device}")

    train_config = config['training']
    hnet_config = config['threshold_net']

    data_info = data.get_dataloaders(
        dataset_path=config['dataset_path'],
        batch_size=train_config['batch_size'],
        mean=config['mean'],
        std=config['std']
    )
    dataloaders = data_info['dataloaders']
    num_classes = len(data_info['class_names'])

    # model = CATModel(num_classes=num_classes, pretrained=config['model']['pretrained'])
    model = models.get_model(
        model_type='cat',
        backbone_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=config['model']['pretrained'])

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("--- Model Parameters ---")
    print(f"Total Parameters:     {total_params / (1024*1024):.2f}M")
    print(f"Trainable Parameters: {trainable_params_count/ (1024*1024):.2f}M")
    print("------------------------")
    
    optimizer_backbone = optim.SGD(
        list(model.backbone.parameters()) + list(model.classifier.parameters()), 
        lr=train_config['learning_rate'], 
        momentum=train_config['momentum'],
        weight_decay=train_config['weight_decay']
    )

    optimizer_h = optim.Adam(
        model.threshold_net.parameters(),
        lr=hnet_config['learning_rate'],
        weight_decay=hnet_config['weight_decay']
    )

    scheduler_backbone = optim.lr_scheduler.StepLR(optimizer_backbone, step_size=30, gamma=0.1)
    

    eval_model = nn.Sequential(
        model.backbone,
        nn.Flatten(),
        model.classifier
    ).to(device)


    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # model_p = f"model_best_{timestamp}.pth"
    model_p = f"model_best.pth"
    if config['dataset']=='breakhis':
        save_dir = os.path.join(config['output_dir'], config['model']['name'], config['xs'])
    else:
        save_dir = os.path.join(config['output_dir'], config['model']['name'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_p)

    early_stopping = cputils.EarlyStopping(patience=15, verbose=True, path=save_path)


    print("\n--- Starting CAT Training ---")
    for epoch in range(train_config['epochs']):
        print(f"\nEpoch {epoch+1}/{train_config['epochs']}")

        train_loss = train_one_epoch_cat(
            model, dataloaders['train'], optimizer_backbone, optimizer_h, config, device
        )
        print(f"Epoch {epoch+1} CAT Train Loss: {train_loss:.4f}")


        val_loss, val_acc = evaluate(eval_model, dataloaders['val'], nn.CrossEntropyLoss(), device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        scheduler_backbone.step()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break 
    print("\n--- CAT Training Finished ---")

    print(f"Loading best model from epoch with val_loss: {early_stopping.val_loss_min:.4f}")
    model.load_state_dict(early_stopping.best_model_state_dict)
    best_eval_model = nn.Sequential(
        model.backbone,
        nn.Flatten(),
        model.classifier
    ).to(device)
    
    test_loss, test_acc = evaluate(best_eval_model, dataloaders['test'], nn.CrossEntropyLoss(), device)
    print(f"\nFinal Test Loss (from best model): {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")
    
    torch.save(model.state_dict(), early_stopping.path)
    print(f"Best model saved to {early_stopping.path}")

