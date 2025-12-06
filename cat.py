# cat.py 

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

from utils import models
from utils import data
from utils import smooth_conformal_prediction as scp
from utils import train_utils as cputils
from train_normal import evaluate

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


def build_loss_matrix(num_classes: int, device, penal_cfg: Dict[str, Any] = None):
    L = torch.eye(num_classes, device=device)
    if not penal_cfg:
        return L
    for y, w in penal_cfg.get("on_diagonal", []):
        L[y, y] = float(w)
    for y, k, w in penal_cfg.get("pairs", []):
        L[y, k] = float(w)
    for g in penal_cfg.get("groups", []):
        w = float(g["w"])
        for y in g["from"]:
            for k in g["to"]:
                L[y, k] = w
    return L


def train_one_epoch_cat(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer_backbone: optim.Optimizer,
    optimizer_h: optim.Optimizer,
    loss_matrix: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
    epoch: int = 0,
    h_only: bool = False, 
) -> float:
    model.train()
    if h_only:
        model.backbone.eval()
        model.classifier.eval()
    running_loss = 0.0

    cp_cfg = config['conformal']
    frac   = cp_cfg.get('fraction', 0.5)
    alpha  = float(cp_cfg.get('alpha', 0.1))
    

    T = float(cp_cfg.get('temperature', 0.5)) 
    reg = float(cp_cfg.get('regularization_strength', 0.1))

    size_weight = float(cp_cfg.get('size_weight', 0.1))
    cross_entropy_weight = float(cp_cfg.get('cross_entropy_weight', 1.0))
    target_size = int(cp_cfg.get('target_size', 1))
    
    # h_reg_w = 0.0 

    warmup_epochs = int(cp_cfg.get('warmup_epochs', 5))
    ce_criterion = nn.CrossEntropyLoss()
    
    # if h_only:
    #     warmup_epochs = 0
    #     cp_scale= 1 


    progress_bar = tqdm(dataloader, desc="CAT Training")

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        B = inputs.size(0)
        cal_B = int(B * frac)
        if cal_B == 0 or cal_B == B:
            continue

        cal_inputs, pred_inputs = inputs[:cal_B], inputs[cal_B:]
        cal_labels, pred_labels = labels[:cal_B], labels[cal_B:]

        # Forward
        cal_logits, cal_h = model(cal_inputs)
        pred_logits, pred_h = model(pred_inputs)

        cal_probs  = torch.softmax(cal_logits, dim=1)
        pred_probs = torch.softmax(pred_logits, dim=1)

        if cal_h.dim() == 1: cal_h = cal_h.unsqueeze(1)
        if pred_h.dim() == 1: pred_h = pred_h.unsqueeze(1)

        # ---- 1. Calibration Step ----
        cal_p_true = cal_probs[torch.arange(cal_B, device=device), cal_labels]
        
        s_cal = torch.relu(cal_h.squeeze(1) - cal_p_true)

        # Calculate Quantile
        q_level = (cal_B + 1) * (1 - alpha) / cal_B
        q_level = min(1.0, max(0.0, q_level))

        qhat_soft = scp.soft_quantile(
            s_cal, 
            q=q_level, 
            regularization_strength=reg
        )
        
        qhat_soft = qhat_soft.detach()

        # ---- 2. Prediction Step ----
        # Threshold tau = h - qhat
        tau_pred = pred_h - qhat_soft
        
        # Soft 
        soft_sets = scp.smooth_predict_threshold(pred_probs, tau_pred, temperature=T)

        # ---- 3. Losses ----
        coverage_loss = cputils.compute_coverage_loss(soft_sets, pred_labels, loss_matrix)
        size_loss     = cputils.compute_size_loss(soft_sets, target_size=target_size)
        
        # CE Loss
        all_logits = torch.cat([cal_logits, pred_logits], dim=0)
        all_labels = torch.cat([cal_labels, pred_labels], dim=0)
        ce_loss = ce_criterion(all_logits, all_labels)

        # Warmup

        # if not h_only:
        if epoch < warmup_epochs:
            cp_scale = 0.0
        else:
            cp_scale = min(1.0, (epoch - warmup_epochs + 1) / warmup_epochs)

        loss = (cp_scale * coverage_loss) + \
               (cp_scale * size_weight * size_loss) + \
               (cross_entropy_weight * ce_loss)

        # Optimization
        # optimizer_backbone.zero_grad()
        optimizer_h.zero_grad()
        if (optimizer_backbone is not None) and (not h_only):
            optimizer_backbone.zero_grad()
        loss.backward()


        if (optimizer_backbone is not None) and (not h_only):
            if cp_cfg.get("grad_clip_norm", None):
                torch.nn.utils.clip_grad_norm_(
                    list(model.backbone.parameters()) + list(model.classifier.parameters()),
                    max_norm=float(cp_cfg["grad_clip_norm"])
                )
            optimizer_backbone.step()
        if cp_cfg.get("h_grad_clip_norm", None):
            torch.nn.utils.clip_grad_norm_(
                model.threshold_net.parameters(),
                max_norm=float(cp_cfg["h_grad_clip_norm"])
            )
        optimizer_h.step()

        running_loss += loss.item() * inputs.size(0)

        progress_bar.set_postfix(
            ce=f"{ce_loss.item():.3f}",
            cov=f"{coverage_loss.item():.3f}",
            size=f"{size_loss.item():.3f}",
            q=f"{qhat_soft.item():.3f}",
            h=f"{pred_h.mean().item():.2f}"
        )

    return running_loss / len(dataloader.dataset)


def run_cat_training(config: Dict[str, Any]):
    device = config['device']
    print(f"Using device: {device}")

    train_cfg = config['training']
    hnet_cfg  = config['threshold_net']
    cp_cfg    = config['conformal']

    h_only = bool(cp_cfg.get("h_only", False))   # 新增：只学 h(x
    data_info = data.get_dataloaders(
        dataset_path=config['dataset_path'],
        batch_size=train_cfg['batch_size'],
        mean=config['mean'],
        std=config['std']
    )
    dls = data_info['dataloaders']
    num_classes = len(data_info['class_names'])

    model = models.get_model(
        model_type='cat',
        backbone_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=config['model']['pretrained']
    ).to(device)

    if h_only:
        init_from = cp_cfg.get("init_backbone_from", None)
        print(f"Initializing CAT backbone from {init_from}")
        raw_state = torch.load(init_from, map_location=device, weights_only=True)

        if any(k.startswith("0.") for k in raw_state.keys()):
            print("[H-only] Detected Sequential-style checkpoint (keys start with '0.' / '2.').")
            backbone_state = {k.replace("0.", "", 1): v
                            for k, v in raw_state.items()
                            if k.startswith("0.")}
            classifier_state = {k.replace("2.", "", 1): v
                                for k, v in raw_state.items()
                                if k.startswith("2.")}

            # 加载到 CAT 的 backbone / classifier 里
            missing_b, unexpected_b = model.backbone.load_state_dict(backbone_state, strict=False)
            missing_c, unexpected_c = model.classifier.load_state_dict(classifier_state, strict=False)

            print("[DEBUG] backbone loaded, missing:", missing_b, "unexpected:", unexpected_b)
            print("[DEBUG] classifier loaded, missing:", missing_c, "unexpected:", unexpected_c)

        else:
            # 兜底：如果 ckpt 本身就是带 backbone.xxx / classifier.xxx 这种结构
            print("[H-only] Detected non-Sequential checkpoint, trying direct load to model.")
            missing, unexpected = model.load_state_dict(raw_state, strict=False)
            print("[DEBUG] missing keys:", missing)
            print("[DEBUG] unexpected keys:", unexpected)


    if h_only:
        print(">>> H-only ablation: freezing backbone + classifier, only training threshold_net.")
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = False


    if not h_only:
        optimizer_backbone = optim.SGD(
            list(model.backbone.parameters()) + list(model.classifier.parameters()),
            lr=train_cfg['learning_rate'],
            momentum=train_cfg['momentum'],
            weight_decay=train_cfg['weight_decay']
        )
    else:
        optimizer_backbone = None

    # h(x) 的 optimizer 一直都有
    optimizer_h = optim.Adam(
        model.threshold_net.parameters(),
        lr=hnet_cfg['learning_rate'],
        weight_decay=hnet_cfg['weight_decay']
    )

    # scheduler
    if optimizer_backbone is not None:
        scheduler_backbone = optim.lr_scheduler.StepLR(
            optimizer_backbone, step_size=train_cfg.get('lr_step', 30), gamma=0.1
        )
    else:
        scheduler_backbone = None

    scheduler_h = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_h, mode='min', factor=0.5, patience=5
    )

    # loss_matrix = build_loss_matrix(num_classes, device, config.get("penalties", None))
    loss_matrix = build_loss_matrix(num_classes, device)
    beta = 0.1
    I = torch.eye(num_classes, device=device)
    loss_matrix = loss_matrix + beta * (1 - I)

    eval_model = nn.Sequential(model.backbone, nn.Flatten(), model.classifier).to(device)

    model_p = "model_best.pth"
    if config['dataset'] == 'breakhis':
        save_dir = os.path.join(config['output_dir'], config['model']['name'], config['xs'])
    else:
        save_dir = os.path.join(config['output_dir'], config['model']['name'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_p)

    early_stopping = cputils.EarlyStopping(
        patience=train_cfg.get("patience", 15),
        verbose=True,
        path=save_path
    )

    print("\n--- Starting CAT Training (Prob-ReLU) ---")
    for epoch in range(train_cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{train_cfg['epochs']}")

        train_loss = train_one_epoch_cat(
            model, dls['train'],
            optimizer_backbone, optimizer_h,
            loss_matrix, config, device,
            epoch=epoch,
            h_only=h_only,      # 关键
        )
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_acc = evaluate(eval_model, dls['val'], nn.CrossEntropyLoss(), device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if scheduler_backbone is not None:
            scheduler_backbone.step()
        scheduler_h.step(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\n--- Training Finished ---")
    print(f"Loading best model (val_loss={early_stopping.val_loss_min:.4f})")
    model.load_state_dict(early_stopping.best_model_state_dict)
    
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")
