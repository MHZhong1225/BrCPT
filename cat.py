# cat.py
import os

from utils.utils import final_evaluation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any
import wandb  #

from utils import models
from utils import data
from utils import smooth_conformal_prediction as scp
from utils import utils as cputils
from train_normal import evaluate




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
):
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
    
    warmup_epochs = int(cp_cfg.get('warmup_epochs', 5))
    ce_criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc="CAT Training")

    total_loss_sum   = 0.0
    cp_loss_sum      = 0.0
    cov_loss_sum     = 0.0
    size_loss_sum    = 0.0
    ce_loss_sum      = 0.0
    
    # WandB
    q_sum            = 0.0
    h_sum            = 0.0
    
    num_samples      = 0

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
        tau_pred = pred_h - qhat_soft
        soft_sets = scp.smooth_predict_threshold(pred_probs, tau_pred, temperature=T)

        # ---- 3. Losses ----
        coverage_loss = cputils.compute_coverage_loss(soft_sets, pred_labels, loss_matrix)
        size_loss     = cputils.compute_size_loss(soft_sets, target_size=target_size)
        
        # CE Loss
        all_logits = torch.cat([cal_logits, pred_logits], dim=0)
        all_labels = torch.cat([cal_labels, pred_labels], dim=0)
        ce_loss = ce_criterion(all_logits, all_labels)

        # Warmup
        if epoch < warmup_epochs:
            cp_scale = 0.0
        else:
            cp_scale = min(1.0, (epoch - warmup_epochs + 1) / warmup_epochs)
            
        cp_loss = cp_scale * (coverage_loss + size_weight * size_loss)
        loss = cp_loss + (cross_entropy_weight * ce_loss)

        # Optimization
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

        # Stats Update
        bs = inputs.size(0)
        num_samples   += bs
        running_loss  += loss.item() * bs
        cp_loss_sum   += cp_loss.item() * bs
        cov_loss_sum  += coverage_loss.item() * bs
        size_loss_sum += size_loss.item() * bs
        ce_loss_sum   += ce_loss.item() * bs
        
        # WandB
        q_sum += qhat_soft.item() * bs
        h_sum += pred_h.mean().item() * bs

        progress_bar.set_postfix(
            ce=f"{ce_loss.item():.3f}",
            cov=f"{coverage_loss.item():.3f}",
            size=f"{size_loss.item():.3f}",
            q=f"{qhat_soft.item():.3f}",
            h=f"{pred_h.mean().item():.2f}"
        )

    # Calculate Averages
    dataset_len = len(dataloader.dataset)
    avg_total = running_loss / dataset_len
    avg_cp    = cp_loss_sum  / dataset_len
    avg_cov   = cov_loss_sum / dataset_len
    avg_size  = size_loss_sum / dataset_len
    avg_ce    = ce_loss_sum  / dataset_len
    
    # WandB
    avg_q     = q_sum / dataset_len
    avg_h     = h_sum / dataset_len
    
    return avg_total, avg_cp, avg_cov, avg_size, avg_ce, avg_q, avg_h


def run_cat_training(config: Dict[str, Any]):

    wandb.init(
        project="CAT-bach", 
        name=config.get('wandb_name', f"{config['model']['name']}_experiment"),
        group=config.get('wandb_group', None),
        tags=config.get('wandb_tags', "").split(",") if config.get('wandb_tags') else [],
        config=config,
        reinit=True
    )
    
    device = config['device']
    print(f"Using device: {device}")

    train_cfg = config['training']
    hnet_cfg  = config['threshold_net']
    cp_cfg    = config['conformal']

    h_only = bool(cp_cfg.get("h_only", False)) 
    
    # Data Loading
    data_info = data.get_dataloaders(
        dataset_path=config['dataset_path'],
        batch_size=train_cfg['batch_size'],
        mean=config['mean'],
        std=config['std']
    )
    dls = data_info['dataloaders']
    num_classes = len(data_info['class_names'])

    # Model Setup
    model = models.get_model(
        model_type='cat',
        backbone_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=config['model']['pretrained']
    ).to(device)

    # ------------------------------------------------------------------
    wandb.watch(model, log="all", log_freq=100)

    # Checkpoint Loading Logic
    if h_only:
        init_from = cp_cfg.get("init_backbone_from", None)
        print(f"Initializing CAT backbone from {init_from}")
        raw_state = torch.load(init_from, map_location=device, weights_only=True)

        if any(k.startswith("0.") for k in raw_state.keys()):
            print("[H-only] Detected Sequential-style checkpoint.")
            backbone_state = {k.replace("0.", "", 1): v for k, v in raw_state.items() if k.startswith("0.")}
            classifier_state = {k.replace("2.", "", 1): v for k, v in raw_state.items() if k.startswith("2.")}
            model.backbone.load_state_dict(backbone_state, strict=False)
            model.classifier.load_state_dict(classifier_state, strict=False)
        else:
            print("[H-only] Detected non-Sequential checkpoint.")
            model.load_state_dict(raw_state, strict=False)

        print(">>> H-only ablation: freezing backbone + classifier.")
        for p in model.backbone.parameters(): p.requires_grad = False
        for p in model.classifier.parameters(): p.requires_grad = False

    # Optimizers
    if not h_only:
        optimizer_backbone = optim.SGD(
            list(model.backbone.parameters()) + list(model.classifier.parameters()),
            lr=train_cfg['learning_rate'],
            momentum=train_cfg['momentum'],
            weight_decay=train_cfg['weight_decay']
        )
    else:
        optimizer_backbone = None

    optimizer_h = optim.Adam(
        model.threshold_net.parameters(),
        lr=hnet_cfg['learning_rate'],
        weight_decay=hnet_cfg['weight_decay']
    )

    # Schedulers
    if optimizer_backbone is not None:
        scheduler_backbone = optim.lr_scheduler.StepLR(
            optimizer_backbone, step_size=train_cfg.get('lr_step', 30), gamma=0.1
        )
    else:
        scheduler_backbone = None

    scheduler_h = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_h, mode='min', factor=0.5, patience=5
    )

    #  If need..
    loss_matrix = build_loss_matrix(num_classes, device)
    beta = 0.1
    I = torch.eye(num_classes, device=device)
    loss_matrix = loss_matrix + beta * (1 - I)

    eval_model = nn.Sequential(model.backbone, nn.Flatten(), model.classifier).to(device)

    # Output paths
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
    
    history = {
        "epoch": [], "train_total": [], "train_cp": [], "train_cov": [],
        "train_size": [], "train_ce": [], "val_loss": [], "val_acc": [],
    }

    print("\n--- Starting CAT Training (Prob-ReLU) ---")
    for epoch in range(train_cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{train_cfg['epochs']}")

        # Train Step
        train_metrics = train_one_epoch_cat(
            model, dls['train'],
            optimizer_backbone, optimizer_h,
            loss_matrix, config, device,
            epoch=epoch,
            h_only=h_only,
        )
        # avg_q \ avg_h
        train_total, train_cp, train_cov, train_size, train_ce, avg_q, avg_h = train_metrics
        
        print(f"Train Loss: {train_total:.4f} | Avg Q: {avg_q:.3f} | Avg H: {avg_h:.3f}")

        # Val Step
        val_loss, val_acc = evaluate(eval_model, dls['val'], nn.CrossEntropyLoss(), device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # LR Step
        if scheduler_backbone is not None:
            scheduler_backbone.step()
        scheduler_h.step(val_loss)

        #  Log
        log_dict = {
            "epoch": epoch + 1,
            # Loss Components
            "Train/Total_Loss": train_total,
            "Train/CP_Loss": train_cp,
            "Train/Coverage_Loss": train_cov,
            "Train/Size_Loss": train_size,
            "Train/CE_Loss": train_ce,
            # CP Dynamics
            "Train/Avg_Q_hat": avg_q,
            "Train/Avg_H_score": avg_h,
            # Validation
            "Val/Loss": val_loss,
            "Val/Accuracy": val_acc,
            # Hyperparams Dynamics
            "LR/H_Net": optimizer_h.param_groups[0]['lr']
        }
        if optimizer_backbone:
            log_dict["LR/Backbone"] = optimizer_backbone.param_groups[0]['lr']
            
        wandb.log(log_dict)
        # ------------------------------------------------------------------

        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        # Append to local history
        history["epoch"].append(epoch + 1)
        history["train_total"].append(train_total)
        history["train_cp"].append(train_cp)
        history["train_cov"].append(train_cov)
        history["train_size"].append(train_size)
        history["train_ce"].append(train_ce)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
    print("\n--- Training Finished ---")
    print(f"Loading best model (val_loss={early_stopping.val_loss_min:.4f})")
    model.load_state_dict(early_stopping.best_model_state_dict)
    
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")

    # if config["save_loss"]:
    #     log_path = os.path.join(save_dir, "loss_curve.csv")
    #     df = pd.DataFrame(history)
    #     df.to_csv(log_path, index=False)
    #     print(f"Training loss curves saved to: {log_path}")

    test_metrics = final_evaluation(model, dls, device, config)
    for k, v in test_metrics.items():
        wandb.run.summary[k] = v
        wandb.log({k: v})
    
    print("Run finished. WandB synced.")
    wandb.finish()
