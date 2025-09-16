# train_uact.py (Corrected Version)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Any

from utils.models import UACTModel
from utils import data
import smooth_conformal_prediction as scp
from utils import train_utils as cputils
from train_normal import evaluate

def train_one_epoch_uact(
    model: UACTModel,
    dataloader: DataLoader,
    optimizer_backbone: optim.Optimizer,
    optimizer_h: optim.Optimizer, # 阈值网络的独立优化器
    config: Dict[str, Any],
    device: torch.device
) -> float:
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc="U-ACT Training")

    conf_config = config['conformal']

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # --- 1. 分割批次 ---
        batch_size = inputs.shape[0]
        cal_split_size = int(batch_size * conf_config['fraction'])
        if cal_split_size == 0 or cal_split_size == batch_size:
            continue

        cal_inputs, pred_inputs = inputs[:cal_split_size], inputs[cal_split_size:]
        cal_labels, pred_labels = labels[:cal_split_size], labels[cal_split_size:]

        # --- 2. 联合前向传播 ---
        # 得到整个批次的logits和个性化阈值
        all_logits, all_h_x = model(inputs)

        cal_logits, pred_logits = all_logits[:cal_split_size], all_logits[cal_split_size:]
        cal_h_x, pred_h_x = all_h_x[:cal_split_size], all_h_x[cal_split_size:]

        # 我们使用softmax概率作为整合分数
        cal_scores = torch.softmax(cal_logits, dim=1)
        pred_scores = torch.softmax(pred_logits, dim=1)

        # --- 3. 更新主干网络 (Backbone) ---
        optimizer_backbone.zero_grad()

        # a) 计算平滑置信集
        # 关键不同：使用阈值网络为每个样本预测的 h(x) 作为tau
        smooth_sets = scp.smooth_predict_threshold(
            pred_scores,
            pred_h_x.detach(), # 使用个性化阈值, detach以确保梯度只流向主干网络
            temperature=conf_config['temperature']
        )

        # b) 计算主干网络的损失
        loss_matrix = torch.eye(pred_logits.shape[1], device=device)
        coverage_loss = cputils.compute_coverage_loss(smooth_sets, pred_labels, loss_matrix)
        size_loss = cputils.compute_size_loss(smooth_sets, target_size=1)
        ce_loss_main = nn.CrossEntropyLoss()(pred_logits, pred_labels)

        loss_backbone = (
            coverage_loss +
            conf_config['size_weight'] * size_loss +
            conf_config['cross_entropy_weight'] * ce_loss_main
        )

        # c) 反向传播并更新主干网络
        # 关键修复：添加 retain_graph=True
        loss_backbone.backward(retain_graph=True)
        optimizer_backbone.step()

        # --- 4. 更新阈值网络 (Threshold Network) ---
        optimizer_h.zero_grad()

        # a) 在校准集上计算 "真实" 的批次级tau
        cal_conformity_scores = cal_scores[torch.arange(cal_split_size), cal_labels]
        tau_cal = scp.smooth_conformal_quantile(
            cal_conformity_scores,
            alpha=conf_config['alpha'],
            regularization_strength=conf_config['regularization_strength']
        )

        # b) 计算阈值网络的损失
        # 目标是让阈值网络预测出的 cal_h_x 尽可能接近批次真实的 tau_cal
        loss_h = nn.MSELoss()(cal_h_x, tau_cal.detach().expand_as(cal_h_x))

        # c) 反向传播并更新阈值网络
        loss_h.backward()
        optimizer_h.step()

        # --- 5. 记录总损失 ---
        total_loss = loss_backbone.item() + loss_h.item()
        running_loss += total_loss * inputs.size(0)
        progress_bar.set_postfix(loss_main=f"{loss_backbone.item():.3f}", loss_h=f"{loss_h.item():.3f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def run_uact_training(config: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model = UACTModel(num_classes=num_classes, pretrained=config['model']['pretrained'])
    model.to(device)

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
    
    # --- 关键修复：创建一个用于评估的、包含完整流程的模型 ---
    eval_model = nn.Sequential(
        model.backbone,
        nn.Flatten(),
        model.classifier
    ).to(device)

    print("\n--- Starting U-ACT Training ---")
    for epoch in range(train_config['epochs']):
        print(f"\nEpoch {epoch+1}/{train_config['epochs']}")

        train_loss = train_one_epoch_uact(
            model, dataloaders['train'], optimizer_backbone, optimizer_h, config, device
        )
        print(f"Epoch {epoch+1} U-ACT Train Loss: {train_loss:.4f}")

        # 使用修复后的评估模型
        val_loss, val_acc = evaluate(eval_model, dataloaders['val'], nn.CrossEntropyLoss(), device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        scheduler_backbone.step()

    print("\n--- U-ACT Training Finished ---")

    # 使用修复后的评估模型进行最终测试
    test_loss, test_acc = evaluate(eval_model, dataloaders['test'], nn.CrossEntropyLoss(), device)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")

    save_path = os.path.join(config['output_dir'], 'uact_model.pth')
    os.makedirs(config['output_dir'], exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    