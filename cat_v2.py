# cat_v2.py
import os, random, numpy as np, torch
SEED = 999
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

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

def build_loss_matrix(num_classes: int, device, penal_cfg: Dict[str, Any] = None):
    L = torch.eye(num_classes, device=device)
    if not penal_cfg:
        return L
    # on-diagonal: [(y, weight), ...]
    for y, w in penal_cfg.get("on_diagonal", []):
        L[y, y] = float(w)
    # pairs: [(y, k, w), ...]
    for y, k, w in penal_cfg.get("pairs", []):
        L[y, k] = float(w)
    # groups: [{"from":[...], "to":[...], "w":..}, ...]
    for g in penal_cfg.get("groups", []):
        w = float(g["w"])
        for y in g["from"]:
            for k in g["to"]:
                L[y, k] = w
    return L

def train_one_epoch_cat_v2(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer_backbone: optim.Optimizer,
    optimizer_h: optim.Optimizer,
    loss_matrix: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device
) -> float:
    model.train()
    running_loss = 0.0
    conf_config = config['conformal']
    size_weight_global = conf_config.get('size_weight', 1.0)
    ce_weight = conf_config.get('cross_entropy_weight', 1.0)
    target_size = conf_config.get('target_size', 1)

    progress_bar = tqdm(dataloader, desc="CAT Training")

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        B = inputs.shape[0]
        cal_B = int(B * conf_config['fraction'])
        if cal_B == 0 or cal_B == B:
            continue
        cal_inputs  = inputs[:cal_B]
        pred_inputs = inputs[cal_B:]
        cal_labels  = labels[:cal_B]
        pred_labels = labels[cal_B:]

        cal_logits, cal_h_x = model(cal_inputs)
        pred_logits, pred_h_x = model(pred_inputs)

        cal_probs  = torch.softmax(cal_logits,  dim=1)
        pred_probs = torch.softmax(pred_logits, dim=1)

        optimizer_backbone.zero_grad()

        smooth_sets = scp.smooth_predict_threshold(
            pred_probs, pred_h_x,
            temperature=conf_config.get('temperature', 10.0)
        )

        # coverage / size / CE
        coverage_loss = cputils.compute_coverage_loss(smooth_sets, pred_labels, loss_matrix)
        size_loss = cputils.compute_size_loss(smooth_sets, target_size=target_size)

        ce_loss = nn.CrossEntropyLoss()(pred_logits, pred_labels)
        loss_backbone = coverage_loss + size_weight_global * size_loss + ce_weight * ce_loss

        loss_backbone.backward()
        if conf_config.get("grad_clip_norm", None):
            torch.nn.utils.clip_grad_norm_(
                list(model.backbone.parameters()) + list(model.classifier.parameters()),
                max_norm=float(conf_config["grad_clip_norm"])
            )
        optimizer_backbone.step()

        optimizer_h.zero_grad()

        cal_conformity = cal_probs[torch.arange(cal_B, device=device), cal_labels]
        tau_cal = scp.smooth_conformal_quantile(
            cal_conformity,
            alpha=conf_config['alpha'],
            regularization_strength=conf_config.get('regularization_strength', 1e-3)
        ) 


        if cal_h_x.dim() == 1:
            cal_h_x = cal_h_x.unsqueeze(1)
        loss_h = nn.MSELoss()(cal_h_x, tau_cal.detach().expand_as(cal_h_x))
        loss_h.backward()
        if conf_config.get("h_grad_clip_norm", None):
            torch.nn.utils.clip_grad_norm_(model.threshold_net.parameters(),
                                           max_norm=float(conf_config["h_grad_clip_norm"]))
        optimizer_h.step()

        total_loss = loss_backbone.item() + loss_h.item()
        running_loss += total_loss * inputs.size(0)
        progress_bar.set_postfix(
            cov=f"{coverage_loss.item():.3f}",
            size=f"{size_loss.item():.3f}",
            ce=f"{ce_loss.item():.3f}",
            h=f"{loss_h.item():.3f}"
        )

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate_cp(model: nn.Module,
                calib_loader: DataLoader,
                test_loader: DataLoader,
                device: torch.device,
                cp_cfg: Dict[str, Any]):
    """
    返回：dict，含 Coverage、Ineff、(可选) 类条件覆盖/集合大小
    做法：
      1) 用 calib_loader 计算一致性分数 s(x)=p_y；
      2) 取 (1-alpha)*(1+1/n) 分位数得到 tau_cal（标量）；
      3) test_loader 上：对每个样本取 p>=max(h(x), tau_cal) 的类别作为集合（非平滑 step），统计覆盖与集合大小。
    说明：
      - 这里将个性化阈值 h(x) 与全局 tau_cal 取 max，可更稳（不低于全局要求）。
      - 若你想只用 h(x) 做自适应阈值，将 rule 改为 p>=h(x) 即可，但覆盖可能更敏感。
    """
    model.eval()

    # 1) 校准阈值（全局）
    cal_scores = []
    for x, y in calib_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=1)
        s = probs[torch.arange(len(y), device=device), y]
        cal_scores.append(s)
    cal_scores = torch.cat(cal_scores, dim=0)
    n = cal_scores.numel()
    alpha = float(cp_cfg['alpha'])
    q = torch.quantile(cal_scores, q=min(1.0, alpha*(1.0 + 1.0/max(1,n))))
    tau_cal = q.item()

    # 2) 测试集上统计 Coverage / Ineff
    total, covered, set_sizes = 0, 0, []
    per_class = {}  # y -> {"cnt":, "covered":, "set_sizes":[]}

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits, h = model(x)
        probs = torch.softmax(logits, dim=1)
        # 确保 h 形状 [B,1]
        if h.dim() == 1:
            h = h.unsqueeze(1)
        # 阈值：max(h(x), tau_cal)
        thr = torch.maximum(h, torch.ones_like(h) * tau_cal)

        # 非平滑集合：p_k >= thr 视为入集
        C = (probs >= thr).to(torch.int)   # [B, K]
        # 保障至少1类（极端情况下可能空集）
        empty = C.sum(dim=1) == 0
        if empty.any():
            top1 = probs.argmax(dim=1)
            C[torch.arange(C.size(0), device=device), top1] = 1

        # 覆盖统计
        inc = C[torch.arange(C.size(0), device=device), y].bool()
        covered += inc.sum().item()
        total += y.numel()
        set_sizes.extend(C.sum(dim=1).tolist())

        # 类条件
        for yi, inc_i, sz in zip(y.tolist(), inc.tolist(), C.sum(dim=1).tolist()):
            if yi not in per_class:
                per_class[yi] = {"cnt":0, "covered":0, "set_sizes":[]}
            per_class[yi]["cnt"] += 1
            per_class[yi]["covered"] += int(inc_i)
            per_class[yi]["set_sizes"].append(int(sz))

    cov = covered / max(1,total)
    ineff = float(np.mean(set_sizes)) if len(set_sizes)>0 else float('nan')
    class_cov = {k: v["covered"]/max(1,v["cnt"]) for k,v in per_class.items()}
    class_ineff = {k: float(np.mean(v["set_sizes"])) for k,v in per_class.items()}

    return {
        "coverage": cov,
        "ineff": ineff,
        "class_coverage": class_cov,
        "class_ineff": class_ineff,
        "tau_cal": tau_cal
    }


def run_cat_training_v2(config: Dict[str, Any]):
    device = config['device']
    print(f"Using device: {device}")
    train_cfg = config['training']
    hnet_cfg  = config['threshold_net']
    cp_cfg    = config['conformal']

    data_info = data.get_dataloaders(
        dataset_path=config['dataset_path'],
        batch_size=train_cfg['batch_size'],
        mean=config['mean'], std=config['std']
    )
    dls = data_info['dataloaders']
    num_classes = len(data_info['class_names'])

    model = models.get_model(
        model_type='cat',
        backbone_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=config['model']['pretrained']
    ).to(device)

    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total {tot/1e6:.2f}M, trainable {trainable/1e6:.2f}M")

    optimizer_backbone = optim.SGD(
        list(model.backbone.parameters()) + list(model.classifier.parameters()),
        lr=train_cfg['learning_rate'],
        momentum=train_cfg['momentum'],
        weight_decay=train_cfg['weight_decay']
    )
    optimizer_h = optim.Adam(
        model.threshold_net.parameters(),
        lr=hnet_cfg['learning_rate'],
        weight_decay=hnet_cfg['weight_decay']
    )
    scheduler_backbone = optim.lr_scheduler.StepLR(
        optimizer_backbone, step_size=train_cfg.get('lr_step', 30), gamma=train_cfg.get('lr_gamma', 0.1)
    )
    scheduler_h = optim.lr_scheduler.ReduceLROnPlateau(optimizer_h, mode='min', factor=0.5, patience=5)

    loss_matrix = build_loss_matrix(num_classes, device, config.get("penalties", None))

    eval_model = nn.Sequential(model.backbone, nn.Flatten(), model.classifier).to(device)

    model_p = "model_best.pth"
    if config['dataset'] == 'breakhis':
        save_dir = os.path.join(config['output_dir'], config['model']['name'], config['xs'])
    else:
        save_dir = os.path.join(config['output_dir'], config['model']['name'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_p)

    early_stopping = cputils.EarlyStopping(patience=train_cfg.get("patience", 15),
                                           verbose=True, path=save_path)

    print("\n--- Starting CAT Training (v2) ---")
    for epoch in range(train_cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{train_cfg['epochs']}")
        train_loss = train_one_epoch_cat_v2(
            model, dls['train'], optimizer_backbone, optimizer_h, loss_matrix, config, device
        )
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_acc = evaluate(eval_model, dls['val'], nn.CrossEntropyLoss(), device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler_backbone.step()
        scheduler_h.step(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\n--- Training Finished ---")
    print(f"Loading best model (val_loss={early_stopping.val_loss_min:.4f})")
    # model.load_state_dict(early_stopping.best_model_state_dict)


    # best_eval_model = nn.Sequential(model.backbone, nn.Flatten(), model.classifier).to(device)
    # test_loss, test_acc = evaluate(best_eval_model, dls['test'], nn.CrossEntropyLoss(), device)
    # print(f"Final Test (classification): loss={test_loss:.4f}, acc={test_acc:.4f}")

    # # CP 
    # cp_metrics = evaluate_cp(model, dls['val'], dls['test'], device, cp_cfg)
    # print(f"CP Metrics @alpha={cp_cfg['alpha']}: "
    #       f"coverage={cp_metrics['coverage']:.4f}, ineff={cp_metrics['ineff']:.4f}, "
    #       f"tau_cal={cp_metrics['tau_cal']:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")
