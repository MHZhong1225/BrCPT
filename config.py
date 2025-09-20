# config.py (PyTorch Version)

from typing import Dict, Any

# ---- Dataset-specific presets (请按你实际数据计算结果替换这些占位值) ----
DATASET_PRESETS = {
    "bach": {        # BACH
        "num_classes": 4,
        'mean': (0.718736, 0.617605, 0.842820),
        'std': (0.143265, 0.163461, 0.106455),
    },
    "breakhis": {    # BreakHis
        "num_classes": 8,
        'mean': (0.804441, 0.653422, 0.774287),  # 40X
        'std': (0.101077, 0.143443, 0.099306),    # 
    },
    "bracs": {    # BRACS
        "num_classes": 7,
        "mean": (0.712733, 0.545225, 0.685850),
        "std": (0.170330, 0.209620, 0.151623), 
    },
    "imagenet": {   # default
        "num_classes": 1000,
        "mean": (0.485, 0.456, 0.406),
        "std":  (0.229, 0.224, 0.225),
    },
}

def apply_dataset_preset(
    config: dict,
    dataset: str | None = None,
    mean: tuple[float,float,float] | None = None,
    std:  tuple[float,float,float] | None = None,
    num_classes: int | None = None
) -> dict:
    """
    按数据集名覆盖 config 里的 num_classes/mean/std；手动传入的 mean/std/num_classes 优先级最高。
    """
    cfg = dict(config)  # 浅拷贝
    if dataset:
        preset = DATASET_PRESETS.get(dataset.lower())
        if preset:
            cfg["num_classes"] = preset["num_classes"]
            cfg["mean"] = preset["mean"]
            cfg["std"]  = preset["std"]
        else:
            print(f"[WARN] Unknown dataset preset: {dataset}. Using defaults in get_config().")
    # 手动覆盖优先
    if num_classes is not None:
        cfg["num_classes"] = num_classes
    if mean is not None:
        cfg["mean"] = mean
    if std is not None:
        cfg["std"] = std
    return cfg

def get_config(
    dataset: str | None = None,
    *,
    mean: tuple[float,float,float] | None = None,
    std:  tuple[float,float,float] | None = None,
    num_classes: int | None = None,
) -> Dict[str, Any]:
    """
    Returns the default configuration dictionary for the project.
    """
    config = {
        # --- General Settings ---
        "seed": 42,
        "device": "cuda", # "cuda" or "cpu"

        # --- Data Settings ---
        "dataset_path": "./histoimage.na.icar.cnr.it/BRACS_RoI/latest_version", # IMPORTANT: Update this path
        "image_size": (224, 224),
        "num_classes": 7,
        # IMPORTANT: Replace these with the actual values calculated from your training set

        
        # --- Model Settings ---
        "model": {
            "name": "resnet50",
            "pretrained": True,
        },

        # --- Training Settings ---
        "training": {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "momentum": 0.9,
            "weight_decay": 5e-4,
        },
        
        # --- Output Settings ---
        "output_dir": "./experiments/",

        # --- Conformal Training Specific Settings ---
        "conformal": {
            "alpha": 0.1,  # Target miscoverage rate (e.g., 0.1 for 90% coverage)
            "fraction": 0.5, # Split ratio for cal/pred sets in a batch (50% / 50%)
            "temperature": 0.1, # For smooth_predict_threshold
            "regularization_strength": 0.01, # For torchsort's soft_quantile
            "size_weight": 0.1, # Weight for the size loss term
            "cross_entropy_weight": 0.1, # Optional weight for standard CE loss for stability
        },
        
        "threshold_net": {
            "learning_rate": 0.001, # 阈值网络可以使用独立的学习率
            "weight_decay": 1e-5,
            "hidden_dim": 128      # 阈值网络MLP的隐藏层维度
        }
    }
    
    config = apply_dataset_preset(
        config, dataset=dataset, mean=mean, std=std, num_classes=num_classes
    )
    return config

if __name__ == '__main__':
    # This is a simple test to print out the default configuration.
    default_config = get_config()
    import json
    print(json.dumps(default_config, indent=2))
