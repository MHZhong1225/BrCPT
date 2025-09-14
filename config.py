# config.py (PyTorch Version)

from typing import Dict, Any

def get_config() -> Dict[str, Any]:
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
        "mean": (0.712733, 0.545225, 0.685850), # Placeholder
        "std": (0.170330, 0.209620, 0.151623),  # Placeholder
        
        # --- Model Settings ---
        "model": {
            "name": "resnet50",
            "pretrained": True,
        },

        # --- Training Settings ---
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "momentum": 0.9,
            "weight_decay": 5e-4,
        },
        
        # --- Output Settings ---
        "output_dir": "./experiments/bracs",

        # --- Conformal Training Specific Settings ---
        "conformal": {
            "alpha": 0.1,  # Target miscoverage rate (e.g., 0.1 for 90% coverage)
            "fraction": 0.5, # Split ratio for cal/pred sets in a batch (50% / 50%)
            "temperature": 0.1, # For smooth_predict_threshold
            "regularization_strength": 0.01, # For torchsort's soft_quantile
            "size_weight": 0.3, # Weight for the size loss term
            "cross_entropy_weight": 0.1, # Optional weight for standard CE loss for stability
        }
    }
    
    return config

if __name__ == '__main__':
    # This is a simple test to print out the default configuration.
    default_config = get_config()
    import json
    print(json.dumps(default_config, indent=2))
