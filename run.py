# run.py (Correct PyTorch Version)

import argparse
import os
import time
from typing import Dict, Any, Tuple
from PIL import Image

from train_uact import run_uact_training
Image.MAX_IMAGE_PIXELS = None
# Import the main training functions and the config loader
from config import get_config
from train_normal import run_normal_training
from train_conformal import run_conformal_training

def main(args: argparse.Namespace):
    """
    The main entry point for running experiments.
    """
    # --- 1. Load and customize configuration ---
    config = get_config()
    
    # Update config with command-line arguments if they are provided
    config['training']['epochs'] = args.epochs or config['training']['epochs']
    config['training']['learning_rate'] = args.lr or config['training']['learning_rate']
    config['training']['batch_size'] = args.batch_size or config['training']['batch_size']
    config['dataset_path'] = args.dataset_path or config['dataset_path']

    # Override conformal weights if provided
    if args.size_weight is not None:
        config['conformal']['size_weight'] = args.size_weight
    if args.ce_weight is not None:
        config['conformal']['cross_entropy_weight'] = args.ce_weight

    # Create a unique output directory for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.mode}_{timestamp}"
    config['output_dir'] = os.path.join(config['output_dir'], run_name)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("--- Configuration for this run ---")
    for section, settings in config.items():
        print(f"[{section}]")
        if isinstance(settings, dict):
            for key, value in settings.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {settings}")
    print("------------------------------------")

    # --- 2. Dispatch to the correct training function based on mode ---
    if args.mode == 'normal':
        run_normal_training(config)
    elif args.mode == 'conformal':
        run_conformal_training(config)
    elif args.mode == 'uact':
        run_uact_training(config)
    else:
        raise ValueError(f"Invalid mode. Choose 'normal', 'conformal', or 'uact'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training for Conformal Prediction.")
    
    parser.add_argument('--mode', type=str, default='uact', choices=['normal', 'conformal', 'uact'],
                        help="The training mode to run ('normal' for baseline, 'conformal' for our method).")
    
    # Optional arguments to override the config file
    parser.add_argument('--dataset_path', type=str, default=None,
                        help="Path to the BRACS dataset root folder. Overrides config default.")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of training epochs. Overrides config default.")
    parser.add_argument('--lr', type=float, default=None,
                        help="Learning rate. Overrides config default.")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size. Overrides config default.")
    parser.add_argument('--size_weight', type=float, default=None,
                        help="Weight for size loss (overrides config['conformal']['size_weight']).")
    parser.add_argument('--ce_weight', type=float, default=None,
                        help="Weight for cross-entropy loss (overrides config['conformal']['cross_entropy_weight']).")

    args = parser.parse_args()
    main(args)