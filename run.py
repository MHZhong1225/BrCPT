# run.py

import argparse
import os
from PIL import Image
import torch
Image.MAX_IMAGE_PIXELS = None
from config import get_config
from train_normal import run_normal_training
from train_conformal import run_conformal_training
from cat import run_cat_training

def main():
    parser = argparse.ArgumentParser(description="Run training for Conformal Prediction.")
    parser.add_argument('--cuda', type=str, default='1', help="Device to use, e.g., 'cpu', '0', '1'.")
    parser.add_argument('--mode', type=str, default='cat', choices=['normal', 'conformal', 'cat'])

    parser.add_argument('--dataset', type=str, default='breakhis', choices=['bracs', 'bach', 'breakhis'])
    parser.add_argument('--num_classes', type=int, default=7,
                        help="Number of classes in the dataset. Overrides config default.")
    parser.add_argument('--xs', type=str, default='40X', choices=['40X', '100X', '200X', '400X'])
    parser.add_argument('--alpha', type=float, default=0.05)

    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50','efficientnet_b0'])
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save experiment results. Overrides config default.")

    # training
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of training epochs. Overrides config default.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size. Overrides config default.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for the backbone. Overrides config default.")
    parser.add_argument('--lr_h', type=float, default=1e-3)


    # conformal
    parser.add_argument("--size_weight", type=float, default=0.05)
    parser.add_argument("--regularization_strength", type=float, default=0.01)
    parser.add_argument("--cross_entropy_weight", type=float, default=0.1)
    parser.add_argument("--h_only", type=bool, default=False)


    args = parser.parse_args()

    config = get_config(dataset=args.dataset)
    args.output_dir = f'./experiments/{args.dataset}'
    if args.dataset: 
        config['dataset_path'] = f'./datasets/{args.dataset}' 
        if args.dataset == 'breakhis': config['dataset_path'] = f'./datasets/{args.dataset}/{args.xs}' 
    if args.num_classes: config['num_classes'] = args.num_classes
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.lr: config['training']['learning_rate'] = args.lr
    if args.lr_h: config['threshold_net']['learning_rate'] = args.lr_h


    if args.output_dir: config['output_dir'] = args.output_dir
    if args.model: config['model']['name'] = args.model
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f'cuda:{args.cuda}')
    config['device'] = device 
    config['xs'] = args.xs
    config['dataset'] = args.dataset
    is_conf_mode = args.mode in ("conformal", "cat")

    if is_conf_mode:
        conf = config.setdefault("conformal", {})
        if args.size_weight is not None:
            conf["size_weight"] = args.size_weight
        if args.cross_entropy_weight is not None:
            conf["cross_entropy_weight"] = args.cross_entropy_weight
        if args.regularization_strength is not None:
            conf["regularization_strength"] = args.regularization_strength
        if args.alpha is not None:
            conf["alpha"] = args.alpha
        conf["h_only"] = args.h_only
    else:
        config.pop("conformal", None)

    run_name = f"{args.mode}"
    config['output_dir'] = os.path.join(config['output_dir'], run_name)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("--- Configuration for this run ---")
    for section, settings in config.items():
        if isinstance(settings, dict):
            print(f"[{section}]")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {settings}")
    print("------------------------------------")

    if args.mode == 'normal':
        run_normal_training(config)
    elif args.mode == 'conformal':
        run_conformal_training(config)
    elif args.mode == 'cat':
        run_cat_training(config)
    else:
        raise ValueError(f"Invalid mode specified: {args.mode}.")

if __name__ == '__main__':
    main()
