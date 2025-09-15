# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Datasets and data augmentation for the BRACS dataset using PyTorch.
This file provides utilities to load the BRACS dataset from image folders
and create DataLoader instances for training, validation, and testing.
"""
import os
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

def get_dataloaders(
    dataset_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    mean: Tuple[float, float, float] = (0.78, 0.62, 0.76), # IMPORTANT: Replace with your calculated values
    std: Tuple[float, float, float] = (0.15, 0.19, 0.14)   # IMPORTANT: Replace with your calculated values
) -> Dict[str, Any]:
    """
    Creates PyTorch DataLoaders for the BRACS dataset.

    This function sets up data augmentations for the training set and
    applies standard preprocessing for all data splits.

    Args:
        dataset_path (str): The root path to the BRACS dataset, 
                            containing 'train', 'val', and 'test' subdirectories.
        image_size (Tuple[int, int]): The target size to resize images to.
        batch_size (int): The number of samples per batch.
        mean (Tuple[float, float, float]): The per-channel mean of the training dataset for normalization.
        std (Tuple[float, float, float]): The per-channel standard deviation of the training dataset for normalization.

    Returns:
        Dict[str, Any]: A dictionary containing DataLoaders for 'train', 'val', and 'test',
                        along with dataset sizes and class names.
    """
    
    # --- 1. Define data transformations ---
    
    # Transformations for the training set, including data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), # Converts PIL image to tensor and scales pixels to [0, 1]
        transforms.Normalize(mean=mean, std=std) # Standardizes the tensor image
    ])
    
    # Transformations for the validation and test sets (no data augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # --- 2. Create Dataset instances using ImageFolder ---
    
    print("Loading datasets from disk...")
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

    print(f"Found {len(train_dataset.classes)} classes: {train_dataset.classes}")
    
    # --- 3. Create DataLoader instances ---
    
    # Recommended to use a number of workers for faster data loading
    # num_workers can be os.cpu_count() for maximum performance
    num_workers = 20

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    # --- 4. Package all information ---

    data_info = {
        'dataloaders': dataloaders,
        'sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset),
        },
        'class_names': train_dataset.classes
    }
    
    return data_info

if __name__ == '__main__':
    # --- This is a test script to verify the data loading process ---
    
    # IMPORTANT: Set this path to your 'latest_version' folder
    # which contains the 'train', 'val', and 'test' subfolders.
    path_to_bracs_latest_version = './histoimage.na.icar.cnr.it/BRACS_RoI/latest_version'
    
    if os.path.exists(path_to_bracs_latest_version):
        # You should run the calculate_stats.py script first to get these values
        # For now, we use placeholder values.
        print("WARNING: Using placeholder mean and std. Please calculate the actual values from your training set.")
        calculated_mean = (0.712733, 0.545225, 0.685850)
        calculated_std = (0.170330, 0.209620, 0.151623)

        data = get_dataloaders(
            dataset_path=path_to_bracs_latest_version,
            mean=calculated_mean,
            std=calculated_std
        )
        
        # Get one batch from the training loader to inspect it
        images, labels = next(iter(data['dataloaders']['train']))
        
        print("\n--- DataLoader Test ---")
        print(f"Dataset sizes: {data['sizes']}")
        print(f"Image batch shape: {images.shape}") # Should be [batch_size, 3, 224, 224]
        print(f"Label batch shape: {labels.shape}")   # Should be [batch_size]
        print(f"Example labels: {labels[:10]}")
    else:
        print(f"Test path '{path_to_bracs_latest_version}' does not exist. Please update the path to test.")