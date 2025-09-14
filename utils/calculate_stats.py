import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# --- 1. 设置您的数据集路径和图像尺寸 ---
# 请将此路径修改为您机器上 BRACS 数据集的 train 文件夹的绝对或相对路径
DATASET_PATH = './histoimage.na.icar.cnr.it/BRACS_RoI/latest_version/train'
IMAGE_SIZE = (224, 224) # 与您模型训练时使用的尺寸保持一致
BATCH_SIZE = 32 # 可以根据您的内存大小调整批次

# 检查路径是否存在
if not os.path.exists(DATASET_PATH):
    print(f"错误：路径 '{DATASET_PATH}' 不存在。请检查路径是否正确。")
    exit()

print("正在创建PyTorch数据集...")
# --- 2. 使用 torchvision 加载数据集 ---
# 定义一个简单的变换，只将图片转换为Tensor，以便计算
# 注意：这里我们不进行标准化，因为我们的目的就是计算标准化的参数
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor() # 将PIL图片转换为[C, H, W]格式的Tensor，并将像素值缩放到[0, 1]
])

# ImageFolder会自动从文件夹结构中加载数据和标签
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# 创建一个DataLoader来高效地遍历数据
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_images = len(dataset)
print(f"成功加载 {num_images} 张训练图片。")

# --- 3. 计算均值和标准差 ---
print("正在计算均值...")

# 初始化变量
# 我们需要对每个通道(R, G, B)分别计算
mean = torch.zeros(3)
std = torch.zeros(3)
num_batches = 0

# 单遍计算均值
for images, _ in tqdm(dataloader, desc="计算均值"):
    # images 的形状是 [B, C, H, W]
    # 我们在 batch, height, width 维度上求均值
    batch_mean = torch.mean(images, dim=[0, 2, 3])
    mean += batch_mean
    num_batches += 1
mean /= num_batches

print(f"\n均值计算完成: {mean.tolist()}")
print("正在计算标准差...")

# 单遍计算标准差
sum_sq_diff = torch.zeros(3)
num_batches = 0
for images, _ in tqdm(dataloader, desc="计算标准差"):
    # 计算 (x - mean)^2 的均值
    # (images - mean.view(3, 1, 1)) 会自动广播mean
    batch_var = torch.mean((images - mean.view(3, 1, 1)) ** 2, dim=[0, 2, 3])
    sum_sq_diff += batch_var
    num_batches += 1
std = torch.sqrt(sum_sq_diff / num_batches)

print(f"标准差计算完成: {std.tolist()}")


# --- 4. 打印结果 ---
print("\n--- 计算完成 ---")
print("请将以下 `mean` 和 `std` 元组复制到您的 `config.py` 文件中：\n")
mean_list = mean.tolist()
std_list = std.tolist()
print(f"    'mean': ({mean_list[0]:.6f}, {mean_list[1]:.6f}, {mean_list[2]:.6f}),  # R, G, B")
print(f"    'std': ({std_list[0]:.6f}, {std_list[1]:.6f}, {std_list[2]:.6f}),    # R, G, B")