import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

# --- 1. 设置您的数据集路径 ---
# 请将此路径修改为您机器上 BRACS 数据集的 train 文件夹的绝对或相对路径
DATASET_PATH = './histoimage.na.icar.cnr.it/BRACS_RoI/latest_version/test'
IMAGE_SIZE = (224, 224) # 与您模型训练时使用的尺寸保持一致

# 检查路径是否存在
if not os.path.exists(DATASET_PATH):
    print(f"错误：路径 '{DATASET_PATH}' 不存在。请检查路径是否正确。")
    exit()

print("正在从文件夹加载图片...")
# 使用 Keras 工具加载数据集，但不打乱，以便我们遍历所有图片
# 注意：我们将加载整个训练集，这可能会消耗一些内存和时间
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    image_size=IMAGE_SIZE,
    batch_size=None,  # 加载所有图片
    shuffle=False
)

# 获取数据集中的图片总数
num_images = len(list(dataset))
print(f"成功加载 {num_images} 张训练图片。")

# --- 2. 高效计算均值和标准差 ---
# 这是一个高效的单遍算法，可以避免将所有图片同时加载到内存中

print("开始计算均值和标准差，这可能需要几分钟...")

# 初始化变量
mean_sum = np.zeros(3, dtype=np.float64)
std_sum_sq = np.zeros(3, dtype=np.float64)
pixel_count = 0

# 使用tqdm显示进度条
for image, label in tqdm(dataset, total=num_images, desc="正在处理图片"):
    # 将图片转换为numpy数组，并归一化到 [0, 1]
    image_np = image.numpy() / 255.0
    
    # 在 H 和 W 维度上求和，保留通道维度
    mean_sum += np.sum(image_np, axis=(0, 1))
    
    # 计算像素总数
    pixel_count += image_np.shape[0] * image_np.shape[1]

# 计算最终均值
mean = mean_sum / pixel_count

print("\n均值计算完成，正在计算标准差...")

# 再次遍历数据集以计算方差
for image, label in tqdm(dataset, total=num_images, desc="正在处理图片"):
    image_np = image.numpy() / 255.0
    
    # 累加 (x - mean)^2
    std_sum_sq += np.sum((image_np - mean) ** 2, axis=(0, 1))

# 计算最终标准差
std = np.sqrt(std_sum_sq / pixel_count)

# --- 3. 打印结果 ---
print("\n--- 计算完成 ---")
print("请将以下 `means` 和 `stds` 列表复制到您的 `data_utils.py` 文件中的 `get_data_stats` 函数内：\n")
print(f"    data['means'] = [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]  # R, G, B")
print(f"    data['stds'] = [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]  # R, G, B")