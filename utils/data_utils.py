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

"""Utilities for loading datasets for training."""
import functools
from typing import Dict, Any, Tuple

from absl import logging
import jax.numpy as jnp
import ml_collections as collections
import tensorflow as tf

# 导入我们上一步创建的 data.py
from utils import data as cpdata


def apply_histopathology_augmentation(
    ds: tf.data.Dataset) -> tf.data.Dataset:
  """
  为我们的病理图像数据集应用数据增强。
  
  Args:
    ds: 需要增强的 tf.data.Dataset 对象。

  Returns:
    增强后的数据集。
  """
  # 我们使用在 data.py 中定义的包含颜色增强的函数
  augment_fn = cpdata.augment_histopathology_with_color
  ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
  return ds


def get_data_stats(config: collections.ConfigDict) -> Dict[str, Any]:
  """获取所选数据集的统计信息。"""
  data = {}
  if config.dataset == 'bracs':
    data['classes'] = 7  # <-- 根据您的文件夹，从0到6一共7个类别

    # 运行calculate_stats
    data['means'] = [0.727460, 0.557052, 0.693719]  # R, G, B
    data['stds'] = [0.197257, 0.234502, 0.176810]  # R, G, B
    # 图像尺寸，这个值将传递给 data.py
    data['shape'] = (224, 224, 3) 
  else:
    raise ValueError(f"Dataset '{config.dataset}' is not supported.")

  data['means'] = jnp.array(data['means'])
  data['stds'] = jnp.array(data['stds'])
  return data


def _check_batch_sizes(config: collections.ConfigDict, data: Dict[str, Any]):
  """
  辅助函数，检查数据集大小是否可以被批次大小整除。
  """
  for key, batch_size in zip(
      ['train', 'test', 'val'],
      [config.batch_size, config.test_batch_size, config.test_batch_size]):
    if data['sizes'].get(key, 0) % batch_size != 0:
      # 在Conformal training中，由于平滑排序的需要，批次大小通常需要固定
      # 因此如果不能整除，我们通常会选择丢弃最后一个不完整的批次
      logging.warning(
          'Dataset split "%s" with size %d is not divisible by batch size %d. '
          'The last batch will be dropped.',
          key, data['sizes'][key], batch_size)


def _batch_sets(
    config: collections.ConfigDict, data: Dict[str, Any], drop_remainder: bool):
  """
  辅助函数，负责数据集的打乱、批处理和重采样。
  """
  if data['sizes']['train'] % config.batch_size != 0:
    drop_remainder = True

  # 为评估创建一个有序、未经增强的训练集版本
  data['train_clean'] = data['train'].batch(
      config.batch_size, drop_remainder=drop_remainder)
  
  # 对训练数据进行打乱
  data['train'] = data['train'].shuffle(
      buffer_size=data['sizes']['train'], seed=config.seed, reshuffle_each_iteration=True)

  # 应用数据增强
  logging.info('Applying histopathology data augmentation.')
  data['train'] = apply_histopathology_augmentation(data['train'])

  # 将数据打包成批次
  data['train'] = data['train'].batch(
      config.batch_size, drop_remainder=drop_remainder)
  data['train'] = data['train'].prefetch(tf.data.AUTOTUNE) # 提升性能

  if data.get('val'):
    data['val'] = data['val'].batch(
        config.test_batch_size, drop_remainder=drop_remainder)
    data['val'] = data['val'].prefetch(tf.data.AUTOTUNE)
  
  if data.get('test'):
    data['test'] = data['test'].batch(
        config.test_batch_size, drop_remainder=drop_remainder)
    data['test'] = data['test'].prefetch(tf.data.AUTOTUNE)


def get_data(config: collections.ConfigDict) -> Dict[str, Any]:
  """获取用于训练和测试的完整数据流程。"""

  def preprocess_image(batch):
    """通用预处理：将像素值从 [0, 255] 归一化到 [0, 1]。"""
    return {
        'image': tf.cast(batch['image'], tf.float32) / 255.0,
        'label': batch['label']
    }

  data = get_data_stats(config)
  config.data_shape = data['shape'] 

  if config.dataset == 'bracs':
      data_split = cpdata.create_data_split(
          dataset_path=config.dataset_path, 
          image_size=(config.data_shape[0], config.data_shape[1]),
          batch_size=config.batch_size) # 传入batch_size以帮助加载

      # 将加载的数据集信息更新到data字典中
      data.update(data_split)

      # 应用预处理
      data['train'] = data['train'].map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
      data['val'] = data['val'].map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) if data.get('val') else None
      data['test'] = data['test'].map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) if data.get('test') else None
  else:
    raise ValueError(f"Dataset loading for '{config.dataset}' is not implemented.")

  # _batch_sets 函数负责打乱、增强和最终的批处理
  _batch_sets(config, data, drop_remainder=True)
  return data
  