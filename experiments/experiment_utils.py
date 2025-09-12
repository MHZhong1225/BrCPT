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

"""Utilities for experiments."""
from typing import Sequence

import numpy as np


def loss_matrix_group_zero(
    off: float, on: float,
    groups: Sequence[int], classes: int) -> Sequence[float]:
  """
  创建一个损失矩阵，用于惩罚“属于组0的样本”的置信集中包含“组1的类别”。
  例如，在BRACS中，组0可以是良性类别，组1可以是恶性类别。
  这个损失会惩罚模型在判断一个良性样本时，把恶性类别也包含进来。
  """
  groups = np.array(groups)
  loss_matrix = np.eye(classes) * on
  true_indices = np.where(groups == 0)[0]
  pred_indices = np.where(groups == 1)[0]
  loss_matrix[np.ix_(true_indices, pred_indices)] = off
  np.fill_diagonal(loss_matrix, on)
  return tuple(loss_matrix.flatten())


def loss_matrix_group_one(
    off: float, on: float,
    groups: Sequence[int], classes: int) -> Sequence[float]:
  """
  与上面相反，惩罚“属于组1的样本”的置信集中包含“组0的类别”。
  """
  groups = np.array(groups)
  loss_matrix = np.eye(classes) * on
  true_indices = np.where(groups == 1)[0]
  pred_indices = np.where(groups == 0)[0]
  loss_matrix[np.ix_(true_indices, pred_indices)] = off
  np.fill_diagonal(loss_matrix, on)
  return tuple(loss_matrix.flatten())


def loss_matrix_importance(
    weights: Sequence[float], classes: int) -> Sequence[float]:
  """
  创建一个对角损失矩阵，对不同类别的“覆盖重要性”赋予不同权重。
  例如，我们可以让模型在确保覆盖恶性类别时的损失权重远大于良性类别，
  从而降低漏诊风险。
  """
  if len(weights) != classes:
      raise ValueError("Length of weights must be equal to the number of classes.")
  loss_matrix = np.eye(classes)
  np.fill_diagonal(loss_matrix, np.array(weights))
  return tuple(loss_matrix.flatten())


def loss_matrix_confusion(
    class_a: int, class_b: int, off_a_b: float, off_b_a: float,
    on: float, classes: int) -> Sequence[float]:
  """
  创建一个损失矩阵，专门用于惩罚某两个易混淆类别之间的“混淆”。
  """
  loss_matrix = np.eye(classes) * on
  loss_matrix[class_a, class_b] = off_a_b
  loss_matrix[class_b, class_a] = off_b_a
  return tuple(loss_matrix.flatten())


def size_weights_group(
    groups: Sequence[int], weights: Sequence[float]) -> Sequence[float]:
  """
  为不同组别的类别设置不同的大小损失权重。
  例如，我们可以允许模型在预测恶性样本时产生更大的置信集（更高的不确定性），
  而在预测良性样本时，则强制其产生更小的置信集。
  """
  groups = np.array(groups)
  weights = np.array(weights)
  unique_groups = np.unique(groups)
  if unique_groups.size != weights.size:
    raise ValueError('Number of groups and weights must match.')
  size_weights = np.zeros(groups.shape)
  for group, weight in zip(unique_groups, weights):
    size_weights[groups == group] = weight
  return tuple(size_weights)


def size_weights_selected(
    selected_classes: Sequence[int],
    weight: float, classes: int) -> Sequence[float]:
  """
  为特定的某个或某几个类别设置一个特殊的大小损失权重。
  """
  selected_classes = np.array(selected_classes)
  size_weights = np.ones(classes)
  size_weights[selected_classes] = weight
  return tuple(size_weights)