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

"""Launch definitions for our BRACS experiments."""
from typing import Tuple, Dict, Any, Optional

import ml_collections as collections

# 我们将很快创建这个文件
# import experiments.experiment_utils as cpeutils


def get_parameters(
    experiment: str,
    sub_experiment: str,
    config: collections.ConfigDict,
) -> Tuple[collections.ConfigDict, Optional[Dict[str, Any]]]:
  """Get parameters for BRACS experiments."""
  
  # --- 基本配置 (适用于所有BRACS实验) ---
  config.architecture = 'resnet'
  config.resnet.version = 50
  config.resnet.channels = 64
  config.epochs = 100
  config.dataset = 'bracs' # 确保数据集名称正确
  # 假设您的BRACS数据存放路径，后续可以通过命令行修改
  config.dataset_path = '/path/to/your/bracs_dataset' 

  parameter_sweep = None

  # --- 实验一: 标准交叉熵训练 (我们的基线) ---
  if experiment == 'baseline':
    config.mode = 'normal'
    config.learning_rate = 0.05
    config.batch_size = 32

  # --- 实验二: 置信训练 (我们的核心方法) ---
  elif experiment == 'conformal':
    config.mode = 'conformal'
    config.conformal.coverage_loss = 'none' # 先从最简单的开始
    config.conformal.loss_transform = 'log'
    config.conformal.rng = False

    # 这是一个子实验，名为 'conformal.training'
    if sub_experiment == 'training':
      config.learning_rate = 0.01
      config.batch_size = 32
      config.conformal.temperature = 1.0
      config.conformal.size_loss = 'valid' # 惩罚大小 < 1 的置信集
      config.conformal.method = 'threshold_logp'
      config.conformal.size_weight = 0.05
    
    # --- 实验三: 带有临床风险控制的置信训练 (高级) ---
    # 这是一个子实验，名为 'conformal.clinical_risk'
    elif sub_experiment == 'clinical_risk':
      config.learning_rate = 0.01
      config.batch_size = 32
      config.conformal.temperature = 1.0
      config.conformal.size_loss = 'valid'
      config.conformal.method = 'threshold_logp'
      config.conformal.size_weight = 0.1
      
      # 启用我们自定义的分类损失
      config.conformal.coverage_loss = 'classification'
      
      # 在这里定义我们的非对称损失矩阵
      # 假设类别 7 是恶性类别，我们想加大惩罚
      # loss_matrix = cpeutils.loss_matrix_importance((1, 1, 1, 1, 1, 1, 1, 5.0), 8)
      # config.conformal.loss_matrix = loss_matrix
      # (注意: cpeutils.py 我们还没创建，所以暂时注释掉)
      
    else:
      raise ValueError(f'Conformal sub-experiment "{sub_experiment}" not implemented.')
  
  else:
    raise ValueError(f'Experiment "{experiment}" not implemented.')
    
  return config, parameter_sweep