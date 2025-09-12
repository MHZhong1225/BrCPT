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

"""Train models for experiments."""
from absl import logging
import haiku as hk
import ml_collections as collections

from utils import data_utils as cpdatautils
import train_conformal as cpconformal
import train_normal as cpnormal
from utils import train_utils as cputils


def train(config: collections.ConfigDict):
  """
  The main training function that orchestrates the entire process.
  """
  rng = hk.PRNGSequence(config.seed)
  data = cpdatautils.get_data(config)
  logging.info('Loaded dataset: %s', config.dataset)

  # Setup Learning Rate Scheduler
  if config.learning_rate_schedule == 'step':
    lr_scheduler_cls = cputils.MultIStepLRScheduler
    args = {'learning_rate_decay': config.step.learning_rate_decay}
  else:
    raise ValueError(f'Invalid learning rate schedule: {config.learning_rate_schedule}')
  
  lr_scheduler = lr_scheduler_cls(
      learning_rate=config.learning_rate,
      num_examples=data['sizes']['train'], batch_size=config.batch_size,
      epochs=config.epochs, **args)
  
  # Setup Optimizer
  if config.optimizer == 'sgd':
    optimizer = cputils.get_sgd_optimizer(
        config.sgd.momentum, config.sgd.nesterov, lr_scheduler)
  elif config.optimizer == 'adam':
    optimizer = cputils.get_adam_optimizer(
        config.adam.b1, config.adam.b2, config.adam.eps, lr_scheduler)
  else:
    raise ValueError(f'Invalid optimizer: {config.optimizer}.')
  logging.info('Loaded optimizer: %s', config.optimizer)

  # --- The Core Dispatcher ---
  # Based on the config, decide which trainer to use.
  if config.mode == 'normal':
    logging.info("Initializing trainer in 'normal' mode.")
    trainer = cpnormal.TrainNormal(config, data, optimizer)
  elif config.mode == 'conformal':
    logging.info("Initializing trainer in 'conformal' mode.")
    trainer = cpconformal.TrainConformal(config, data, optimizer)
  else:
    raise ValueError(f'Invalid training mode: {config.mode}.')
  
  # Start the training run
  trainer.run(rng)
  