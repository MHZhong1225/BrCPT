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

"""Utils for file I/O, particularly for saving and loading checkpoints."""
import os
import pickle
from typing import Any, Tuple, Dict

from absl import logging
import jax.numpy as jnp
import ml_collections as collections


def _dump_pickle(mixed: Any, path: str):
  """Write data to a pickle file."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'wb') as f:
    pickle.dump(mixed, f)
  logging.info('Wrote %s', path)


def _load_pickle(path: str) -> Any:
  """Load data from a pickle file."""
  with open(path, 'rb') as f:
    mixed = pickle.load(f)
  logging.info('Read %s', path)
  return mixed


class Checkpoint:
  """Checkpoint to save and load models."""

  class State:
    """State holding parameters, model and optimizer state and epoch."""
    def __init__(self):
      self.params = None
      self.model_state = None
      self.optimizer_state = None
      self.epoch = None

  def __init__(self, path: str = './'):
    """Create a checkpoint in the provided path."""
    self.state = Checkpoint.State()
    self.path = path
    self.params_file = os.path.join(self.path, 'params.pkl')
    self.model_state_file = os.path.join(self.path, 'model_state.pkl')
    self.optimizer_state_file = os.path.join(self.path, 'optimizer_state.pkl')
    self.epoch_file = os.path.join(self.path, 'epoch.pkl')

  def _exists(self):
    """Check if checkpoint exists."""
    return all(os.path.isfile(p) for p in [
        self.params_file, self.model_state_file,
        self.optimizer_state_file, self.epoch_file,
    ])

  def restore(self):
    """Restore checkpoint from files."""
    if not self._exists():
      raise FileNotFoundError(f'Checkpoint {self.path} not found.')
    self.state.params = _load_pickle(self.params_file)
    self.state.model_state = _load_pickle(self.model_state_file)
    self.state.optimizer_state = _load_pickle(self.optimizer_state_file)
    self.state.epoch = _load_pickle(self.epoch_file)

  def save(self):
    """Save checkpoint to files."""
    os.makedirs(self.path, exist_ok=True)
    _dump_pickle(self.state.params, self.params_file)
    _dump_pickle(self.state.model_state, self.model_state_file)
    _dump_pickle(self.state.optimizer_state, self.optimizer_state_file)
    _dump_pickle(self.state.epoch, self.epoch_file)

  def restore_or_save(self):
    """Restore or save checkpoint."""
    if self._exists():
      self.restore()
    else:
      self.save()


def create_checkpoint(config: collections.ConfigDict) -> Checkpoint:
  """Create a checkpoint."""
  return Checkpoint(config.path)


def load_checkpoint(config: collections.ConfigDict) -> Tuple[Checkpoint, str]:
  """Loads the checkpoint using the provided config.path."""
  checkpoint = Checkpoint(config.path)
  checkpoint.restore()
  return checkpoint, config.path


class PickleWriter:
  """Pickle writer to save evaluation."""
  def __init__(self, path: str, name: str):
    self.path = os.path.join(path, name + '.pkl')

  def write(self, values: Any):
    _dump_pickle(values, self.path)


def create_writer(config: collections.ConfigDict, key: str) -> Any:
  """Create a writer to save evaluation results."""
  return PickleWriter(config.path, key)


class PickleReader:
  """Pickle reader to load evaluation."""
  def __init__(self, path: str, name: str):
    self.path = os.path.join(path, name + '.pkl')

  def read(self) -> Any:
    return _load_pickle(self.path)


def load_predictions(
    path: str, val_examples: int = 0) -> Dict[str, Any]:
  """Load model predictions/logits for a specific experiment."""
  test_reader = PickleReader(path, 'eval_test')
  eval_test = test_reader.read()

  model = {
      'data': {'groups': {}, 'classes': eval_test['logits'].shape[1]},
      'test_logits': eval_test['logits'],
      'test_labels': eval_test['labels'],
      'val_logits': jnp.array([]),
      'val_labels': jnp.array([]),
  }

  logging.info('Loaded %s: %d test examples', path, model['test_labels'].shape[0])

  if val_examples > 0 and os.path.exists(os.path.join(path, 'eval_val.pkl')):
    val_reader = PickleReader(path, 'eval_val')
    eval_val = val_reader.read()
    model['val_logits'] = eval_val['logits']
    model['val_labels'] = eval_val['labels']
    logging.info('Loaded %s: %d val examples', path, model['val_labels'].shape[0])

  return model