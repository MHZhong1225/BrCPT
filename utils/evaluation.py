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

"""Evaluation metrics for conformal prediction."""
from typing import Tuple

import jax
import jax.numpy as jnp


def _check_labels(probabilities: jnp.ndarray, labels: jnp.ndarray):
  """Helper to check shapes or probabilities/sets and labels."""
  if probabilities.ndim != 2:
    raise ValueError('Expecting probabilities/confidence sets of '
                     'shape n_examples x n_classes.')
  if labels.ndim != 1:
    raise ValueError('Expecting labels of shape n_examples.')
  if probabilities.shape[1] == 0:
    raise ValueError('Expecting at least one class.')
  if probabilities.shape[0] != labels.shape[0]:
    raise ValueError('Number of probabilities/confidence sets does '
                     'not match number of labels.')
  if not jnp.issubdtype(labels.dtype, jnp.integer):
    raise ValueError('Expecting labels to be integers.')
  if jnp.max(labels) >= probabilities.shape[1]:
    raise ValueError(
        'labels contains more classes than probabilities/confidence sets.')


def compute_accuracy(probabilities: jnp.ndarray, labels: jnp.ndarray) -> float:
  """Compute unconditional accuracy."""
  predictions = jnp.argmax(probabilities, axis=1)
  return jnp.mean(predictions == labels)

def compute_conditional_accuracy(
    probabilities: jnp.ndarray, labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """Computes conditional accuracy given a condition."""
  selected = (conditional_labels == conditional_label)
  num_examples = jnp.sum(selected)
  if num_examples == 0:
    return 1.0 # Return perfect accuracy if no examples match condition
  
  predictions = jnp.argmax(probabilities, axis=1)
  correct_predictions = (predictions == labels)
  
  accuracy = jnp.sum(jnp.where(selected, correct_predictions, 0)) / num_examples
  return accuracy


def compute_coverage(
    confidence_sets: jnp.ndarray, labels: jnp.ndarray) -> float:
  """
  Compute unconditional coverage.
  """
  return jnp.mean(confidence_sets[jnp.arange(len(labels)), labels])


def compute_conditional_coverage(
    confidence_sets: jnp.ndarray, labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """
  Compute conditional coverage.
  """
  selected = (conditional_labels == conditional_label)
  num_examples = jnp.sum(selected)
  if num_examples == 0:
      return 1.0 # Return perfect coverage if no examples match condition
  
  covered = confidence_sets[jnp.arange(len(labels)), labels]
  return jnp.sum(jnp.where(selected, covered, 0)) / num_examples


def compute_size(confidence_sets: jnp.ndarray) -> Tuple[float, int]:
  """
  Compute average confidence set size.
  """
  sizes = jnp.sum(confidence_sets, axis=1)
  return jnp.mean(sizes), confidence_sets.shape[0]


def compute_conditional_size(
    confidence_sets: jnp.ndarray,
    conditional_labels: jnp.ndarray,
    conditional_label: int) -> Tuple[float, int]:
  """
  Compute conditional confidence set size.
  """
  selected = (conditional_labels == conditional_label)
  num_examples = jnp.sum(selected)
  if num_examples == 0:
      return 0.0, 0
  
  sizes = jnp.sum(confidence_sets, axis=1)
  avg_size = jnp.sum(jnp.where(selected, sizes, 0)) / num_examples
  return avg_size, num_examples

def compute_miscoverage(
    confidence_sets: jnp.ndarray, one_hot_labels: jnp.ndarray) -> float:
  """
  Compute mis-coverage for given one-hot labels.
  Mis-coverage is the coverage for multiple labels as given
  in one_hot_labels that should not be included in the sets.
  """
  covered = jnp.clip(jnp.sum(confidence_sets * one_hot_labels, axis=1), 0, 1)
  return jnp.mean(covered)

def compute_conditional_miscoverage(
    confidence_sets: jnp.ndarray, one_hot_labels: jnp.ndarray,
    conditional_labels: jnp.ndarray, conditional_label: int) -> float:
  """Compute conditional mis-coverage for given one-hot labels."""
  selected = (conditional_labels == conditional_label)
  num_examples = jnp.sum(selected)
  if num_examples == 0:
      return 0.0

  covered = jnp.clip(jnp.sum(confidence_sets * one_hot_labels, axis=1), 0, 1)
  return jnp.sum(jnp.where(selected, covered, 0)) / num_examples