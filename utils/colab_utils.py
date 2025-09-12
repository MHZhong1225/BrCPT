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

"""Utils for evaluation."""
from typing import Tuple, Callable, Dict, Any, List

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import sklearn.metrics

import conformal_prediction as cp
import evaluation as cpeval
import open_source_utils as cpstaging # 我们将在下一步创建这个文件


_CalibrateFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
_PredictFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
load_predictions = cpstaging.load_predictions


def get_threshold_fns(
    alpha: float, jit: bool = True) -> Tuple[_CalibrateFn, _PredictFn]:
  """Prediction and calibration function for threshold conformal prediction."""
  def calibrate_threshold_fn(logits, labels, rng):  # pylint: disable=unused-argument
    probabilities = jax.nn.softmax(logits, axis=1)
    return cp.calibrate_threshold(
        probabilities, labels, alpha=alpha)
  def predict_threshold_fn(logits, tau, rng):  # pylint: disable=unused-argument
    probabilities = jax.nn.softmax(logits, axis=1)
    return cp.predict_threshold(
        probabilities, tau)
  if jit:
    calibrate_threshold_fn = jax.jit(calibrate_threshold_fn)
    predict_threshold_fn = jax.jit(predict_threshold_fn)
  return calibrate_threshold_fn, predict_threshold_fn


def get_raps_fns(
    alpha: float, k_reg: int, lambda_reg: float,
    jit: bool = True) -> Tuple[_CalibrateFn, _PredictFn]:
  """Prediction and calibration function for RAPS."""
  def calibrate_raps_fn(logits, labels, rng):
    probabilities = jax.nn.softmax(logits, axis=1)
    return cp.calibrate_raps(
        probabilities, labels, alpha=alpha,
        k_reg=k_reg, lambda_reg=lambda_reg, rng=rng)
  def predict_raps_fn(logits, tau, rng):
    probabilities = jax.nn.softmax(logits, axis=1)
    return cp.predict_raps(
        probabilities, tau, k_reg=k_reg, lambda_reg=lambda_reg, rng=rng)
  if jit:
    calibrate_raps_fn = jax.jit(calibrate_raps_fn)
    predict_raps_fn = jax.jit(predict_raps_fn)
  return calibrate_raps_fn, predict_raps_fn


def get_groups(dataset: str, key: str) -> jnp.ndarray:
  """Helper to define groups for evaluation."""
  if dataset == 'bracs':
    if key == 'groups':
      # 定义BRACS的临床分组:
      # 类别 0, 1, 2 是良性 (Benign) -> 组 0
      # 类别 3, 4, 5, 6 是非典型或恶性 (Atypical/Malignant) -> 组 1
      groups = jnp.array([0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
    else:
      raise NotImplementedError(f"Group key '{key}' not implemented for bracs.")
  else:
    raise ValueError('No groups defined for dataset %s.' % dataset)

  return groups


def _evaluate_accuracy(
    logits: jnp.ndarray, labels: jnp.ndarray) -> pd.DataFrame:
  """Helper to compute accuracy on single dataset."""
  classes = logits.shape[1]
  probabilities = jax.nn.softmax(logits, axis=1)
  accuracy = float(cpeval.compute_accuracy(probabilities, labels))
  accuracies = [float(cpeval.compute_conditional_accuracy(
        probabilities, labels, labels, k)) for k in range(classes)]
  columns = ['accuracy'] + [f'accuracy_{i}' for i in range(classes)]
  data = np.array([accuracy] + accuracies)
  return pd.DataFrame(np.expand_dims(data, axis=0), columns=columns)


def evaluate_metrics(
    data: Dict[str, Any], logits: jnp.ndarray,
    confidence_sets: jnp.ndarray, labels: jnp.ndarray) -> List[pd.DataFrame]:
  """Evaluate all metrics on a validation or test set."""
  accuracy_df = _evaluate_accuracy(logits, labels)
  
  # --- 计算其他所有指标 ---
  # (这部分代码比较长，但逻辑是重复的：为每个指标创建一个字典，然后转成DataFrame)
  
  # Coverage
  coverage = float(cpeval.compute_coverage(confidence_sets, labels))
  coverage_values = {'coverage': coverage}
  for k in range(data['classes']):
      coverage_values[f'class_coverage_{k}'] = float(cpeval.compute_conditional_coverage(
          confidence_sets, labels, labels, k))
  coverage_df = pd.DataFrame([coverage_values])

  # Size
  size, count = cpeval.compute_size(confidence_sets)
  size_values = {'size': float(size), 'count': count}
  for k in range(data['classes']):
      size_values[f'class_size_{k}'], _ = cpeval.compute_conditional_size(
          confidence_sets, labels, k)
      size_values[f'class_size_{k}'] = float(size_values[f'class_size_{k}'])
  size_df = pd.DataFrame([size_values])
  
  results = [accuracy_df, coverage_df, size_df]
  return results


def evaluate_conformal_prediction(
    model: Dict[str, Any], calibrate_fn: _CalibrateFn, predict_fn: _PredictFn,
    trials: int, rng: jnp.ndarray) -> Dict[str, Any]:
  """
  The main evaluation loop.
  It applies calibration and prediction on multiple random splits of the data.
  """
  keys = model.keys()
  if 'val_logits' not in keys or 'val_labels' not in keys:
    raise ValueError('val_logits or val_labels not present.')
  if 'test_logits' not in keys or 'test_labels' not in keys:
    raise ValueError('test_logits or test_labels not present.')

  rngs = jax.random.split(rng, 3*trials)
  val_examples = model['val_labels'].shape[0]
  
  # Combine validation and test sets for random splitting
  logits = jnp.concatenate((model['val_logits'], model['test_logits']), axis=0)
  labels = jnp.concatenate((model['val_labels'], model['test_labels']), axis=0).astype(int)
  num_examples = labels.shape[0]

  all_results = []
  for t in range(trials):
    perm_rng, cal_rng, test_rng = rngs[3*t : 3*t + 3]
    
    # Randomly shuffle and split data into new calibration and test sets
    perm = jax.random.permutation(perm_rng, jnp.arange(num_examples))
    cal_logits_t = logits[perm[:val_examples]]
    cal_labels_t = labels[perm[:val_examples]]
    test_logits_t = logits[perm[val_examples:]]
    test_labels_t = labels[perm[val_examples:]]

    # Perform calibration and prediction
    tau = calibrate_fn(cal_logits_t, cal_labels_t, cal_rng)
    test_confidence_sets_t = predict_fn(test_logits_t, tau, test_rng)
    
    # Evaluate all metrics for this trial
    trial_results_list = evaluate_metrics(
        model['data'], test_logits_t, test_confidence_sets_t, test_labels_t)
    
    # Combine results into a single row (DataFrame)
    trial_results_df = pd.concat(trial_results_list, axis=1)
    all_results.append(trial_results_df)
    logging.info('Trial %d: tau=%f, test_acc=%f, test_size=%f', t, tau, 
                 trial_results_df['accuracy'].iloc[0], trial_results_df['size'].iloc[0])

  # Concatenate results from all trials and calculate mean and std
  final_results_df = pd.concat(all_results, axis=0)
  results = {
      'mean': {'test': final_results_df.mean(0)},
      'std': {'test': final_results_df.std(0)},
  }
  return results