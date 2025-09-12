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

"""Variational sorting networks."""
import functools
import jax
import jax.numpy as jnp


def _swap_prob_hard(x1, x2):
  return jnp.array(jnp.greater(x1, x2), dtype=jnp.float32)

_DELTA_THRESHOLD_EXPECTED = 0.001
_DELTA_THRESHOLD_SAMPLE = 0.001
_EPS = 1e-9


def _swap_prob_entropy_reg(x1, x2, dispersion=1.0):
  """Swapping probability, entropy regularization."""
  d = 2 * jax.nn.relu((x2-x1))/dispersion
  d2 = 2*jax.nn.relu((x1-x2))/dispersion
  return jnp.exp(d2 - jnp.logaddexp(d, d2))


def _swap_prob_entropy_reg_l2(x1, x2, dispersion=1.0):
  """Swapping probability, entropy regularization."""
  d = 2*jnp.square(jax.nn.relu(x2-x1))/dispersion
  d2 = 2*jnp.square(jax.nn.relu(x1-x2))/dispersion
  return jnp.exp(d2 - jnp.logaddexp(d, d2))


def _swap_prob_entropy_reg_lp(x1, x2, dispersion=1.0, norm_p=1.0):
  """Swapping probability, entropy regularization."""
  d = 2*jnp.power(jax.nn.relu(x2-x1), norm_p)/dispersion
  d2 = 2*jnp.power(jax.nn.relu(x1-x2), norm_p)/dispersion
  return jnp.exp(d2 - jnp.logaddexp(d, d2))


def butterfly(lam, x1, x2):
  return lam*x2+(1-lam)*x1, lam*x1+(1-lam)*x2


def forward_step(
    x,
    stage_idx,
    comms,
    dispersion=1.0,
    swap_prob_fun=_swap_prob_entropy_reg,
    hard_swap_prob_fun=_swap_prob_hard,
    key=None):
  """Computes swapping probabilities at stage_idx of the sorting network."""

  idx1 = comms["edge_list"][stage_idx][:, 0]
  idx2 = comms["edge_list"][stage_idx][:, 1]

  x1, x2 = butterfly(hard_swap_prob_fun(x[idx1], x[idx2]), x[idx1], x[idx2])
  if key is None:
    lam = swap_prob_fun(x[idx1], x[idx2], dispersion)
  else:
    subkey = jax.random.split(key, comms["edge_list"][stage_idx].shape[0])
    lam = swap_prob_fun(subkey, x[idx1], x[idx2], dispersion)

  x = x.at[idx1].set(x1, indices_are_sorted=True)
  x = x.at[idx2].set(x2, indices_are_sorted=True)

  return x, lam


def backward_step(u, stage_idx, comms, lam):
  """Executes in parallel stage_idx of the sorting network."""

  idx1 = comms["edge_list"][stage_idx][:, 0]
  idx2 = comms["edge_list"][stage_idx][:, 1]

  if len(u.shape) > 1:
    u1, u2 = butterfly(jnp.reshape(lam, (lam.shape[0], 1)),
                       u[idx1, :], u[idx2, :])
    u = u.at[idx1, :].set(u1, indices_are_sorted=True)
    u = u.at[idx2, :].set(u2, indices_are_sorted=True)
  else:
    u1, u2 = butterfly(lam, u[idx1], u[idx2])
    u = u.at[idx1].set(u1, indices_are_sorted=True)
    u = u.at[idx2].set(u2, indices_are_sorted=True)

  return u


def forward_only_step(
    x, v,
    stage_idx,
    comms,
    dispersion=1.0,
    swap_prob_fun=_swap_prob_entropy_reg,
    hard_swap_prob_fun=_swap_prob_hard,
    key=None):
  """Executes in parallel stage_idx of the sorting network."""

  idx1 = comms["edge_list"][stage_idx][:, 0]
  idx2 = comms["edge_list"][stage_idx][:, 1]

  x1, x2 = butterfly(hard_swap_prob_fun(x[idx1], x[idx2]), x[idx1], x[idx2])
  if key is None:
    lam = swap_prob_fun(x[idx1], x[idx2], dispersion)
  else:
    subkey = jax.random.split(key, comms["edge_list"][stage_idx].shape[0])
    lam = swap_prob_fun(subkey, x[idx1], x[idx2], dispersion)

  x = x.at[idx1].set(x1, indices_are_sorted=True)
  x = x.at[idx2].set(x2, indices_are_sorted=True)

  if len(v.shape) > 1:
    v1, v2 = butterfly(jnp.reshape(lam, (lam.shape[0], 1)),
                       v[idx1, :], v[idx2, :])
    v = v.at[idx1, :].set(v1, indices_are_sorted=True)
    v = v.at[idx2, :].set(v2, indices_are_sorted=True)
  else:
    v1, v2 = butterfly(lam, v[idx1], v[idx2])
    v = v.at[idx1].set(v1, indices_are_sorted=True)
    v = v.at[idx2].set(v2, indices_are_sorted=True)
  return x, v, lam


class VariationalSortingNet(object):
  """Class for efficient and differentiable order statistics."""

  def __init__(
      self, comms,
      smoothing_strategy="entropy_reg",
      sorting_strategy="hard",
      sorting_dispersion=0.001,
      norm_p=1):
    """
    Generate a sorting network that sort the input vector and values.
    """
    assert smoothing_strategy in ["entropy_reg"]
    assert sorting_strategy in ["hard", "entropy_reg"]
    assert norm_p > 0

    if norm_p == 1 or norm_p is None:
      norm_choice = 1
    elif norm_p == 2:
      norm_choice = 2
    else:
      norm_choice = 0

    self.comms = comms
    if smoothing_strategy == "entropy_reg":
      funcs = [functools.partial(_swap_prob_entropy_reg_lp, norm_p=norm_p),
               _swap_prob_entropy_reg,
               _swap_prob_entropy_reg_l2]
      swap_prob_fun = funcs[norm_choice]
      self._is_sampler = False

    if sorting_strategy == "hard":
      hard_swap_prob_fun = _swap_prob_hard
    elif sorting_strategy == "entropy_reg":
      hard_swap_prob_fun = functools.partial(
          _swap_prob_entropy_reg, dispersion=sorting_dispersion)

    if self._is_sampler:
      self.stage_fwd_only = functools.partial(
          forward_only_step, swap_prob_fun=swap_prob_fun,
          hard_swap_prob_fun=hard_swap_prob_fun)
      self.stage_fwd = functools.partial(
          forward_step, swap_prob_fun=swap_prob_fun,
          hard_swap_prob_fun=hard_swap_prob_fun)
    else:
      self.stage_fwd_only = functools.partial(
          forward_only_step, swap_prob_fun=swap_prob_fun,
          hard_swap_prob_fun=hard_swap_prob_fun, key=None)
      self.stage_fwd = functools.partial(
          forward_step, swap_prob_fun=swap_prob_fun,
          hard_swap_prob_fun=hard_swap_prob_fun, key=None)

  def forward_only(
      self, x, v, u=None, dispersion=1.,
      lower=0, upper=None, key=None):
    r"""Evaluate order statistics u^\top P(x) v by forward only propagation."""
    assert self.comms["num_wires"] == x.shape[0]

    if upper is None:
      upper = self.comms["num_stages"]

    if not self._is_sampler:
      for i in range(lower, upper):
        x, v, _ = self.stage_fwd_only(x, v, i,
                                      self.comms, dispersion=dispersion)
    else:
      subkey = jax.random.split(key, upper-lower)
      for i in range(lower, upper):
        x, v, _ = self.stage_fwd_only(x, v, i,
                                      self.comms,
                                      dispersion=dispersion,
                                      key=subkey[i])

    if u is None:
      return x, v
    else:
      return x, u.T.dot(v)

  def quantile(self, x, dispersion, alpha=0.5, tau=0.5, key=None):
    """Retrieves the smoothed alpha quantile."""
    length = self.comms["num_wires"]
    idx1 = jnp.floor(alpha * (length-1)).astype(int)
    idx2 = jnp.ceil(alpha * (length-1)).astype(int)
    
    # Create a one-hot vector for interpolation
    u = tau * jax.nn.one_hot(idx2, length)
    u += (1 - tau) * jax.nn.one_hot(idx1, length)
    
    _, x_ss = self.forward_only(x, x, u=u, dispersion=dispersion, key=key)
    return x_ss