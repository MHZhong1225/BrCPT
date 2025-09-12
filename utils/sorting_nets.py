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

"""Module for constructing sorting networks."""
import numpy as np
jnp = np


def comm_pattern_from_list(snet_list, make_parallel=False):
  """A fixed network from a list of comperators."""
  if make_parallel:
    snet_list = parallelize(snet_list)
  total_stages = len(snet_list)
  edge_list = []
  max_wire_seen = 0
  comp_count = 0
  for a in snet_list:
    if not a: continue # Skip empty stages
    v = np.array(a)
    max_wire_seen = max(max_wire_seen, np.max(v))
    comp_count = comp_count + v.shape[0]
    idx = np.argsort(v[:, 0])
    edge_list.append(jnp.array(v[idx, :]))

  return {"alg": "fixed",
          "num_wires": max_wire_seen+1,
          "num_stages": total_stages,
          "num_comparators": comp_count,
          "edge_list": edge_list}


def parallelize(snet_lst):
  """Organize comparators that can be run in parallel in stages."""
  stage_sets = [set()]
  stage = [[]]
  
  # Flatten the list of stages into a single list of comparators
  all_comparators = [edge for stage_list in snet_lst for edge in stage_list]

  for edge in all_comparators:
    edge = tuple(sorted(edge)) # Ensure consistent edge representation
    placed = False
    for stage_idx in range(len(stage)):
      if (edge[0] not in stage_sets[stage_idx]) and \
         (edge[1] not in stage_sets[stage_idx]):
        stage[stage_idx].append(list(edge))
        stage_sets[stage_idx].update(edge)
        placed = True
        break
    if not placed:
      stage.append([list(edge)])
      stage_sets.append(set(edge))
      
  return [s for s in stage if s] # Return only non-empty stages


def generate_list_bitonic(length, make_parallel=True):
  """Generate a Bitonic sorting network list of arbitrary length."""
  
  snet_list = []
  
  def bitonic_sort(lo, n, direction):
    if n > 1:
      m = n // 2
      bitonic_sort(lo, m, not direction)
      bitonic_sort(lo + m, n - m, direction)
      bitonic_merge(lo, n, direction)

  def bitonic_merge(lo, n, direction):
    if n > 1:
      # Find the greatest power of two less than n
      m = 1
      while m < n:
          m <<= 1
      m >>= 1
      
      for i in range(lo, lo + n - m):
        comparator = [i, i + m]
        if not direction:
          comparator = comparator[::-1]
        # Append as a new stage, parallelize will group them later
        snet_list.append([comparator])

      bitonic_merge(lo, m, direction)
      bitonic_merge(lo + m, n - m, direction)

  bitonic_sort(0, length, True)
  return parallelize(snet_list) if make_parallel else snet_list


def comm_pattern_batcher(length, make_parallel=True):
  """Batcher bitonic communication pattern for an array with size length."""
  snet_list = generate_list_bitonic(length, make_parallel)
  comms = comm_pattern_from_list(snet_list)
  comms["alg"] = "batcher-bitonic"
  return comms