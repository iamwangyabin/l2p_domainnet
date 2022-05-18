# coding=utf-8
# Copyright 2020 The Learning-to-Prompt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific Learning-to-Prompt governing permissions and
# limitations under the License.
# ==============================================================================
"""Generate evaluation metrics for each task."""

from clu import metrics  # pylint: disable=unused-import
import flax  # pylint: disable=unused-import

# temporary workaround, # tasks could not exceed 1000 at the moment
# for i in range(1000):
#     exec(  # pylint: disable=exec-used
#         f"@flax.struct.dataclass\n"
#         f"class EvalMetrics_{i}(metrics.Collection):\n"
#         f"  accuracy_{i}: metrics.Accuracy\n"
#         f"  eval_loss_{i}: metrics.Average.from_output(\"loss\")")


# for binary use
from clu.metrics import Average, Metric
import jax.numpy as jnp
@flax.struct.dataclass
class Accuracy_b(Average):
  """Computes the accuracy from model outputs `logits` and `labels`.

  `labels` is expected to be of dtype=int32 and to have 0 <= ndim <= 2, and
  `logits` is expected to have ndim = labels.ndim + 1.

  See also documentation of `Metric`.
  """

  @classmethod
  def from_model_output(cls, *, logits: jnp.array, labels: jnp.array,
                        **kwargs) -> Metric:
    if logits.ndim != labels.ndim + 1 or labels.dtype != jnp.int32:
      raise ValueError(
          f"Expected labels.dtype==jnp.int32 and logits.ndim={logits.ndim}=="
          f"labels.ndim+1={labels.ndim + 1}")
    return super().from_model_output(
        values=(logits.argmax(axis=-1)%345 == labels%345).astype(jnp.float32), **kwargs)

for i in range(1000):
    exec(  # pylint: disable=exec-used
        f"@flax.struct.dataclass\n"
        f"class EvalMetrics_{i}(metrics.Collection):\n"
        f"  accuracy_{i}: Accuracy_b\n"
        f"  eval_loss_{i}: metrics.Average.from_output(\"loss\")")




EvalMetrics_list = [globals()[f"EvalMetrics_{i}"] for i in range(1000)]
