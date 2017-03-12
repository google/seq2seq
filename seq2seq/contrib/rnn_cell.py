# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collection of RNN Cells
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import inspect

import tensorflow as tf
from tensorflow.python.ops import array_ops  # pylint: disable=E0611
from tensorflow.python.util import nest  # pylint: disable=E0611
from tensorflow.contrib.rnn import MultiRNNCell  # pylint: disable=E0611

# Import all cell classes from Tensorflow
TF_CELL_CLASSES = [
    x for x in tf.contrib.rnn.__dict__.values()
    if inspect.isclass(x) and issubclass(x, tf.contrib.rnn.RNNCell)
]
for cell_class in TF_CELL_CLASSES:
  setattr(sys.modules[__name__], cell_class.__name__, cell_class)


class ExtendedMultiRNNCell(MultiRNNCell):
  """Extends the Tensorflow MultiRNNCell with residual connections"""

  def __init__(self,
               cells,
               residual_connections=False,
               residual_combiner="add",
               residual_dense=False):
    """Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.
      residual_connections: If true, add residual connections between all cells.
        This requires all cells to have the same output_size. Also, iff the
        input size is not equal to the cell output size, a linear transform
        is added before the first layer.
      residual_combiner: One of "add" or "concat". To create inputs for layer
        t+1 either "add" the inputs from the prev layer or concat them.
      residual_dense: Densely connect each layer to all other layers

    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    super(ExtendedMultiRNNCell, self).__init__(cells, state_is_tuple=True)
    assert residual_combiner in ["add", "concat", "mean"]

    self._residual_connections = residual_connections
    self._residual_combiner = residual_combiner
    self._residual_dense = residual_dense

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    if not self._residual_connections:
      return super(ExtendedMultiRNNCell, self).__call__(
          inputs, state, (scope or "extended_multi_rnn_cell"))

    with tf.variable_scope(scope or "extended_multi_rnn_cell"):
      # Adding Residual connections are only possible when input and output
      # sizes are equal. Optionally transform the initial inputs to
      # `cell[0].output_size`
      if self._cells[0].output_size != inputs.get_shape().as_list()[1] and \
          (self._residual_combiner in ["add", "mean"]):
        inputs = tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=self._cells[0].output_size,
            activation_fn=None,
            scope="input_transform")

      # Iterate through all layers (code from MultiRNNCell)
      cur_inp = inputs
      prev_inputs = [cur_inp]
      new_states = []
      for i, cell in enumerate(self._cells):
        with tf.variable_scope("cell_%d" % i):
          if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" %
                (len(self.state_size), state))
          cur_state = state[i]
          next_input, new_state = cell(cur_inp, cur_state)

          # Either combine all previous inputs or only the current input
          input_to_combine = prev_inputs[-1:]
          if self._residual_dense:
            input_to_combine = prev_inputs

          # Add Residual connection
          if self._residual_combiner == "add":
            next_input = next_input + sum(input_to_combine)
          if self._residual_combiner == "mean":
            combined_mean = tf.reduce_mean(tf.stack(input_to_combine), 0)
            next_input = next_input + combined_mean
          elif self._residual_combiner == "concat":
            next_input = tf.concat([next_input] + input_to_combine, 1)
          cur_inp = next_input
          prev_inputs.append(cur_inp)

          new_states.append(new_state)
    new_states = (tuple(new_states)
                  if self._state_is_tuple else array_ops.concat(new_states, 1))
    return cur_inp, new_states
