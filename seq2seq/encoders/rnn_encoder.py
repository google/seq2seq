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

"""
Collection of RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn

from seq2seq.graph_module import GraphModule

RNNEncoderOutput = collections.namedtuple("RNNEncoderOutput",
                                          ["outputs", "final_state"])


class UnidirectionalRNNEncoder(GraphModule):
  """
  A unidirectional RNN encoder. Stacking should be performed as
  part of the cell.

  Args:
    cell: An instance of tf.contrib.rnn.RNNCell
    name: A name for the encoder
  """

  def __init__(self, cell_fn, name="forward_rnn_encoder"):
    super(UnidirectionalRNNEncoder, self).__init__(name)
    self.cell_fn = cell_fn

  def _build(self, inputs, sequence_length, **kwargs):
    outputs, state = tf.nn.dynamic_rnn(
        cell=self.cell_fn(),
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        **kwargs)
    return RNNEncoderOutput(outputs=outputs, final_state=state)


class BidirectionalRNNEncoder(GraphModule):
  """
  A bidirectional RNN encoder. Uses the same cell for both the
  forward and backward RNN. Stacking should be performed as part of
  the cell.

  Args:
    cell: An instance of tf.contrib.rnn.RNNCell
    name: A name for the encoder
  """

  def __init__(self, cell_fn, name="bidi_rnn_encoder"):
    super(BidirectionalRNNEncoder, self).__init__(name)
    self.cell_fn = cell_fn

  def _build(self, inputs, sequence_length, **kwargs):
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=self.cell_fn(),
        cell_bw=self.cell_fn(),
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        **kwargs)

    # Concatenate outputs and states of the forward and backward RNNs
    outputs_concat = tf.concat(outputs, 2)

    return RNNEncoderOutput(outputs=outputs_concat, final_state=states)


class StackBidirectionalRNNEncoder(GraphModule):
  """
  A stacked bidirectional RNN encoder. Uses the same cell for both the
  forward and backward RNN. Stacking should be performed as part of
  the cell.

  Args:
    cell: An instance of tf.contrib.rnn.RNNCell
    name: A name for the encoder
  """

  def __init__(self, cell_fn, name="stacked_bidi_rnn_encoder"):
    super(StackBidirectionalRNNEncoder, self).__init__(name)
    self.cell_fn = cell_fn

  def _unpack_cell(self, cell):
    """Unpack the cells because the stack_bidirectional_dynamic_rnn
    expects a list of cells, one per layer."""
    if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
      return cell._cells #pylint: disable=W0212
    else:
      return [cell]

  def _build(self, inputs, sequence_length, **kwargs):

    fw_cell = self._unpack_cell(self.cell_fn())
    bw_cell = self._unpack_cell(self.cell_fn())

    result = rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=fw_cell,
        cells_bw=bw_cell,
        inputs=inputs,
        dtype=tf.float32,
        sequence_length=sequence_length,
        **kwargs)
    outputs_concat, _output_state_fw, _output_state_bw = result
    final_state = (_output_state_fw, _output_state_bw)
    return RNNEncoderOutput(outputs=outputs_concat, final_state=final_state)
