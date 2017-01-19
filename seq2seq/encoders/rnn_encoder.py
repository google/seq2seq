"""
Collection of RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
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

  def __init__(self, cell, name="forward_rnn_encoder"):
    super(UnidirectionalRNNEncoder, self).__init__(name)
    self.cell = cell

  def _build(self, inputs, sequence_length, **kwargs):
    outputs, state = tf.nn.dynamic_rnn(
        cell=self.cell,
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

  def __init__(self, cell, name="bidi_rnn_encoder"):
    super(BidirectionalRNNEncoder, self).__init__(name)
    self.cell = cell

  def _build(self, inputs, sequence_length, **kwargs):
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=self.cell,
        cell_bw=self.cell,
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        **kwargs)

    # Concatenate outputs and states of the forward and backward RNNs
    outputs_concat = tf.concat_v2(outputs, 2)

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

  def __init__(self, cell, name="stacked_bidi_rnn_encoder"):
    super(StackBidirectionalRNNEncoder, self).__init__(name)
    self.cell = cell

  def _build(self, inputs, sequence_length, **kwargs):
    # "Unpack" the cells because the stack_bidirectional_dynamic_rnn
    # expects a list of cells, one per layer.
    if isinstance(self.cell, tf.contrib.rnn.MultiRNNCell):
      cells = self.cell._cells #pylint: disable=W0212
    else:
      cells = [self.cell]

    result = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells,
        cells_bw=cells,
        inputs=inputs,
        dtype=tf.float32,
        sequence_length=sequence_length,
        **kwargs)
    outputs_concat, _output_state_fw, _output_state_bw = result
    final_state = (_output_state_fw, _output_state_bw)
    return RNNEncoderOutput(outputs=outputs_concat, final_state=final_state)
