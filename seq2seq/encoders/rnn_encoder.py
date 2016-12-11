"""
Collection of RNN encoders.
"""

import collections
import tensorflow as tf
from seq2seq import GraphModule

RNNEncoderOutput = collections.namedtuple("RNNEncoderOutput",
                                          ["outputs", "final_state"])


class UnidirectionalRNNEncoder(GraphModule):
  """
  A unidirectional RNN encoder. Stacking should be performed as
  part of the cell.

  Args:
    cell: An instance of tf.nn.rnn_cell.RNNCell
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
    cell: An instance of tf.nn.rnn_cell.RNNCell
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
    outputs_concat = tf.concat(2, outputs)

    return RNNEncoderOutput(outputs=outputs_concat, final_state=states)
