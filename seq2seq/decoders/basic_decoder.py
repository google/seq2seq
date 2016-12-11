"""
A basic sequence decoder that performs a softmax based on the RNN state.
"""

import tensorflow as tf
from seq2seq.decoders import DecoderBase, DecoderOutput, DecoderStepOutput


class BasicDecoder(DecoderBase):
  """Simple RNN decoder that performed a softmax operations on the cell output.

  Args:
    cell: An instance of ` tf.nn.rnn_cell.RNNCell`
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    max_decode_length: Maximum length for decoding steps for each example
      of shape `[B]`.
    prediction_fn: Optional. A function that generates a predictions
      of shape `[B]` from a logits of shape `[B, vocab_size]`.
      By default, this is argmax.
  """

  def __init__(self,
               cell,
               vocab_size,
               max_decode_length,
               prediction_fn=None,
               name="basic_decoder"):
    super(BasicDecoder, self).__init__(cell, max_decode_length, name)
    self.vocab_size = vocab_size
    self.prediction_fn = prediction_fn

    # By default, choose the highest logit score as the prediction
    if not prediction_fn:
      self.prediction_fn = lambda logits: tf.stop_gradient(tf.argmax(logits, 1))

  def _step(self, time_, cell_output, cell_state, loop_state, next_input_fn):
    initial_call = (cell_output is None)

    if initial_call:
      cell_output = tf.zeros([1, self.cell.output_size])

    logits = tf.contrib.layers.fully_connected(
        inputs=cell_output, num_outputs=self.vocab_size, activation_fn=None)

    if initial_call:
      outputs = DecoderOutput(
          logits=tf.zeros([self.vocab_size]),
          predictions=tf.zeros(
              [], dtype=tf.int64))
    else:
      predictions = self.prediction_fn(logits)
      outputs = DecoderOutput(logits, predictions)

    next_input = next_input_fn(time_, (None if initial_call else cell_output),
                               cell_state, loop_state, outputs)
    return DecoderStepOutput(
        outputs=outputs,
        next_input=next_input,
        next_cell_state=cell_state,
        next_loop_state=None)
