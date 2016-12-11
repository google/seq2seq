"""
A basic sequence decoder that performs a softmax based on the RNN state.
"""

from collections import namedtuple
import tensorflow as tf
from seq2seq.decoders import DecoderBase, DecoderOutput, DecoderStepOutput


class AttentionDecoderOutput(
    namedtuple("DecoderOutput", ["logits", "predictions", "attention_scores"])):
  """Augmented decoder output that also includes the attention scores.
  """
  pass


class AttentionDecoder(DecoderBase):
  """An RNN Decoder that uses attention over an input sequence.

  Args:
    cell: An instance of ` tf.nn.rnn_cell.RNNCell`
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_inputs: The sequence to take attentio over.
      A tensor of shaoe `[B, T, ...]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    max_decode_length: Maximum length for decoding steps
      for each example of shape `[B]`.
    prediction_fn: Optional. A function that generates a predictions
      of shape `[B]` from a logits of shape `[B, vocab_size]`.
      By default, this is argmax.
  """

  def __init__(self,
               cell,
               vocab_size,
               attention_inputs,
               attention_fn,
               max_decode_length,
               prediction_fn=None,
               name="attention_decoder"):
    super(AttentionDecoder, self).__init__(cell, max_decode_length, name)
    self.vocab_size = vocab_size
    self.prediction_fn = prediction_fn
    self.attention_inputs = attention_inputs
    self.attention_fn = attention_fn

    # By default, choose the highest logit score as the prediction
    if not prediction_fn:
      self.prediction_fn = lambda logits: tf.stop_gradient(tf.argmax(logits, 1))

  @staticmethod
  def _pack_outputs(outputs_ta, final_loop_state):
    logits, predictions = DecoderBase._pack_outputs(outputs_ta,
                                                    final_loop_state)
    attention_scores = tf.transpose(final_loop_state.pack(), [1, 0, 2])
    return AttentionDecoderOutput(logits, predictions, attention_scores)

  def _step(self, time_, cell_output, cell_state, loop_state, next_input_fn):
    initial_call = (cell_output is None)

    if initial_call:
      cell_output = tf.zeros(
          [tf.shape(self.attention_inputs)[0], self.cell.output_size])
      # Initialize the TensorArray that will hold the attention scores
      next_loop_state = tf.TensorArray(
          dtype=tf.float32, size=1, dynamic_size=True)

    # Compute attention
    att_scores, attention_context = self.attention_fn(cell_output,
                                                      self.attention_inputs)

    # In the first step the attention vector is set to all zeros
    if initial_call:
      attention_context = tf.zeros_like(attention_context)
    else:
      next_loop_state = loop_state.write(time_ - 1, att_scores)

    # Softmax computation
    softmax_input = tf.concat(1, [cell_output, attention_context])
    logits = tf.contrib.layers.fully_connected(
        inputs=softmax_input,
        num_outputs=self.vocab_size,
        activation_fn=None,
        scope="logits")
    predictions = self.prediction_fn(logits)
    outputs = DecoderOutput(logits, predictions)

    if initial_call:
      outputs = DecoderOutput(
          logits=tf.zeros([self.vocab_size]),
          predictions=tf.zeros(
              [], dtype=tf.int64))

    # Append the attention context to the inputs
    next_input = next_input_fn(time_, (None if initial_call else cell_output),
                               cell_state, loop_state, outputs)
    next_input = tf.concat(1, [next_input, attention_context])

    return DecoderStepOutput(
        outputs=outputs,
        next_input=next_input,
        next_cell_state=cell_state,
        next_loop_state=next_loop_state)
