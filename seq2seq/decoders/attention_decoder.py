"""
A basic sequence decoder that performs a softmax based on the RNN state.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
from seq2seq.decoders import DecoderBase, DecoderStepOutput


class AttentionDecoderOutput(
    namedtuple(
        "DecoderOutput",
        ["logits", "predicted_ids", "attention_scores", "attention_context"])):
  """Augmented decoder output that also includes the attention scores.
  """
  pass


class AttentionDecoder(DecoderBase):
  """An RNN Decoder that uses attention over an input sequence.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_inputs: The sequence to take attentio over.
      A tensor of shaoe `[B, T, ...]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    max_decode_length: Maximum length for decoding steps
      for each example of shape `[B]`.
    reverse_scores: Optional, an array of sequence length. If set,
      reverse the attention scores in the output. This is used for when
      a reversed source sequence is fed as an input but you want to
      return the scores in non-reversed order.
    prediction_fn: Optional. A function that generates a predictions
      of shape `[B]` from a logits of shape `[B, vocab_size]`.
      By default, this is argmax.
  """

  def __init__(self,
               cell,
               input_fn,
               vocab_size,
               attention_inputs,
               attention_fn,
               max_decode_length,
               reverse_scores_lengths=None,
               attention_inputs_max_len=500,
               prediction_fn=None,
               name="attention_decoder"):
    super(AttentionDecoder, self).__init__(
        cell, input_fn, max_decode_length, prediction_fn, name)
    self.vocab_size = vocab_size
    self.attention_inputs = attention_inputs
    self.attention_fn = attention_fn
    self.attention_inputs_max_len = attention_inputs_max_len
    self.reverse_scores_lengths = reverse_scores_lengths

  def pack_outputs(self, outputs_ta, final_loop_state):
    logits, predicted_ids = DecoderBase.pack_outputs(self, outputs_ta,
                                                     final_loop_state)

    attention_scores = outputs_ta.attention_scores.stack()
    attention_input_len = tf.shape(self.attention_inputs)[1]

    # Slice attention scores to actual length of the inputs
    attention_scores = attention_scores[:, :, :attention_input_len]

    if self.reverse_scores_lengths is not None:
      attention_scores = tf.reverse_sequence(
          input=attention_scores,
          seq_lengths=self.reverse_scores_lengths,
          seq_dim=2,
          batch_dim=1)

    attention_context = outputs_ta.attention_context.stack()
    return AttentionDecoderOutput(logits, predicted_ids, attention_scores,
                                  attention_context)

  def compute_output(self, cell_output):
    # Compute attention
    att_scores, attention_context = self.attention_fn(
        cell_output, self.attention_inputs)

    # TODO: Make this a parameter: We may or may not want this.
    # Transform attention context.
    # This makes the softmax smaller and allows us to synthesize information
    # between decoder state and attention context
    # see https://arxiv.org/abs/1508.04025v5
    softmax_input = tf.contrib.layers.fully_connected(
        inputs=tf.concat_v2([cell_output, attention_context], 1),
        num_outputs=self.cell.output_size,
        activation_fn=tf.nn.tanh,
        scope="attention_mix")

    # Softmax computation
    logits = tf.contrib.layers.fully_connected(
        inputs=softmax_input,
        num_outputs=self.vocab_size,
        activation_fn=None,
        scope="logits")

    return logits, att_scores, attention_context

  def output_shapes(self):
    return AttentionDecoderOutput(
        logits=tf.zeros([self.vocab_size]),
        predicted_ids=tf.zeros([], dtype=tf.int64),
        attention_scores=tf.zeros([self.attention_inputs_max_len]),
        attention_context=tf.zeros([self.attention_inputs.get_shape()[2]]))

  def create_next_input(self, time_, initial_call, output):
    next_input, elements_finished = self.input_fn(
        time_, initial_call, output.predicted_ids)
    if initial_call:
      attention_context = tf.zeros([
          tf.shape(next_input)[0],
          self.attention_inputs.get_shape().as_list()[2]
      ])
    else:
      attention_context = output.attention_context

    next_input = tf.concat_v2([next_input, attention_context], 1)
    return next_input, elements_finished

  def _pad_att_scores(self, scores):
    """Pads attention scores to fixed length. This is a hack because raw_rnn
    requirs a fully defined shape for all outputs."""
    # TODO: File a tensorflow bug and get rid of this hack
    max_len = self.attention_inputs_max_len
    scores = tf.pad(scores, [[0, 0], [0, max_len - tf.shape(scores)[1]]])
    scores.set_shape([None, max_len])
    return scores

  def step(self, time_, cell_output, cell_state, loop_state):
    initial_call = (cell_output is None)

    if initial_call:
      outputs = self.output_shapes()
      cell_output = tf.zeros(
          [tf.shape(self.attention_inputs)[0], self.cell.output_size])
      _, _, attention_context = self.compute_output(cell_output)
      predicted_ids = None
    else:
      logits, att_scores, attention_context = self.compute_output(cell_output)
      attention_scores = self._pad_att_scores(att_scores)
      predicted_ids = self.prediction_fn(logits)
      outputs = AttentionDecoderOutput(
          logits=logits,
          predicted_ids=predicted_ids,
          attention_scores=attention_scores,
          attention_context=attention_context)

    return DecoderStepOutput(
        outputs=outputs,
        next_cell_state=cell_state,
        next_loop_state=None)
