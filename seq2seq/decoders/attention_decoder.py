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
A basic sequence decoder that performs a softmax based on the RNN state.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import tensorflow as tf
from seq2seq.decoders.rnn_decoder import RNNDecoder

from seq2seq.contrib.seq2seq.helper import CustomHelper


class AttentionDecoderOutput(
    namedtuple("DecoderOutput", [
        "logits", "predicted_ids", "cell_output", "attention_scores",
        "attention_context"
    ])):
  """Augmented decoder output that also includes the attention scores.
  """
  pass


class AttentionDecoder(RNNDecoder):
  """An RNN Decoder that uses attention over an input sequence.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_keys: The sequence used to calculate attention scores.
      A tensor of shape `[B, T, ...]`.
    attention_values: The sequence to attend over.
      A tensor of shape `[B, T, input_dim]`.
    attention_values_length: Sequence length of the attention values.
      An int32 Tensor of shape `[B]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    reverse_scores: Optional, an array of sequence length. If set,
      reverse the attention scores in the output. This is used for when
      a reversed source sequence is fed as an input but you want to
      return the scores in non-reversed order.
  """

  def __init__(self,
               params,
               mode,
               vocab_size,
               attention_keys,
               attention_values,
               attention_values_length,
               attention_fn,
               reverse_scores_lengths=None,
               name="attention_decoder"):
    super(AttentionDecoder, self).__init__(params, mode, name)
    self.vocab_size = vocab_size
    self.attention_keys = attention_keys
    self.attention_values = attention_values
    self.attention_values_length = attention_values_length
    self.attention_fn = attention_fn
    self.reverse_scores_lengths = reverse_scores_lengths

  @property
  def output_size(self):
    return AttentionDecoderOutput(
        logits=self.vocab_size,
        predicted_ids=tf.TensorShape([]),
        cell_output=self.cell.output_size,
        attention_scores=tf.shape(self.attention_values)[1:-1],
        attention_context=self.attention_values.get_shape()[-1])

  @property
  def output_dtype(self):
    return AttentionDecoderOutput(
        logits=tf.float32,
        predicted_ids=tf.int32,
        cell_output=tf.float32,
        attention_scores=tf.float32,
        attention_context=tf.float32)

  def initialize(self, name=None):
    finished, first_inputs = self.helper.initialize()

    # Concat empty attention context
    attention_context = tf.zeros([
        tf.shape(first_inputs)[0],
        self.attention_values.get_shape().as_list()[-1]
    ])
    first_inputs = tf.concat([first_inputs, attention_context], 1)

    return finished, first_inputs, self.initial_state

  def compute_output(self, cell_output):
    """Computes the decoder outputs."""

    # Compute attention
    att_scores, attention_context = self.attention_fn(
        query=cell_output,
        keys=self.attention_keys,
        values=self.attention_values,
        values_length=self.attention_values_length)

    # TODO: Make this a parameter: We may or may not want this.
    # Transform attention context.
    # This makes the softmax smaller and allows us to synthesize information
    # between decoder state and attention context
    # see https://arxiv.org/abs/1508.04025v5
    softmax_input = tf.contrib.layers.fully_connected(
        inputs=tf.concat([cell_output, attention_context], 1),
        num_outputs=self.cell.output_size,
        activation_fn=tf.nn.tanh,
        scope="attention_mix")

    # Softmax computation
    logits = tf.contrib.layers.fully_connected(
        inputs=softmax_input,
        num_outputs=self.vocab_size,
        activation_fn=None,
        scope="logits")

    return softmax_input, logits, att_scores, attention_context

  def _setup(self, initial_state, helper):
    self.initial_state = initial_state

    def att_next_inputs(time, outputs, state, sample_ids, name=None):
      """Wraps the original decoder helper function to append the attention
      context.
      """
      finished, next_inputs, next_state = helper.next_inputs(
          time=time,
          outputs=outputs,
          state=state,
          sample_ids=sample_ids,
          name=name)
      next_inputs = tf.concat([next_inputs, outputs.attention_context], 1)
      return (finished, next_inputs, next_state)

    self.helper = CustomHelper(
        initialize_fn=helper.initialize,
        sample_fn=helper.sample,
        next_inputs_fn=att_next_inputs)

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self.cell(inputs, state)
    cell_output_new, logits, attention_scores, attention_context = \
      self.compute_output(cell_output)

    if self.reverse_scores_lengths is not None:
      attention_scores = tf.reverse_sequence(
          input=attention_scores,
          seq_lengths=self.reverse_scores_lengths,
          seq_dim=1,
          batch_dim=0)

    sample_ids = self.helper.sample(
        time=time_, outputs=logits, state=cell_state)

    outputs = AttentionDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids,
        cell_output=cell_output_new,
        attention_scores=attention_scores,
        attention_context=attention_context)

    finished, next_inputs, next_state = self.helper.next_inputs(
        time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

    return (outputs, next_state, next_inputs, finished)
