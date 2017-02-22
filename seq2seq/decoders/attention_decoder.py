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

from collections import namedtuple
import tensorflow as tf
from seq2seq.decoders.rnn_decoder import RNNDecoder

from seq2seq.contrib.seq2seq.helper import CustomHelper


class AttentionDecoderOutput(
    namedtuple(
        "DecoderOutput",
        ["logits", "predicted_ids", "cell_output",
         "attention_scores", "attention_context"])):
  """Augmented decoder output that also includes the attention scores.
  """
  pass


class AttentionDecoder(RNNDecoder):
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
               helper,
               initial_state,
               vocab_size,
               attention_inputs,
               attention_fn,
               max_decode_length,
               reverse_scores_lengths=None,
               attention_inputs_max_len=500,
               name="attention_decoder"):
    super(AttentionDecoder, self).__init__(
        cell, helper, initial_state, max_decode_length, name)
    self.vocab_size = vocab_size
    self.attention_inputs = attention_inputs
    self.attention_fn = attention_fn
    self.attention_inputs_max_len = attention_inputs_max_len
    self.reverse_scores_lengths = reverse_scores_lengths

    def att_next_inputs(time, outputs, state, sample_ids, name=None):
      finished, next_inputs, next_state = helper.next_inputs(
          time=time, outputs=outputs, state=state, sample_ids=sample_ids, name=name)
      next_inputs = tf.concat([next_inputs, outputs.attention_context], 1)
      return (finished, next_inputs, next_state)

    self.helper = CustomHelper(
        initialize_fn=helper.initialize,
        sample_fn=helper.sample,
        next_inputs_fn=att_next_inputs)


  @property
  def output_size(self):
    return AttentionDecoderOutput(
        logits=self.vocab_size,
        predicted_ids=tf.TensorShape([]),
        cell_output=self.cell.output_size,
        attention_scores=self.attention_inputs_max_len,
        attention_context=self.attention_inputs.get_shape()[2])

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
        self.attention_inputs.get_shape().as_list()[2]])
    first_inputs = tf.concat([first_inputs, attention_context], 1)

    return finished, first_inputs, self.initial_state

  def _build(self):
    outputs, final_state = super(AttentionDecoder, self)._build()

    # Slice attention scores to actual length
    source_len = tf.shape(self.attention_inputs)[1]
    outputs = outputs._replace(
        attention_scores=outputs.attention_scores[:, :, :source_len])
    return outputs, final_state

  # def transform_inputs(self, inputs, decoder_outputs):
  #   return tf.concat([inputs, decoder_outputs.attention_context], 1)

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

  def _pad_att_scores(self, scores):
    """Pads attention scores to fixed length. This is a hack because raw_rnn
    requirs a fully defined shape for all outputs."""
    # TODO: File a tensorflow bug and get rid of this hack
    max_len = self.attention_inputs_max_len
    scores = tf.pad(scores, [[0, 0], [0, max_len - tf.shape(scores)[1]]])
    scores.set_shape([None, max_len])
    return scores

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self.cell(inputs, state)
    cell_output_new, logits, att_scores, attention_context = \
      self.compute_output(cell_output)
    attention_scores = self._pad_att_scores(att_scores)

    if self.reverse_scores_lengths is not None:
      attention_scores = tf.reverse_sequence(
          input=attention_scores,
          seq_lengths=self.reverse_scores_lengths,
          seq_dim=1,
          batch_dim=0)

    sample_ids = self.helper.sample(
        time=time_,
        outputs=logits,
        state=cell_state)

    outputs = AttentionDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids,
        cell_output=cell_output_new,
        attention_scores=attention_scores,
        attention_context=attention_context)

    finished, next_inputs, next_state = self.helper.next_inputs(
        time=time_,
        outputs=outputs,
        state=cell_state,
        sample_ids=sample_ids)

    # next_inputs = self.transform_inputs(next_inputs, outputs)

    return (outputs, next_state, next_inputs, finished)
