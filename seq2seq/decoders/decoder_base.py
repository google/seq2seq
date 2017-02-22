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
Base class for sequence decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf
from tensorflow.python.util import nest

from seq2seq.graph_module import GraphModule
from seq2seq.contrib.seq2seq.decoder import Decoder, dynamic_decode


class DecoderOutput(namedtuple(
    "DecoderOutput", ["logits", "predicted_ids", "cell_output"])):
  """Output of a decoder.

  Note that we output both the logits and predictions because during
  dynamic decoding the predictions may not correspond to max(logits).
  For example, we may be sampling from the logits instead.
  """
  pass


# class DecoderInputs(GraphModule):
#   """Abstract base class for decoder input feeding.
#   """

#   def __init__(self, name):
#     super(DecoderInputs, self).__init__(name)

#   def _build(self, time_, initial_call, predicted_ids):
#     """Returns the input for the given time step.

#     Args:
#       time_: An int32 scalar
#       initial_call: True iff this is the first time step.
#       predicted_ids: The predictions of the decoder. An int32 1-d tensor.

#     Returns:
#       A tuple of tensors (next_input, finished) where next_input
#       is a  a tensor of shape `[B, ...]` and  finished is a boolean tensor
#       of shape `[B]`. When `time_` is past the maximum
#       sequence length a zero tensor is fed as input for performance purposes.
#     """
#     raise NotImplementedError

# class FixedDecoderInputs(DecoderInputs):
#   """An operation that feeds fixed inputs to a decoder,
#   also known as "teacher forcing".

#   Args:
#     inputs: The inputs to feed to the decoder.
#       A tensor of shape `[B, T, ...]`. At each time step T, one slice
#       of shape `[B, ...]` is fed to the decoder.
#     sequence_length: A tensor of shape `[B]` that specifies the
#       sequence length for each example.

#   """

#   def __init__(self, inputs, sequence_length, name="fixed_decoder_inputs"):
#     super(FixedDecoderInputs, self).__init__(name)
#     self.inputs = inputs
#     self.sequence_length = sequence_length

#     with self.variable_scope():
#       self.inputs_ta = tf.TensorArray(
#           dtype=self.inputs.dtype,
#           size=tf.shape(self.inputs)[1],
#           name="inputs_ta")
#       self.inputs_ta = self.inputs_ta.unstack(
#           tf.transpose(self.inputs, [1, 0, 2]))
#       self.max_seq_len = tf.reduce_max(sequence_length, name="max_seq_len")
#       self.batch_size = tf.identity(tf.shape(inputs)[0], name="batch_size")
#       self.input_dim = tf.identity(tf.shape(inputs)[-1], name="input_dim")

#   def _build(self, time_, initial_call, predictions):
#     all_finished = (time_ >= self.max_seq_len)
#     next_input = tf.cond(
#         all_finished,
#         lambda: tf.zeros([self.batch_size, self.input_dim], dtype=tf.float32),
#         lambda: self.inputs_ta.read(time_))
#     next_input.set_shape([None, self.inputs.get_shape().as_list()[-1]])
#     return next_input, (time_ >= self.sequence_length)


# class DynamicDecoderInputs(DecoderInputs):
#   """An operation that feeds dynamic inputs to a decoder according to some
#   arbitrary function that creates a new input from the decoder output at
#   the current step, e.g. `embed(argmax(logits))`.

#   Args:
#     initial_inputs: An input to feed at the first time step.
#       A tensor of shape `[B, ...]`.
#     make_input_fn: A function that mapes from `predictions -> next_input`,
#       where `next_input` must be a Tensor of shape `[B, ...]`.
#     max_decode_length: Decode to at most this length
#     elements_finished_fn: A function that maps from (time_, predictions) =>
#       a boolean vector of shape `[B]` used for early stopping.
#   """

#   def __init__(self, initial_inputs, make_input_fn,
#                max_decode_length,
#                elements_finished_fn=None,
#                name="fixed_decoder_inputs"):
#     super(DynamicDecoderInputs, self).__init__(name)
#     self.initial_inputs = initial_inputs
#     self.make_input_fn = make_input_fn
#     self.max_decode_length = max_decode_length
#     self.elements_finished_fn = elements_finished_fn
#     self.batch_size = tf.shape(self.initial_inputs)[0]

#   def _build(self, time_, initial_call, predictions):
#     if initial_call:
#       next_input = self.initial_inputs
#       elements_finished = tf.zeros([self.batch_size], dtype=tf.bool)
#     else:
#       batch_size = tf.shape(nest.flatten(predictions)[0])[0]
#       next_input = self.make_input_fn(predictions)
#       max_decode_length_batch = tf.cast(
#           tf.ones([batch_size]) * self.max_decode_length,
#           dtype=time_.dtype)
#       elements_finished = (time_ >= max_decode_length_batch)
#       if self.elements_finished_fn:
#         elements_finished = tf.logical_or(
#             elements_finished, self.elements_finished_fn(time_, predictions))
#     return next_input, elements_finished


class DecoderBase(GraphModule, Decoder):
  """Base class for RNN decoders.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    name: A name for this module
    input_fn: A function that generates the next input, e.g. an
      instance of `FixedDecoderInputs` or `DynamicDecoderInputs`.
  """

  def __init__(self, cell, helper, initial_state, max_decode_length, name):
    GraphModule.__init__(self, name)
    self.cell = cell
    self.max_decode_length = max_decode_length
    self.helper = helper
    self.initial_state = initial_state

  @property
  def batch_size(self):
    return tf.shape(nest.flatten([self.initial_state])[0])[0]

  def transform_inputs(self, inputs, decoder_outputs):
    return inputs

  def _build(self):
    return dynamic_decode(
        decoder=self,
        output_time_major=True,
        impute_finished=True,
        maximum_iterations=self.max_decode_length)
