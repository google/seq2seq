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

import tensorflow as tf
from seq2seq.decoders import DecoderBase, DecoderOutput, DecoderStepOutput


class BasicDecoder(DecoderBase):
  """Simple RNN decoder that performed a softmax operations on the cell output.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
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
               input_fn,
               vocab_size,
               max_decode_length,
               prediction_fn=None,
               name="basic_decoder"):
    super(BasicDecoder, self).__init__(
        cell, input_fn, max_decode_length, prediction_fn, name)
    self.vocab_size = vocab_size

  def compute_output(self, cell_output):
    return tf.contrib.layers.fully_connected(
        inputs=cell_output,
        num_outputs=self.vocab_size,
        activation_fn=None)

  def output_shapes(self):
    return DecoderOutput(
        logits=tf.zeros([self.vocab_size]),
        predicted_ids=tf.zeros([], dtype=tf.int64),
        cell_output=tf.zeros([self.cell.output_size]))

  def step(self, time_, cell_output, cell_state, loop_state):
    initial_call = (cell_output is None)

    if initial_call:
      outputs = self.output_shapes()
      predicted_ids = None
      # We need to call this here to create variables
      cell_output = tf.zeros([1, self.cell.output_size])
      self.compute_output(cell_output)
    else:
      logits = self.compute_output(cell_output)
      predicted_ids = self.prediction_fn(logits)
      outputs = DecoderOutput(
          logits=logits,
          predicted_ids=predicted_ids,
          cell_output=cell_output)

    return DecoderStepOutput(
        outputs=outputs,
        next_cell_state=cell_state,
        next_loop_state=None)
