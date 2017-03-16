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

import tensorflow as tf
from seq2seq.decoders.rnn_decoder import RNNDecoder, DecoderOutput


class BasicDecoder(RNNDecoder):
  """Simple RNN decoder that performed a softmax operations on the cell output.
  """

  def __init__(self, params, mode, vocab_size, name="basic_decoder"):
    super(BasicDecoder, self).__init__(params, mode, name)
    self.vocab_size = vocab_size

  def compute_output(self, cell_output):
    """Computes the decoder outputs."""
    return tf.contrib.layers.fully_connected(
        inputs=cell_output, num_outputs=self.vocab_size, activation_fn=None)

  @property
  def output_size(self):
    return DecoderOutput(
        logits=self.vocab_size,
        predicted_ids=tf.TensorShape([]),
        cell_output=self.cell.output_size)

  @property
  def output_dtype(self):
    return DecoderOutput(
        logits=tf.float32, predicted_ids=tf.int32, cell_output=tf.float32)

  def initialize(self, name=None):
    finished, first_inputs = self.helper.initialize()
    return finished, first_inputs, self.initial_state

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self.cell(inputs, state)
    logits = self.compute_output(cell_output)
    sample_ids = self.helper.sample(
        time=time_, outputs=logits, state=cell_state)
    outputs = DecoderOutput(
        logits=logits, predicted_ids=sample_ids, cell_output=cell_output)
    finished, next_inputs, next_state = self.helper.next_inputs(
        time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)
    return (outputs, next_state, next_inputs, finished)
