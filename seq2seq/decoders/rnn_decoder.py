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


class RNNDecoder(GraphModule, Decoder):
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

  def _build(self):
    return dynamic_decode(
        decoder=self,
        output_time_major=True,
        impute_finished=False,
        maximum_iterations=self.max_decode_length)
