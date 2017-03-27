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
from __future__ import unicode_literals

import abc
from collections import namedtuple

import six
import tensorflow as tf
from tensorflow.python.util import nest  # pylint: disable=E0611

from seq2seq.graph_module import GraphModule
from seq2seq.configurable import Configurable
from seq2seq.contrib.seq2seq.decoder import Decoder, dynamic_decode
from seq2seq.encoders.rnn_encoder import _default_rnn_cell_params
from seq2seq.encoders.rnn_encoder import _toggle_dropout
from seq2seq.training import utils as training_utils


class DecoderOutput(
    namedtuple("DecoderOutput", ["logits", "predicted_ids", "cell_output"])):
  """Output of an RNN decoder.

  Note that we output both the logits and predictions because during
  dynamic decoding the predictions may not correspond to max(logits).
  For example, we may be sampling from the logits instead.
  """
  pass


@six.add_metaclass(abc.ABCMeta)
class RNNDecoder(Decoder, GraphModule, Configurable):
  """Base class for RNN decoders.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    name: A name for this module
  """

  def __init__(self, params, mode, name):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)
    self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)
    self.cell = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    # Not initialized yet
    self.initial_state = None
    self.helper = None

  @abc.abstractmethod
  def initialize(self, name=None):
    raise NotImplementedError

  @abc.abstractmethod
  def step(self, name=None):
    raise NotImplementedError

  @property
  def batch_size(self):
    return tf.shape(nest.flatten([self.initial_state])[0])[0]

  def _setup(self, initial_state, helper):
    """Sets the initial state and helper for the decoder.
    """
    self.initial_state = initial_state
    self.helper = helper

  def finalize(self, outputs, final_state):
    """Applies final transformation to the decoder output once decoding is
    finished.
    """
    #pylint: disable=R0201
    return (outputs, final_state)

  @staticmethod
  def default_params():
    return {
        "max_decode_length": 100,
        "rnn_cell": _default_rnn_cell_params(),
        "init_scale": 0.04,
    }

  def _build(self, initial_state, helper):
    if not self.initial_state:
      self._setup(initial_state, helper)

    scope = tf.get_variable_scope()
    scope.set_initializer(tf.random_uniform_initializer(
        -self.params["init_scale"],
        self.params["init_scale"]))

    maximum_iterations = None
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      maximum_iterations = self.params["max_decode_length"]

    outputs, final_state = dynamic_decode(
        decoder=self,
        output_time_major=True,
        impute_finished=False,
        maximum_iterations=maximum_iterations)
    return self.finalize(outputs, final_state)
