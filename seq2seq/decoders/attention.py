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
""" Implementations of attention layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six

import tensorflow as tf
from tensorflow.python.framework import function  # pylint: disable=E0611

from seq2seq.graph_module import GraphModule
from seq2seq.configurable import Configurable


@function.Defun(
    tf.float32,
    tf.float32,
    tf.float32,
    func_name="att_sum_bahdanau",
    noinline=True)
def att_sum_bahdanau(v_att, keys, query):
  """Calculates a batch- and timweise dot product with a variable"""
  return tf.reduce_sum(v_att * tf.tanh(keys + tf.expand_dims(query, 1)), [2])


@function.Defun(tf.float32, tf.float32, func_name="att_sum_dot", noinline=True)
def att_sum_dot(keys, query):
  """Calculates a batch- and timweise dot product"""
  return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])


@six.add_metaclass(abc.ABCMeta)
class AttentionLayer(GraphModule, Configurable):
  """
  Attention layer according to https://arxiv.org/abs/1409.0473.

  Params:
    num_units: Number of units used in the attention layer
  """

  def __init__(self, params, mode, name="attention"):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)

  @staticmethod
  def default_params():
    return {"num_units": 128}

  @abc.abstractmethod
  def score_fn(self, keys, query):
    """Computes the attention score"""
    raise NotImplementedError

  def _build(self, query, keys, values, values_length):
    """Computes attention scores and outputs.

    Args:
      query: The query used to calculate attention scores.
        In seq2seq this is typically the current state of the decoder.
        A tensor of shape `[B, ...]`
      keys: The keys used to calculate attention scores. In seq2seq, these
        are typically the outputs of the encoder and equivalent to `values`.
        A tensor of shape `[B, T, ...]` where each element in the `T`
        dimension corresponds to the key for that value.
      values: The elements to compute attention over. In seq2seq, this is
        typically the sequence of encoder outputs.
        A tensor of shape `[B, T, input_dim]`.
      values_length: An int32 tensor of shape `[B]` defining the sequence
        length of the attention values.

    Returns:
      A tuple `(scores, context)`.
      `scores` is vector of length `T` where each element is the
      normalized "score" of the corresponding `inputs` element.
      `context` is the final attention layer output corresponding to
      the weighted inputs.
      A tensor fo shape `[B, input_dim]`.
    """
    values_depth = values.get_shape().as_list()[-1]

    # Fully connected layers to transform both keys and query
    # into a tensor with `num_units` units
    att_keys = tf.contrib.layers.fully_connected(
        inputs=keys,
        num_outputs=self.params["num_units"],
        activation_fn=None,
        scope="att_keys")
    att_query = tf.contrib.layers.fully_connected(
        inputs=query,
        num_outputs=self.params["num_units"],
        activation_fn=None,
        scope="att_query")

    scores = self.score_fn(att_keys, att_query)

    # Replace all scores for padded inputs with tf.float32.min
    num_scores = tf.shape(scores)[1]
    scores_mask = tf.sequence_mask(
        lengths=tf.to_int32(values_length),
        maxlen=tf.to_int32(num_scores),
        dtype=tf.float32)
    scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)

    # Normalize the scores
    scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

    # Calculate the weighted average of the attention inputs
    # according to the scores
    context = tf.expand_dims(scores_normalized, 2) * values
    context = tf.reduce_sum(context, 1, name="context")
    context.set_shape([None, values_depth])


    return (scores_normalized, context)


class AttentionLayerDot(AttentionLayer):
  """An attention layer that calculates attention scores using
  a dot product.
  """

  def score_fn(self, keys, query):
    return att_sum_dot(keys, query)


class AttentionLayerBahdanau(AttentionLayer):
  """An attention layer that calculates attention scores using
  a parameterized multiplication."""

  def score_fn(self, keys, query):
    v_att = tf.get_variable(
        "v_att", shape=[self.params["num_units"]], dtype=tf.float32)
    return att_sum_bahdanau(v_att, keys, query)
