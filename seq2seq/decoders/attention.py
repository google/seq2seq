""" Implementations of attention layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seq2seq.graph_module import GraphModule
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.ops import math_ops

@function.Defun(
    tf.float32, tf.float32, tf.float32,
    func_name="att_sum_bahdanau",
    noinline=True)
def att_sum_bahdanau(v_att, keys, query):
  """Calculates a batch- and timweise dot product with a variable"""
  return tf.reduce_sum(
      v_att * math_ops.tanh(keys + tf.expand_dims(query, 1)), [2])

@function.Defun(tf.float32, tf.float32, func_name="att_sum_dot", noinline=True)
def att_sum_dot(keys, query):
  """Calculates a batch- and timweise dot product"""
  return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])


class AttentionLayer(GraphModule):
  """
  Attention layer according to https://arxiv.org/abs/1409.0473.

  Args:
    num_units: Number of units used in the attention layer
    name: Name for this graph module
  """

  def __init__(self, num_units, score_type="bahdanau", name="attention"):
    super(AttentionLayer, self).__init__(name=name)
    self.num_units = num_units

    score_fn_name = "_{}_score".format(score_type)
    if not hasattr(self, score_fn_name):
      raise ValueError("Invalid attention score type: " + score_type)
    self.score_fn = getattr(self, score_fn_name)

  def _bahdanau_score(self, keys, query):
    """Computes Bahdanau-style attention scores.
    """
    v_att = tf.get_variable("v_att", shape=[self.num_units], dtype=tf.float32)
    return att_sum_bahdanau(v_att, keys, query)

  def _dot_score(self, keys, query):
    """Computes Bahdanau-style attention scores.
    """
    return att_sum_dot(keys, query)

  def _build(self, state, inputs):
    """Computes attention scores and outputs.

    Args:
      state: The state based on which to calculate attention scores.
        In seq2seq this is typically the current state of the decoder.
        A tensor of shape `[B, ...]`
      inputs: The elements to compute attention *over*. In seq2seq this is
        typically the sequence of encoder outputs.
        A tensor of shape `[B, T, input_dim]`

    Returns:
      A tuple `(scores, context)`.
      `scores` is vector of length `T` where each element is the
      normalized "score" of the corresponding `inputs` element.
      `context` is the final attention layer output corresponding to
      the weighted inputs.
      A tensor fo shape `[B, input_dim]`.
    """
    inputs_dim = inputs.get_shape().as_list()[-1]

    # Fully connected layers to transform both inputs and state
    # into a tensor with `num_units` units
    att_keys = tf.contrib.layers.fully_connected(
        inputs=inputs,
        num_outputs=self.num_units,
        activation_fn=None,
        scope="att_keys")
    att_query = tf.contrib.layers.fully_connected(
        inputs=state,
        num_outputs=self.num_units,
        activation_fn=None,
        scope="att_query")

    scores = self.score_fn(att_keys, att_query)

    # Show, Attend, Spell type of attention
    # Take the dot product of state for each time step in inputs
    # Result: A tensor of shape [B, T]
    # att_keys_flat = tf.reshape(att_keys, [-1, self.num_units])
    # att_query_flat = tf.reshape(
    #     tf.tile(att_query, [1, inputs_timesteps]),
    #     [inputs_timesteps * batch_size, self.num_units])
    # scores = tf.matmul(
    #     tf.expand_dims(att_keys_flat, 1), tf.expand_dims(att_query_flat, 2))
    # scores = tf.reshape(scores, [batch_size, inputs_timesteps], name="scores")

    # Normalize the scores
    scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

    # Calculate the weighted average of the attention inputs
    # according to the scores
    context = tf.expand_dims(scores_normalized, 2) * inputs
    context = tf.reduce_sum(context, 1, name="context")
    context.set_shape([None, inputs_dim])

    return (scores_normalized, context)
