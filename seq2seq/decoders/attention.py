""" Implementations of attention layers.
"""

import tensorflow as tf
from seq2seq import GraphModule


class AttentionLayer(GraphModule):
  """
  Attention layer according to https://arxiv.org/abs/1409.0473.

  Args:
    num_units: Number of units used in the attention layer
    name: Name for this graph module
  """

  def __init__(self, num_units, name="attention"):
    super(AttentionLayer, self).__init__(name=name)
    self.num_units = num_units

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
    batch_size, inputs_timesteps, _ = tf.unpack(tf.shape(inputs))
    inputs_dim = inputs.get_shape().as_list()[-1]

    # Fully connected layers to transform both inputs and state
    # into a tensor with `num_units` units
    inputs_att = tf.contrib.layers.fully_connected(
        inputs=inputs,
        num_outputs=self.num_units,
        activation_fn=None,
        scope="inputs_att")
    state_att = tf.contrib.layers.fully_connected(
        inputs=state,
        num_outputs=self.num_units,
        activation_fn=None,
        scope="state_att")

    # Take the dot product of state for each time step in inputs
    # Result: A tensor of shape [B, T]
    inputs_att_flat = tf.reshape(inputs_att, [-1, self.num_units])
    state_att_flat = tf.reshape(
        tf.tile(state_att, [1, inputs_timesteps]),
        [inputs_timesteps * batch_size, self.num_units])
    scores = tf.matmul(
        tf.expand_dims(inputs_att_flat, 1), tf.expand_dims(state_att_flat, 2))
    scores = tf.reshape(scores, [batch_size, inputs_timesteps], name="scores")

    # Normalize the scores
    scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

    # Calculate the weighted average of the attention inputs
    # according to the scores
    context = tf.expand_dims(scores_normalized, 2) * inputs
    context = tf.reduce_sum(context, 1, name="context")
    context.set_shape([None, inputs_dim])

    return (scores_normalized, context)
