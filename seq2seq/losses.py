"""Operations related to calculating sequence losses.
"""

import tensorflow as tf

def sequence_mask(inputs, sequence_length):
  """Creates a mask of shape X according to sequence lengths.

  Args:
    inputs: A tensor of shape `[B, T]` for which a mask should be created
    sequence_length: An int32 tensor of shape `[B]` corresponding to the length of each input

  Returns:
    A boolean tensor with the same shape as `inputs` that is `False` past the
    sequence length of an input and `True` otherwise.

  Example:
    If `inputs` is of shape `[2, 3]` and sequence_length = `[1, 2]` this function returns
    `[[True, False, False], [True, True, False]]`.
  """
  with tf.name_scope("sequence_mask"):
    max_sequence_len = tf.shape(inputs)[1]
    sequence_range = tf.range(max_sequence_len, dtype=tf.int32)
    return  tf.map_fn(
      lambda length: tf.greater(length, sequence_range),
      elems=tf.to_int32(sequence_length),
      dtype=tf.bool)

def cross_entropy_sequence_loss(logits, targets, sequence_length):
  """Calculates the per-example Ccross-entropy loss for a sequence of logits and
    masks out all losses passed the sequence length.

  Args:
    logits: Logits of shape `[B, T, vocab_size]`
    targets: Target classes of shape `[B, T]`
    sequence_length: An int32 tensor of shape `[B]` corresponding to the length of each input

  Returns:
    A tensor of shape [B, T] that contains the loss per example, per time step.
  """
  with tf.name_scope("cross_entropy_sequence_loss"):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)

    # Mask out the losses we don't care about
    loss_mask = sequence_mask(targets, sequence_length)
    losses = losses * tf.to_float(loss_mask)

    return losses
