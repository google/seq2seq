"""Operations related to calculating sequence losses.
"""

import tensorflow as tf


def cross_entropy_sequence_loss(logits, targets, sequence_length):
  """Calculates the per-example Ccross-entropy loss for a sequence of logits and
    masks out all losses passed the sequence length.

  Args:
    logits: Logits of shape `[B, T, vocab_size]`
    targets: Target classes of shape `[B, T]`
    sequence_length: An int32 tensor of shape `[B]` corresponding
      to the length of each input

  Returns:
    A tensor of shape [B, T] that contains the loss per example, per time step.
  """
  with tf.name_scope("cross_entropy_sequence_loss"):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)

    # Mask out the losses we don't care about
    loss_mask = tf.sequence_mask(
        tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[1]))
    losses = losses * tf.to_float(loss_mask)

    return losses
