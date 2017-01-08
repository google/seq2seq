"""Collection of metrics for training and evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from tensorflow.contrib import metrics
from tensorflow.contrib.learn import metric_spec
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops

from seq2seq.training import utils as training_utils

def accumulate_strings(values, name="strings"):
  """Accumulates strings into a vector.

  Args:
    values: A 1-d string tensor that contains values to add to the accumulator.

  Returns:
    A tuple (value_tensor, update_op).
  """
  tf.assert_type(values, tf.string)
  strings = tf.Variable(
      name=name,
      initial_value=[],
      dtype=tf.string,
      trainable=False,
      collections=[tf.GraphKeys.LOCAL_VARIABLES],
      validate_shape=True)
  value_tensor = tf.identity(strings)
  update_op = tf.assign(
      ref=strings,
      value=tf.concat_v2([strings, values], 0),
      validate_shape=False)
  return value_tensor, update_op


def streaming_bleu(predictions,
                   labels,
                   eos_token="SEQUENCE_END",
                   lowercase=False,
                   metrics_collections=None,
                   updates_collections=None,
                   name=None):
  """Calculates BLEU scores by accumulating hypotheses and references in memory
  over multiple batches and calling the multi-bleu script at each step.

  Args:
    predictions: A tensor with token predictions from the model. Should be
      of shape `[batch, sequence_length]` and dtype string.
    labels: The expected target tokens. Same shape and type as
      `predictions`.
    eos_token: A string that marks the end of a sequence. All predictions
      and labels will be sliced until this string is found.
    lowercase: If set to true, evaluate lowercase BLEU. This is equivalent
      to passing the "-lc" flag to the multi-bleu.perl script.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    A tuple (bleu_value, update_op).
  """
  with variable_scope.variable_scope(name, "bleu_metric"):
    # Join tokens into single strings
    predictions_flat = tf.reduce_join(predictions, 1, separator=" ")
    labels_flat = tf.reduce_join(labels, 1, separator=" ")

    sources_value, sources_update = accumulate_strings(
        values=predictions_flat, name="sources")
    targets_value, targets_update = accumulate_strings(
        values=labels_flat, name="targets")

    bleu_value = tf.py_func(
        func=functools.partial(
            training_utils.moses_multi_bleu,
            eos_token=eos_token,
            lowercase=lowercase),
        inp=[sources_value, targets_value],
        Tout=tf.float32,
        name="value")

    with tf.control_dependencies([sources_update, targets_update]):
      update_op = tf.identity(bleu_value, name="update_op")

    if metrics_collections:
      ops.add_to_collections(metrics_collections, bleu_value)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return bleu_value, update_op


def make_bleu_metric_spec():
  """Creates a `MetricSpec` instance to calculate the BLEU
  score based on model predicitons and labels.
  """
  def bleu_wrapper(predictions, labels, **kwargs):
    """Remove the SEQUENCE_START token from the labels before
    feeding it into the bleu metric."""
    return streaming_bleu(predictions, labels[:, 1:], **kwargs)

  return metric_spec.MetricSpec(
      metric_fn=bleu_wrapper,
      label_key="target_tokens",
      prediction_key="predicted_tokens")


def streaming_log_perplexity():
  """Creates a MetricSpec that calculates the log perplexity.
  """

  def perplexity_metric(losses, target_len):
    """Calculates the mean log perplexity based on losses and target_len"""
    loss_mask = tf.sequence_mask(
        lengths=tf.to_int32(target_len - 1),
        maxlen=tf.to_int32(tf.shape(losses)[1]))
    return metrics.streaming_mean(losses, loss_mask)

  return metric_spec.MetricSpec(
      metric_fn=perplexity_metric,
      label_key="target_len",
      prediction_key="losses")
